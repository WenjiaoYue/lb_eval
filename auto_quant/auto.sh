#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    CYAN=''
    BOLD=''
    NC=''
fi

log_info() { echo -e "${CYAN}[auto.sh]${NC} $*"; }
log_ok() { echo -e "${GREEN}[auto.sh]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[auto.sh]${NC} $*"; }
log_error() { echo -e "${RED}[auto.sh]${NC} $*"; }
log_step() { echo -e "\n${BOLD}${CYAN}========== $* ==========${NC}\n"; }

usage() {
    cat <<'EOF'
Usage:
  bash auto.sh <task_json_file> [options]

Options:
  --skip-upload   Skip all uploads
  --skip-hf       Skip HuggingFace model upload
  --skip-github   Skip GitHub results upload
  --dry-run       Print resolved configuration and exit
  -h, --help      Show this help
EOF
}

require_command() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        log_error "Required command not found: $cmd"
        exit 1
    fi
}

resolve_first_existing_dir() {
    local candidate
    for candidate in "$@"; do
        if [[ -n "$candidate" && -d "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

resolve_first_existing_file() {
    local candidate
    for candidate in "$@"; do
        if [[ -n "$candidate" && -f "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

resolve_skill_path() {
    local skill_name="$1"
    resolve_first_existing_file \
        "${OPENCLAW_WORKSPACE_DIR}/skills/${skill_name}/SKILL.md" \
        "/root/leaderboard_Agent/tasks/lb_eval/openclaw_config/workspace/skills/${skill_name}/SKILL.md" \
        "/root/leaderboard_Agent/auto_quant/openclaw_home/workspace/skills/${skill_name}/SKILL.md" \
        "/root/leaderboard_Agent/auto_eval/openclaw_home/workspace/skills/${skill_name}/SKILL.md" \
        "/root/.openclaw/workspace/skills/${skill_name}/SKILL.md"
}

resolve_json_file() {
    local input_path="$1"
    if [[ -f "$input_path" ]]; then
        printf '%s\n' "$input_path"
        return 0
    fi
    if [[ -f "$SCRIPT_DIR/$input_path" ]]; then
        printf '%s\n' "$SCRIPT_DIR/$input_path"
        return 0
    fi
    return 1
}

json_status() {
    local path="$1"
    local default_value="${2:-missing}"
    if [[ ! -f "$path" ]]; then
        printf '%s\n' "$default_value"
        return 0
    fi
    python3 - "$path" "$default_value" <<'PY'
import json
import sys

path = sys.argv[1]
default_value = sys.argv[2]

try:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    print(data.get("status", default_value))
except Exception:
    print(default_value)
PY
}

run_step() {
    local title="$1"
    shift

    log_step "$title"

    local rendered_command
    printf -v rendered_command '%q ' "$@"
    if [[ ${#rendered_command} -gt 800 ]]; then
        rendered_command="${rendered_command:0:800} ... [truncated]"
    fi
    log_info "Command: ${rendered_command% }"

    "$@"
    local status=$?
    LAST_EXIT_CODE=$status

    if [[ $status -eq 0 ]]; then
        log_ok "$title succeeded"
    else
        log_warn "$title failed with exit code $status"
        FAILED_STEPS+=("$title (exit=$status)")
    fi

    return 0
}

copy_if_exists() {
    local source_path="$1"
    local target_path="$2"
    local label="$3"

    if [[ -f "$source_path" ]]; then
        run_step "$label" cp "$source_path" "$target_path"
    else
        log_warn "$label skipped: file not found: $source_path"
    fi
}

record_failure() {
    local message="$1"
    log_warn "$message"
    FAILED_STEPS+=("$message")
    LAST_EXIT_CODE=1
}

show_json_if_exists() {
    local title="$1"
    local path="$2"

    if [[ -f "$path" ]]; then
        log_step "$title"
        cat "$path"
    else
        log_warn "$title skipped: file not found: $path"
    fi
}

ensure_runtime_dirs() {
    mkdir -p "$RUN_OUTPUT_DIR" "$LOG_DIR"
}

save_prompt_copy() {
    local file_name="$1"
    local prompt_text="$2"

    ensure_runtime_dirs
    printf '%s\n' "$prompt_text" > "$LOG_DIR/$file_name"
}

ensure_quantize_script_artifact() {
    local quant_script_path="${RUN_OUTPUT_DIR}/quantize.py"
    local legacy_script_path="${RUN_OUTPUT_DIR}/quantize_script.py"

    ensure_runtime_dirs

    if [[ -f "$quant_script_path" ]]; then
        return 0
    fi

    if [[ -f "$legacy_script_path" ]]; then
        run_step "Normalize quantization script artifact" cp "$legacy_script_path" "$quant_script_path"
    fi

    if [[ ! -f "$quant_script_path" ]]; then
        record_failure "Quantization script missing: expected ${quant_script_path}"
    fi
}

normalize_json_gpu_metadata() {
    local target_path="$1"
    local mode="$2"
    local quant_gpus="$3"
    local eval_gpus="$4"

    [[ -f "$target_path" ]] || return 0

    run_step \
        "Normalize $(basename "$target_path") GPU metadata" \
        python3 - "$target_path" "$mode" "$quant_gpus" "$eval_gpus" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
mode = sys.argv[2]
quant_gpus = sys.argv[3]
eval_gpus = sys.argv[4]

with path.open(encoding="utf-8") as handle:
    data = json.load(handle)

if not isinstance(data, dict):
    raise SystemExit(f"{path} must contain a JSON object")

if mode == "quant_summary":
    data["quant_num_gpus"] = quant_gpus
    data["num_gpus"] = quant_gpus
elif mode == "accuracy_auto_quant":
    data["eval_num_gpus"] = eval_gpus
    data["num_gpus"] = eval_gpus
elif mode == "accuracy_auto_eval":
    data["num_gpus"] = eval_gpus
else:
    raise SystemExit(f"unsupported mode: {mode}")

with path.open("w", encoding="utf-8") as handle:
    json.dump(data, handle, indent=2, ensure_ascii=False)
    handle.write("\n")
PY
}

write_quant_prompt() {
    cat <<EOF
You are an expert in LLM quantization using the Intel Auto-Round toolkit.
You MUST follow the skill instructions in: ${QUANT_SKILL_PATH}

Model: ${MODEL_ID}
Quantization: ${SCHEME} / ${METHOD}
Export format: ${EXPORT_FORMAT}
Quantized Model Output directory: ${QUANTIZED_MODEL_DIR}
Runtime artifact directory: ${RUN_OUTPUT_DIR}
Runtime device: ${DEVICE}
Num gpus: ${NUM_GPUS}

Directory responsibilities:
- Write exported model files to: ${QUANTIZED_MODEL_DIR}
- Write runtime artifacts such as quant_summary.json, quantize.py, logs, prompts, copied request/session files, and the venv to: ${RUN_OUTPUT_DIR}

CRITICAL SCRIPT REQUIREMENT:
- Before starting quantization, you MUST first generate the quantization script file:
    ${RUN_OUTPUT_DIR}/quantize.py
- The file name must be exactly: quantize.py
- Run quantization by executing that generated quantize.py script
- Do not use quantize_script.py as the final artifact name

CRITICAL ENVIRONMENT NOTE:
- System Python has torch+cuda pre-installed. When creating venvs, ALWAYS use:
    python3 -m venv --system-site-packages <path>
  This ensures the venv inherits torch+cuda. Do NOT pip install torch inside the venv.
- If /root/.venv exists, reuse /root/.venv before creating a new venv.
- Use uv pip for dependency installation. Prefer:
    uv pip install --python <venv>/bin/python <packages>
- Do NOT reinstall torch or flash_attn if they already import successfully from the reused environment. Only install them when missing or incompatible.
- This workflow is CUDA-focused. For AutoRound device selection:
    - if Num gpus == 1, prefer device="cuda"
    - if Num gpus > 1, prefer device_map="auto"
  Do NOT default to device_map="0" or device_map="0,1,2,3" unless manual mapping is truly required after auto placement fails.

IMPORTANT - After quantization completes (success or failure), you MUST produce:

${QUANT_SUMMARY_JSON} - structured summary:
{
  "model_id": "${MODEL_ID}",
  "scheme": "${SCHEME}",
  "method": "${METHOD}",
  "export_format": "${EXPORT_FORMAT}",
  "device": "${DEVICE}",
  "quant_num_gpus": "${NUM_GPUS}",
  "num_gpus": "${NUM_GPUS}",
  "output_dir": "${RUN_OUTPUT_DIR}",
  "runtime_output_dir": "${RUN_OUTPUT_DIR}",
  "quantized_model_dir": "${QUANTIZED_MODEL_DIR}",
  "status": "success" or "failed",
  "duration_seconds": <float>,
  "original_size_mb": <float or null>,
  "quantized_size_mb": <float or null>,
  "compression_ratio": <float or null>,
  "errors": [<list of error strings>],
  "solutions": [<list of solution strings>],
  "output_files": [<list of file paths in runtime_output_dir>]
}

Write as valid JSON. If quantization fails, still write quant_summary.json with status=failed.
EOF
}

write_eval_prompt() {
    cat <<EOF
You are an expert in evaluating quantized LLM models.
You MUST follow the skill instructions in: ${EVAL_SKILL_PATH}

Quantized model path: ${QUANTIZED_MODEL_DIR}
Runtime artifact directory: ${RUN_OUTPUT_DIR}
Raw lm_eval output directory: ${LM_EVAL_OUTPUT_DIR}
Evaluation tasks: ${EVAL_TASKS}
Batch size: ${EVAL_BATCH_SIZE}
Num gpus: ${EVAL_NUM_GPUS}

The quantized model was produced by auto_quant with scheme=${SCHEME}, export_format=${EXPORT_FORMAT}.
A venv may already exist at ${RUN_OUTPUT_DIR}/venv (created by auto_quant with --system-site-packages).

CRITICAL ENVIRONMENT NOTE:
- System Python has torch+cuda pre-installed. When creating venvs, ALWAYS use:
    python3 -m venv --system-site-packages <path>
  This ensures the venv inherits torch+cuda. Do NOT pip install torch inside the venv.
- If /root/.venv exists, reuse /root/.venv before creating a new venv.
- If a venv already exists at ${RUN_OUTPUT_DIR}/venv, reuse it - just install lm_eval and vllm into it.
- Use uv pip for dependency installation. Prefer:
    uv pip install --python <venv>/bin/python <packages>
- Do NOT reinstall torch or flash_attn if they already import successfully from the reused environment. Only install them when missing or incompatible.
- Write evaluation outputs, logs, prompts, copied request/session files, and other runtime artifacts to: ${RUN_OUTPUT_DIR}
- When invoking lm_eval, you MUST pass:
    --output_path ${LM_EVAL_OUTPUT_DIR}
- Do NOT omit --output_path. Keep the raw lm_eval output files under that exact directory for later upload.

IMPORTANT - After evaluation completes, you MUST produce:

${ACCURACY_JSON} - evaluation results:
{
  "model_id": "${MODEL_ID}",
  "model_path": "${QUANTIZED_MODEL_DIR}",
  "scheme": "${SCHEME}",
  "device": "${DEVICE}:${DEVICE_INDEX}",
  "num_gpus": "${EVAL_NUM_GPUS}",
  "tasks": {
    "<task_name>": {
      "accuracy": <float>,
      "accuracy_stderr": <float or null>
    }
  },
  "status": "success" or "failed",
  "duration_seconds": <float>,
  "eval_framework": "lm_eval+vllm" or "lm_eval+hf" or "manual",
  "errors": [<list of error strings if any>]
}

${LM_EVAL_OUTPUT_DIR}/ - raw lm_eval output directory created by:
    lm_eval ... --output_path ${LM_EVAL_OUTPUT_DIR}

The accuracy values MUST be real numbers from actual evaluation runs.
Write as valid JSON. If evaluation fails, still write accuracy.json with status=failed.
EOF
}

JSON_INPUT=""
SKIP_UPLOAD=false
SKIP_HF=false
SKIP_GITHUB=false
DRY_RUN=false

for arg in "$@"; do
    case "$arg" in
        --skip-upload) SKIP_UPLOAD=true ;;
        --skip-hf) SKIP_HF=true ;;
        --skip-github) SKIP_GITHUB=true ;;
        --dry-run) DRY_RUN=true ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            if [[ -z "$JSON_INPUT" ]]; then
                JSON_INPUT="$arg"
            else
                log_error "Unknown argument: $arg"
                usage
                exit 1
            fi
            ;;
    esac
done

if [[ -z "$JSON_INPUT" ]]; then
    usage
    exit 1
fi

JSON_FILE="$(resolve_json_file "$JSON_INPUT")" || {
    log_error "JSON file not found: $JSON_INPUT"
    exit 1
}
JSON_FILENAME="$(basename "$JSON_FILE")"

CONFIG_FILE="${SCRIPT_DIR}/config.env"
if [[ -f "$CONFIG_FILE" ]]; then
    set -a
    source "$CONFIG_FILE"
    set +a
fi

require_command python3

eval "$(python3 - "$JSON_FILE" <<'PY'
import json
import re
import shlex
import sys

json_file = sys.argv[1]
with open(json_file, encoding="utf-8") as handle:
    data = json.load(handle)

job_type = data.get("job_type", "")
script = data.get("script", "")
model = data.get("model", "")
pipeline = "auto_quant" if script == "auto_quant" or job_type == "quantization & evaluation" else "auto_eval"
quant_scheme_full = data.get("quant_scheme", data.get("compute_dtype", "INT4 (W4A16)"))
match = re.search(r"W\d+A\d+", str(quant_scheme_full))
scheme = match.group(0) if match else "W4A16"
model_slug = model.replace("/", "_").replace(" ", "_")
quant_gpu = str(data.get("quant_gpu_nums", data.get("gpu_nums", 1)))
eval_gpu = str(data.get("eval_gpu_nums", data.get("gpu_nums", quant_gpu)))
model_path_override = model if pipeline == "auto_eval" else ""

exports = {
    "JOB_TYPE": job_type,
    "PIPELINE": pipeline,
    "MODEL_ID": model,
    "MODEL_SLUG": model_slug,
    "REVISION": data.get("revision", "main"),
    "SCHEME": scheme,
    "QUANT_SCHEME_FULL": quant_scheme_full,
    "NUM_GPUS": quant_gpu,
    "EVAL_NUM_GPUS": eval_gpu,
    "QUANT_PRECISION": data.get("quant_precision", data.get("precision", "4bit")),
    "QUANT_WEIGHT_DTYPE": data.get("quant_weight_dtype", data.get("weight_dtype", "int4")),
    "MODEL_PATH_OVERRIDE": model_path_override,
}

for key, value in exports.items():
    print(f"export {key}={shlex.quote(str(value))}")
PY
)"

OUTPUT_DIR="${OUTPUT_DIR:-/root/.openclaw/workspace/quantized}"
RUNTIME_OUTPUT_BASE_DIR="${RUNTIME_OUTPUT_BASE_DIR:-/root/.openclaw/workspace/quantized/runs}"
METHOD="${METHOD:-RTN}"
EXPORT_FORMAT="${EXPORT_FORMAT:-auto_round}"
DEVICE="${DEVICE:-cuda}"
DEVICE_INDEX="${DEVICE_INDEX:-0}"
TIMEOUT="${TIMEOUT:-36000}"
EVAL_TASKS="${EVAL_TASKS:-piqa,mmlu,hellaswag,gsm8k}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
OPENCLAW_WORKSPACE_DIR="${OPENCLAW_WORKSPACE_DIR:-}"
OPENCLAW_SESSIONS_DIR="${OPENCLAW_SESSIONS_DIR:-/root/.openclaw/agents/main/sessions}"

MODEL_OUTPUT_DIR="${OUTPUT_DIR}/${MODEL_SLUG}-${SCHEME}"
RUN_OUTPUT_DIR="${RUNTIME_OUTPUT_BASE_DIR}/${MODEL_SLUG}-${SCHEME}"
QUANTIZED_MODEL_DIR="$MODEL_OUTPUT_DIR"
if [[ "$PIPELINE" == "auto_eval" && -n "${MODEL_PATH_OVERRIDE:-}" ]]; then
    QUANTIZED_MODEL_DIR="$MODEL_PATH_OVERRIDE"
fi

if [[ -z "$OPENCLAW_WORKSPACE_DIR" ]]; then
    OPENCLAW_WORKSPACE_DIR="$(
        resolve_first_existing_dir \
            "/root/leaderboard_Agent/auto_quant/openclaw_home/workspace" \
            "/root/leaderboard_Agent/tasks/lb_eval/openclaw_config/workspace" \
            "/root/.openclaw/workspace"
    )" || {
        log_error "Could not resolve OPENCLAW_WORKSPACE_DIR; set it in config.env"
        exit 1
    }
fi

LOG_DIR="${RUN_OUTPUT_DIR}/logs"
ensure_runtime_dirs
LOG_FILE="${LOG_DIR}/auto.log"
: > "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

FAILED_STEPS=()
LAST_EXIT_CODE=0

QUANT_SKILL_PATH=""
if [[ "$PIPELINE" == "auto_quant" ]]; then
    QUANT_SKILL_PATH="$(resolve_skill_path "auto_quant")" || {
        log_error "Quant skill file not found"
        exit 1
    }
fi
EVAL_SKILL_NAME="auto_eval"
if [[ "$PIPELINE" == "auto_eval" ]]; then
    EVAL_SKILL_NAME="auto_eval_vllm"
fi
EVAL_SKILL_PATH="$(resolve_skill_path "$EVAL_SKILL_NAME")" || {
    log_error "Eval skill file not found for skill: $EVAL_SKILL_NAME"
    exit 1
}
QUANT_SESSION="autoeval_quant_$$"
EVAL_SESSION="autoeval_eval_$$"
QUANT_SUMMARY_JSON="${RUN_OUTPUT_DIR}/quant_summary.json"
ACCURACY_JSON="${RUN_OUTPUT_DIR}/accuracy.json"
LM_EVAL_OUTPUT_DIR="${RUN_OUTPUT_DIR}/lm_eval_results"
REQUEST_JSON="${RUN_OUTPUT_DIR}/request.json"
QUANT_SESSION_SRC="${OPENCLAW_SESSIONS_DIR}/${QUANT_SESSION}.jsonl"
EVAL_SESSION_SRC="${OPENCLAW_SESSIONS_DIR}/${EVAL_SESSION}.jsonl"
QUANT_SESSION_DST="${RUN_OUTPUT_DIR}/session_quant_$$.jsonl"
EVAL_SESSION_DST="${RUN_OUTPUT_DIR}/session_eval_$$.jsonl"
FORMATTER="${SCRIPT_DIR}/format_sessions.py"
AZURE_UPLOADER="${SCRIPT_DIR}/upload_results_azure.py"

log_step "Resolved configuration"
echo "JSON file           : $JSON_FILENAME"
echo "Job type            : $JOB_TYPE"
echo "Pipeline            : $PIPELINE"
echo "Model               : $MODEL_ID"
echo "Revision            : $REVISION"
echo "Scheme              : $SCHEME ($QUANT_SCHEME_FULL)"
echo "Quant GPUs          : $NUM_GPUS"
echo "Eval GPUs           : $EVAL_NUM_GPUS"
echo "OpenClaw workspace  : $OPENCLAW_WORKSPACE_DIR"
echo "OpenClaw sessions   : $OPENCLAW_SESSIONS_DIR"
echo "Eval skill          : $EVAL_SKILL_NAME"
echo "Quant skill path    : ${QUANT_SKILL_PATH:-'(not used)'}"
echo "Eval skill path     : $EVAL_SKILL_PATH"
echo "Model output dir    : $MODEL_OUTPUT_DIR"
echo "Runtime output dir  : $RUN_OUTPUT_DIR"
echo "Quantized model dir : $QUANTIZED_MODEL_DIR"
echo "Log file            : $LOG_FILE"
echo "Skip upload(all)    : $SKIP_UPLOAD"
echo "Skip HF upload      : $SKIP_HF"
echo "Skip GitHub upload  : $SKIP_GITHUB"

if [[ "$PIPELINE" == "auto_quant" && ! -f "$QUANT_SKILL_PATH" ]]; then
    log_error "Quant skill file not found: $QUANT_SKILL_PATH"
    exit 1
fi
if [[ ! -f "$EVAL_SKILL_PATH" ]]; then
    log_error "Eval skill file not found: $EVAL_SKILL_PATH"
    exit 1
fi

ensure_runtime_dirs
run_step "Copy request JSON" cp "$JSON_FILE" "$REQUEST_JSON"

if [[ "$DRY_RUN" == "true" ]]; then
    log_ok "Dry run complete"
    exit 0
fi

require_command openclaw

QUANT_STATUS="$(json_status "$QUANT_SUMMARY_JSON")"
if [[ "$PIPELINE" == "auto_quant" ]]; then
    if [[ "$QUANT_STATUS" != "success" ]]; then
        QUANT_PROMPT="$(write_quant_prompt)"
        save_prompt_copy "quant_prompt.txt" "$QUANT_PROMPT"
        run_step \
            "Run auto_quant" \
            env \
                http_proxy="${HTTP_PROXY:-}" \
                https_proxy="${HTTPS_PROXY:-}" \
                HTTP_PROXY="${HTTP_PROXY:-}" \
                HTTPS_PROXY="${HTTPS_PROXY:-}" \
                openclaw agent --local \
                    --session-id "$QUANT_SESSION" \
                    --message "$QUANT_PROMPT" \
                    --timeout "$TIMEOUT"
        QUANT_STATUS="$(json_status "$QUANT_SUMMARY_JSON")"
    else
        log_ok "Quantization already succeeded, skipping auto_quant"
    fi
else
    QUANT_STATUS="success"
fi

if [[ "$PIPELINE" == "auto_quant" ]]; then
    ensure_runtime_dirs
    copy_if_exists "$QUANT_SESSION_SRC" "$QUANT_SESSION_DST" "Copy quant session log"
    ensure_quantize_script_artifact
fi

EVAL_STATUS="$(json_status "$ACCURACY_JSON")"
if [[ "$EVAL_STATUS" != "success" ]]; then
    if [[ "$QUANT_STATUS" == "success" ]]; then
        EVAL_PROMPT="$(write_eval_prompt)"
        save_prompt_copy "eval_prompt.txt" "$EVAL_PROMPT"
        run_step \
            "Run ${EVAL_SKILL_NAME}" \
            env \
                http_proxy="${HTTP_PROXY:-}" \
                https_proxy="${HTTPS_PROXY:-}" \
                HTTP_PROXY="${HTTP_PROXY:-}" \
                HTTPS_PROXY="${HTTPS_PROXY:-}" \
                openclaw agent --local \
                    --session-id "$EVAL_SESSION" \
                    --message "$EVAL_PROMPT" \
                    --timeout "$TIMEOUT"
        EVAL_STATUS="$(json_status "$ACCURACY_JSON")"
    else
        log_warn "Skipping auto_eval because quantization status is $QUANT_STATUS"
    fi
else
    log_ok "Evaluation already succeeded, skipping auto_eval"
fi

ensure_runtime_dirs
copy_if_exists "$EVAL_SESSION_SRC" "$EVAL_SESSION_DST" "Copy eval session log"

if [[ "$PIPELINE" == "auto_quant" ]]; then
    normalize_json_gpu_metadata "$QUANT_SUMMARY_JSON" "quant_summary" "$NUM_GPUS" "$EVAL_NUM_GPUS"
    normalize_json_gpu_metadata "$ACCURACY_JSON" "accuracy_auto_quant" "$NUM_GPUS" "$EVAL_NUM_GPUS"
else
    normalize_json_gpu_metadata "$ACCURACY_JSON" "accuracy_auto_eval" "$NUM_GPUS" "$EVAL_NUM_GPUS"
fi

SESSION_INPUTS=()
if [[ -f "$QUANT_SESSION_DST" ]]; then
    SESSION_INPUTS+=("$QUANT_SESSION_DST")
fi
if [[ -f "$EVAL_SESSION_DST" ]]; then
    SESSION_INPUTS+=("$EVAL_SESSION_DST")
fi
if [[ ${#SESSION_INPUTS[@]} -gt 0 ]]; then
    run_step "Format session logs" python3 "$FORMATTER" "${SESSION_INPUTS[@]}"
else
    log_warn "Format session logs skipped: no session JSONL files were copied"
fi

show_json_if_exists "Quant summary" "$QUANT_SUMMARY_JSON"
show_json_if_exists "Accuracy summary" "$ACCURACY_JSON"

if [[ "$PIPELINE" == "auto_quant" && "$SKIP_UPLOAD" != "true" && "$SKIP_HF" != "true" ]]; then
    if [[ "$QUANT_STATUS" == "success" ]]; then
        MODEL_SHORT="${MODEL_ID#*/}"
        HF_REPO_NAME="${MODEL_SHORT}-autoround-${SCHEME}"
        run_step \
            "Upload quantized model to HuggingFace" \
            python3 "$SCRIPT_DIR/upload_model_hf.py" \
                "$MODEL_OUTPUT_DIR" \
                "$HF_REPO_NAME" \
                --summary-json "$QUANT_SUMMARY_JSON"
    else
        log_warn "Skipping HuggingFace upload because quantization status is $QUANT_STATUS"
    fi
fi

if [[ "$SKIP_UPLOAD" != "true" && "$SKIP_GITHUB" != "true" ]]; then
    run_step \
        "Upload result artifacts to Azure Blob Storage" \
        python3 "$AZURE_UPLOADER" \
            "$RUN_OUTPUT_DIR" \
            "$MODEL_ID" \
            --pipeline "$PIPELINE" \
            --scheme "$SCHEME" \
            --quant-num-gpus "$NUM_GPUS" \
            --eval-num-gpus "$EVAL_NUM_GPUS" \
            --model-output-dir "$QUANTIZED_MODEL_DIR" \
            --request-filename "$JSON_FILENAME"
fi

log_step "Final summary"
echo "Quant status : $QUANT_STATUS"
echo "Eval status  : $EVAL_STATUS"
echo "Model dir    : $QUANTIZED_MODEL_DIR"
echo "Runtime dir  : $RUN_OUTPUT_DIR"
echo "Log file     : $LOG_FILE"

if [[ ${#FAILED_STEPS[@]} -gt 0 ]]; then
    echo "Step failures:"
    printf '  - %s\n' "${FAILED_STEPS[@]}"
fi

OVERALL_EXIT=0
if [[ "$PIPELINE" == "auto_quant" && "$QUANT_STATUS" != "success" ]]; then
    OVERALL_EXIT=1
fi
if [[ "$EVAL_STATUS" != "success" ]]; then
    OVERALL_EXIT=1
fi
if [[ ${#FAILED_STEPS[@]} -gt 0 ]]; then
    OVERALL_EXIT=1
fi

if [[ $OVERALL_EXIT -eq 0 ]]; then
    log_ok "Pipeline finished successfully"
else
    log_warn "Pipeline finished with failures"
fi

exit "$OVERALL_EXIT"
