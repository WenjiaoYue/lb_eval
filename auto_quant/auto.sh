#!/usr/bin/env bash
# auto_v3.sh — Phases-based quantization pipeline (v3)
#
# Architecture:
#   Phase 1: setup_env.sh     (deterministic environment install)
#   Phase 2: quantize.py      (deterministic quantization with recipes)
#   Phase 3: evaluate.sh      (deterministic evaluation, hf/vllm backend)
#   Phase 4: upload           (reuse existing upload_model_hf.py + upload_results_github.py)
#
#   On failure: agent_fix_loop attempts repair via OpenClaw agent
#
# Usage:
#   bash auto_v3.sh <task_json_file> [options]
#
# Options:
#   --skip-upload      Skip all uploads
#   --skip-agent       Skip agent fix loop (fail immediately on error)
#   --dry-run          Print resolved configuration and exit
#   -h, --help         Show this help

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASES_DIR="${SCRIPT_DIR}/phases"

# ═══ Global log capture ═══
# Capture entire pipeline stdout+stderr to auto.log for full traceability
_AUTO_LOG="${SCRIPT_DIR}/output/.auto_v3_$$.log"
mkdir -p "$(dirname "${_AUTO_LOG}")"
exec > >(tee -a "${_AUTO_LOG}") 2>&1

# ═══ Colors ═══
if [[ -t 1 ]]; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
    CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; CYAN=''; BOLD=''; NC=''
fi

log_info()  { echo -e "${CYAN}[auto_v3]${NC} $*"; }
log_ok()    { echo -e "${GREEN}[auto_v3]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[auto_v3]${NC} $*"; }
log_error() { echo -e "${RED}[auto_v3]${NC} $*"; }
log_step()  { echo -e "\n${BOLD}${CYAN}═══════ $* ═══════${NC}\n"; }

# ═══ Load config ═══
if [[ -f "${SCRIPT_DIR}/config.env" ]]; then
    source "${SCRIPT_DIR}/config.env"
fi

# ═══ Source agent fix loop library ═══
source "${PHASES_DIR}/agent_fix_loop.sh"

# ═══ Parse arguments ═══
TASK_JSON=""
SKIP_UPLOAD=false
SKIP_AGENT=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-upload)  SKIP_UPLOAD=true; shift ;;
        --skip-agent)   SKIP_AGENT=true; shift ;;
        --dry-run)      DRY_RUN=true; shift ;;
        -h|--help)
            echo "Usage: bash auto_v3.sh <task_json_file> [--skip-upload] [--skip-agent] [--dry-run]"
            exit 0 ;;
        *)
            if [[ -z "$TASK_JSON" ]]; then
                TASK_JSON="$1"
            fi
            shift ;;
    esac
done

if [[ -z "$TASK_JSON" ]]; then
    log_error "No task JSON file specified"
    echo "Usage: bash auto_v3.sh <task_json_file>"
    exit 1
fi

# Resolve JSON path
if [[ ! -f "$TASK_JSON" ]] && [[ -f "${SCRIPT_DIR}/${TASK_JSON}" ]]; then
    TASK_JSON="${SCRIPT_DIR}/${TASK_JSON}"
fi
if [[ ! -f "$TASK_JSON" ]]; then
    log_error "Task JSON not found: $TASK_JSON"
    exit 1
fi

# ═══ Parse task JSON ═══
eval "$(python3 - "$TASK_JSON" <<'PYEOF'
import json
import sys

with open(sys.argv[1]) as f:
    task = json.load(f)

# Extract fields with defaults
model = task.get("model", "")
scheme = task.get("scheme", task.get("quant_scheme", task.get("quant_type", "W4A16")))
method = task.get("method", "RTN")
export_format = task.get("export_format", "auto_round")
auto_round_ref = task.get("auto_round_ref", "latest")
transformers_ref = task.get("transformers_ref", "auto")
request_filename = task.get("request_filename", "")
# Explicit GPU card pinning (AWS B200 / local-agent path). Comma-separated
# physical card indices, e.g. "0" or "0,1,3". Empty when not pinned.
cuda_visible_devices = str(task.get("cuda_visible_devices", "") or "").strip()
# Optional advanced quant controls (whitelisted submissions only).
ignore_layers = str(task.get("ignore_layers", "") or "").strip()
layer_config = str(task.get("layer_config", "") or "").strip()
# Hardware / agent type — used to place the HF cache on the B200 node's mounted
# /azure disk instead of the container layer. B200 submissions set agent_type="local"
# and carry an "AWS B200" gpu/hardware label.
agent_type = str(task.get("agent_type", "") or "").strip()
_gpu_type = " ".join(str(task.get(k, "") or "") for k in
                     ("hardware", "gpu_type", "quant_gpu_type", "eval_gpu_type"))
is_b200 = ("b200" in _gpu_type.lower()) or (agent_type.lower() == "local")
# If request_filename not in JSON, derive from the JSON filename itself
if not request_filename:
    import os
    request_filename = os.path.basename(sys.argv[1])

# Normalize scheme from various request formats
scheme_map = {
    "INT4 (W4A16)": "W4A16",
    "INT8 (W8A16)": "W8A16",
    "INT4 (W4A8)": "W4A8",
    "int4": "W4A16",
    "int8": "W8A16",
    "nvfp4": "NVFP4",
    "mxfp4": "MXFP4",
}
scheme = scheme_map.get(scheme, scheme)

# Normalize method from iters
iters = task.get("iters", None)
if iters is not None:
    method = "RTN" if int(iters) == 0 else "TUNING"

print(f'MODEL_ID="{model}"')
print(f'SCHEME="{scheme}"')
print(f'METHOD="{method}"')
print(f'EXPORT_FORMAT="{export_format}"')
print(f'AUTO_ROUND_REF="{auto_round_ref}"')
print(f'TRANSFORMERS_REF="{transformers_ref}"')
print(f'REQUEST_FILENAME="{request_filename}"')
print(f'REQ_CUDA_VISIBLE_DEVICES="{cuda_visible_devices}"')
print(f'IS_B200="{"true" if is_b200 else "false"}"')
# Use shlex.quote for free-form advanced values so the shell `eval` is injection-safe.
import shlex
print(f'REQ_IGNORE_LAYERS={shlex.quote(ignore_layers)}')
print(f'REQ_LAYER_CONFIG={shlex.quote(layer_config)}')
PYEOF
)"

# ═══ Derive variables ═══
case "${EXPORT_FORMAT}" in
    auto_round)      EVAL_BACKEND="hf" ;;
    llm_compressor)  EVAL_BACKEND="vllm" ;;
    *)               EVAL_BACKEND="hf" ;;
esac

case "${METHOD}" in
    RTN)        ITERS=0;   METHOD_SUFFIX="RTN";      MODEL_FREE=false ;;
    TUNING)     ITERS=200; METHOD_SUFFIX="Tuning";   MODEL_FREE=false ;;
    MODEL_FREE) ITERS=0;   METHOD_SUFFIX="ModelFree"; MODEL_FREE=true ;;
    *)          ITERS=0;   METHOD_SUFFIX="${METHOD}"; MODEL_FREE=false ;;
esac

# Use config.env defaults where task JSON didn't override
DEVICE="${DEVICE:-cuda}"
DEVICE_INDEX="${DEVICE_INDEX:-0}"
EVAL_TASKS="${EVAL_TASKS:-piqa,mmlu,hellaswag}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
NUM_GPUS="${NUM_GPUS:-1}"

# Advanced quant controls (empty unless a whitelisted submission set them).
IGNORE_LAYERS="${REQ_IGNORE_LAYERS:-}"
LAYER_CONFIG="${REQ_LAYER_CONFIG:-}"

# ═══ Explicit GPU card pinning (AWS B200 / local-agent path) ═══
# When the request.json specifies cuda_visible_devices (e.g. "0,1"), pin the run
# to exactly those physical cards for BOTH quantize and evaluate. We export
# CUDA_VISIBLE_DEVICES so torch/vLLM only see those cards (re-indexed to 0..N-1),
# make the card count authoritative for NUM_GPUS, and reset DEVICE_INDEX to 0
# (the first *visible* card after masking).
if [[ -n "${REQ_CUDA_VISIBLE_DEVICES:-}" ]]; then
    # Validate: comma-separated digits only (defensive; UI already validates).
    if [[ "${REQ_CUDA_VISIBLE_DEVICES}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        export CUDA_VISIBLE_DEVICES="${REQ_CUDA_VISIBLE_DEVICES}"
        NUM_GPUS=$(awk -F',' '{print NF}' <<< "${REQ_CUDA_VISIBLE_DEVICES}")
        DEVICE_INDEX=0
        log_info "GPU pinning: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (NUM_GPUS=${NUM_GPUS}, DEVICE_INDEX=0)"
    else
        log_warn "Ignoring malformed cuda_visible_devices='${REQ_CUDA_VISIBLE_DEVICES}' (expected e.g. '0' or '0,1')"
    fi
fi

# Output directories
MODEL_SHORT="${MODEL_ID#*/}"
HF_REPO_NAME="${MODEL_SHORT}-AutoRound-${SCHEME}-${METHOD_SUFFIX}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output}"
RUNTIME_OUTPUT_BASE_DIR="${RUNTIME_OUTPUT_BASE_DIR:-${OUTPUT_DIR}/runs}"
RUN_OUTPUT_DIR="${RUNTIME_OUTPUT_BASE_DIR}/${HF_REPO_NAME}"
QUANTIZED_MODEL_DIR="${RUN_OUTPUT_DIR}/quantized_model"
EVAL_OUTPUT_DIR="${RUN_OUTPUT_DIR}/lm_eval_results"
LOG_DIR="${RUN_OUTPUT_DIR}/logs"

# ═══ HuggingFace cache placement ═══
# On AWS B200 (local-agent) nodes, /azure is a large mounted disk — put the HF
# cache there so big model/dataset downloads don't fill the container's writable
# layer. For every other hardware, keep HuggingFace's default (~/.cache/huggingface).
#   - config.env HF_HOME set (non-empty) → always honor it verbatim (explicit override)
#   - else B200 submission                → HF_HOME=/azure/hf_cache
#   - else                                → default (~/.cache/huggingface)
if [[ -n "${HF_HOME:-}" ]]; then
    export HF_HOME
    mkdir -p "${HF_HOME}"
    export HF_HUB_CACHE="${HF_HOME}/hub" HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
    export TRANSFORMERS_CACHE="${HF_HOME}/hub" HF_DATASETS_CACHE="${HF_HOME}/datasets"
    log_info "HF cache: ${HF_HOME} (explicit HF_HOME override)"
elif [[ "${IS_B200:-false}" == "true" ]]; then
    export HF_HOME="/azure/hf_cache"
    mkdir -p "${HF_HOME}"
    export HF_HUB_CACHE="${HF_HOME}/hub" HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
    export TRANSFORMERS_CACHE="${HF_HOME}/hub" HF_DATASETS_CACHE="${HF_HOME}/datasets"
    log_info "HF cache: ${HF_HOME} (AWS B200 mounted disk)"
else
    log_info "HF cache: HuggingFace default (~/.cache/huggingface)"
fi

# lb_eval repo (for upload_results_github.py clone target)
LB_EVAL_REPO_DIR="${GIT_RESULTS_REPO_DIR:-${SCRIPT_DIR}/lb_eval}"
# Lessons are stored alongside phases in the checkout, not inside the clone target
LESSONS_DIR="${SCRIPT_DIR}/lessons"
GIT_BRANCH="${GIT_BRANCH:-main}"

# Export for child scripts
export MODEL_ID SCHEME METHOD ITERS EXPORT_FORMAT EVAL_BACKEND MODEL_FREE
export IGNORE_LAYERS LAYER_CONFIG
export AUTO_ROUND_REF TRANSFORMERS_REF
export DEVICE DEVICE_INDEX EVAL_TASKS EVAL_BATCH_SIZE NUM_GPUS
export RUN_OUTPUT_DIR QUANTIZED_MODEL_DIR EVAL_OUTPUT_DIR
export DEVICE_MAP="${DEVICE_MAP:-auto}"
export LB_EVAL_REPO_DIR LESSONS_DIR GIT_BRANCH
export REQUEST_FILENAME
# Tokens — needed by upload scripts and error_analysis (Python subprocesses)
export GIT_TOKEN="${GIT_TOKEN:-}"
export HF_TOKEN="${HF_TOKEN:-${HF_TOKENS%%,*}}"
export HF_TOKENS="${HF_TOKENS:-}"

mkdir -p "${RUN_OUTPUT_DIR}" "${LOG_DIR}" "${LESSONS_DIR}"

# Relocate global auto.log into the proper log directory
if [[ -f "${_AUTO_LOG}" ]]; then
    mv "${_AUTO_LOG}" "${LOG_DIR}/auto.log" 2>/dev/null || true
    _AUTO_LOG="${LOG_DIR}/auto.log"
    exec > >(tee -a "${_AUTO_LOG}") 2>&1
fi

# ═══ Dry run ═══
if [[ "$DRY_RUN" == "true" ]]; then
    log_step "DRY RUN — Resolved Configuration"
    echo "  MODEL_ID:         ${MODEL_ID}"
    echo "  SCHEME:           ${SCHEME}"
    echo "  METHOD:           ${METHOD} (iters=${ITERS})"
    echo "  EXPORT_FORMAT:    ${EXPORT_FORMAT}"
    echo "  EVAL_BACKEND:     ${EVAL_BACKEND}"
    echo "  AUTO_ROUND_REF:   ${AUTO_ROUND_REF}"
    echo "  TRANSFORMERS_REF: ${TRANSFORMERS_REF}"
    echo "  RUN_OUTPUT_DIR:   ${RUN_OUTPUT_DIR}"
    echo "  QUANTIZED_MODEL:  ${QUANTIZED_MODEL_DIR}"
    echo "  EVAL_OUTPUT:      ${EVAL_OUTPUT_DIR}"
    echo "  LESSONS_DIR:      ${LESSONS_DIR}"
    echo "  SKIP_UPLOAD:      ${SKIP_UPLOAD}"
    echo "  SKIP_AGENT:       ${SKIP_AGENT}"
    exit 0
fi

# ═══ Pull latest lessons ═══
if [[ -d "${LB_EVAL_REPO_DIR}/.git" ]]; then
    cd "${LB_EVAL_REPO_DIR}"
    git pull --rebase 2>/dev/null || log_warn "git pull failed (non-fatal)"
    cd - > /dev/null
fi

# ═══ Copy task JSON for reference ═══
cp "${TASK_JSON}" "${RUN_OUTPUT_DIR}/request.json" 2>/dev/null || true

# ═══════════════════════════════════════════════════════════════════
# PIPELINE EXECUTION
# ═══════════════════════════════════════════════════════════════════
log_step "Pipeline: ${MODEL_ID} | ${SCHEME}/${METHOD}/${EXPORT_FORMAT}"
PIPELINE_START=$(date +%s)
FAILED_STEPS=()

# --- Phase 1: Environment Setup ---
if [[ "$SKIP_AGENT" == "true" ]]; then
    bash "${PHASES_DIR}/setup_env.sh" 2>&1 | tee "${LOG_DIR}/setup_env.log"
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        log_error "setup_env failed (no agent retry)"
        FAILED_STEPS+=("setup_env")
    fi
else
    agent_fix_loop "setup_env" "${PHASES_DIR}/setup_env.sh" || {
        FAILED_STEPS+=("setup_env")
        log_error "setup_env failed after all fix attempts"
    }
fi

# --- Phase 2: Quantization ---
if [[ ${#FAILED_STEPS[@]} -eq 0 ]]; then
    if [[ "$SKIP_AGENT" == "true" ]]; then
        bash "${PHASES_DIR}/quantize_wrapper.sh" 2>&1 | tee "${LOG_DIR}/quantize.log"
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            FAILED_STEPS+=("quantize")
        fi
    else
        agent_fix_loop "quantize" "${PHASES_DIR}/quantize_wrapper.sh" || {
            FAILED_STEPS+=("quantize")
        }
    fi
fi

# --- Phase 3: Evaluation ---
if [[ ${#FAILED_STEPS[@]} -eq 0 ]]; then
    if [[ "$SKIP_AGENT" == "true" ]]; then
        bash "${PHASES_DIR}/evaluate.sh" "${QUANTIZED_MODEL_DIR}" 2>&1 | tee "${LOG_DIR}/evaluate.log"
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            FAILED_STEPS+=("evaluate")
        fi
    else
        agent_fix_loop "evaluate" "${PHASES_DIR}/evaluate.sh" "${QUANTIZED_MODEL_DIR}" || {
            FAILED_STEPS+=("evaluate")
        }
    fi
fi

# ═══ Determine pipeline status ═══
PIPELINE_END=$(date +%s)
PIPELINE_DURATION=$((PIPELINE_END - PIPELINE_START))

if [[ ${#FAILED_STEPS[@]} -eq 0 ]]; then
    PIPELINE_STATUS="Finished"
    log_ok "Pipeline completed successfully in ${PIPELINE_DURATION}s"
else
    PIPELINE_STATUS="Failed"
    log_error "Pipeline failed at: ${FAILED_STEPS[*]} (${PIPELINE_DURATION}s)"
fi

# ═══ Record working recipe (positive KB) on success ═══
# Captures which scheme/method/params produced a working quantized model + its
# accuracy, so future runs can look up a known-good recipe for this model.
if [[ "$PIPELINE_STATUS" == "Finished" ]]; then
    RECIPE_SCRIPT="${SCRIPT_DIR}/recipes/record_recipe.py"
    if [[ -f "${RECIPE_SCRIPT}" ]]; then
        python3 "${RECIPE_SCRIPT}" record \
            --model "${MODEL_ID}" \
            --scheme "${SCHEME}" \
            --method "${METHOD}" \
            --iters "${ITERS:-0}" \
            --export-format "${EXPORT_FORMAT}" \
            --device "${DEVICE}" \
            --num-gpus "${NUM_GPUS:-1}" \
            --layer-config "${LAYER_CONFIG:-}" \
            --ignore-layers "${IGNORE_LAYERS:-}" \
            --auto-round-ref "${AUTO_ROUND_REF:-}" \
            --transformers-ref "${TRANSFORMERS_REF:-}" \
            --accuracy-json "${RUN_OUTPUT_DIR}/accuracy.json" \
            --status "finished" \
            --source-task "${MODEL_ID}_${SCHEME}_${METHOD}" \
            2>&1 | tee "${LOG_DIR}/recipe.log" || log_warn "Recipe recording failed (non-fatal)"
    fi
fi

# ═══ Collect OpenClaw session logs ═══
# Copy .jsonl session files from the openclaw sessions directory into RUN_OUTPUT_DIR,
# then format them to human-readable .md (matching old pipeline behavior)
OPENCLAW_SESSIONS_DIR="${OPENCLAW_SESSIONS_DIR:-/root/.openclaw/agents/main/sessions}"
if [[ -d "${OPENCLAW_SESSIONS_DIR}" ]]; then
    _session_count=0
    for _jsonl in "${OPENCLAW_SESSIONS_DIR}"/*.jsonl; do
        [[ -f "$_jsonl" ]] || continue
        # Only copy sessions created during this pipeline run (mtime > PIPELINE_START)
        if [[ $(stat -c %Y "$_jsonl" 2>/dev/null || echo 0) -ge ${PIPELINE_START} ]]; then
            # Rename to session_* prefix so upload script can find them
            _basename="$(basename "$_jsonl")"
            if [[ "$_basename" != session_* ]]; then
                _basename="session_${_basename}"
            fi
            cp "$_jsonl" "${RUN_OUTPUT_DIR}/${_basename}" 2>/dev/null && ((_session_count++)) || true
        fi
    done
    if [[ $_session_count -gt 0 ]]; then
        log_info "Collected ${_session_count} openclaw session(s)"
        # Format sessions to Markdown for human readability
        FORMATTER="${SCRIPT_DIR}/format_sessions.py"
        if [[ -f "${FORMATTER}" ]]; then
            python3 "${FORMATTER}" "${RUN_OUTPUT_DIR}"/session_*.jsonl 2>/dev/null || true
        fi
    fi
fi

# ═══ Generate Report (before upload so it gets included) ═══
log_info "Generating run report..."
python3 "${PHASES_DIR}/generate_report.py" "${RUN_OUTPUT_DIR}" || log_warn "Report generation failed (non-fatal)"

# ═══ Phase 4: Upload ═══
if [[ "$SKIP_UPLOAD" != "true" ]]; then
    log_step "Upload Results"

    # 4a: Upload quantized model to HF Hub
    if [[ -d "${QUANTIZED_MODEL_DIR}" ]] && [[ "$PIPELINE_STATUS" == "Finished" ]]; then
        log_info "Uploading quantized model to HuggingFace Hub..."
        python3 "${SCRIPT_DIR}/upload_model_hf.py" \
            "${QUANTIZED_MODEL_DIR}" \
            "${HF_REPO_NAME}" \
            --tokens "${HF_TOKENS:-}" \
            --orgs "${HF_UPLOAD_ORGS:-}" \
            --account-ids "${HF_ACCOUNT_IDS:-}" \
            --summary-json "${RUN_OUTPUT_DIR}/quant_summary.json" \
            --accuracy-json "${RUN_OUTPUT_DIR}/accuracy.json" \
            --usage-file "${HF_USAGE_FILE:-}" \
            --capacity-gb "${HF_ACCOUNT_CAPACITY_GB:-1000}" \
            --shared-ledger-enabled "${HF_SHARED_LEDGER_ENABLED:-false}" \
            --shared-ledger-repo "${HF_SHARED_LEDGER_REPO:-}" \
            --shared-ledger-token "${HF_SHARED_LEDGER_TOKEN:-}" \
            --shared-ledger-branch "${HF_SHARED_LEDGER_BRANCH:-main}" \
            2>&1 | tee "${LOG_DIR}/upload_hf.log" || log_warn "HF upload failed"
    fi

    # 4b: Upload results to lb_eval GitHub
    log_info "Uploading results to lb_eval GitHub..."
    python3 "${SCRIPT_DIR}/upload_results_github.py" \
        "${RUN_OUTPUT_DIR}" \
        "${MODEL_ID}" \
        --scheme "${SCHEME}" \
        --method "${METHOD}" \
        --model-output-dir "${QUANTIZED_MODEL_DIR}" \
        --repo-dir "${LB_EVAL_REPO_DIR}" \
        --git-repo "${GIT_REPO:-}" \
        --git-token "${GIT_TOKEN:-}" \
        --request-filename "${REQUEST_FILENAME:-}" \
        --git-user-name "${GIT_USER_NAME:-auto-pipeline}" \
        --git-user-email "${GIT_USER_EMAIL:-auto@pipeline.local}" \
        2>&1 | tee "${LOG_DIR}/upload_github.log" || log_warn "GitHub upload failed"
fi

# ═══ Error Analysis & Community Reporting (on failure) ═══
if [[ "$PIPELINE_STATUS" == "Failed" ]]; then
    log_step "Error Analysis"
    ERROR_ANALYSIS_SCRIPT="${SCRIPT_DIR}/error_analysis/analyze_failures.py"
    if [[ -f "${ERROR_ANALYSIS_SCRIPT}" ]]; then
        # Determine which phase log to analyze
        _FAILED_PHASE="${FAILED_STEPS[0]}"
        _FAILED_LOG="${LOG_DIR}/${_FAILED_PHASE}.log"

        if [[ -f "${_FAILED_LOG}" ]]; then
            log_info "Analyzing failure: ${_FAILED_PHASE} phase..."

            # Run analysis with agent (unless --skip-agent), push to github, submit community
            _ANALYSIS_ARGS=(
                --run-dir "${RUN_OUTPUT_DIR}"
                --limit 1
                --repo-dir "${LB_EVAL_REPO_DIR}"
                --org "${MODEL_ID%%/*}"
                --artifact-name "${HF_REPO_NAME}"
            )
            if [[ "$SKIP_AGENT" == "true" ]]; then
                _ANALYSIS_ARGS+=(--no-agent)
            fi
            # Always push diagnosis to GitHub (results already uploaded)
            _ANALYSIS_ARGS+=(--push-github)
            # Submit to community discussion for visibility
            _ANALYSIS_ARGS+=(--submit-community)

            python3 "${ERROR_ANALYSIS_SCRIPT}" "${_ANALYSIS_ARGS[@]}" \
                2>&1 | tee "${LOG_DIR}/error_analysis.log" || log_warn "Error analysis failed (non-fatal)"
        else
            log_warn "No log found for failed phase: ${_FAILED_PHASE}"
        fi
    else
        log_info "Error analysis script not found, skipping"
    fi
fi

# ═══ Summary ═══
log_step "Pipeline Summary"
echo "  Model:    ${MODEL_ID}"
echo "  Scheme:   ${SCHEME} / ${METHOD}"
echo "  Status:   ${PIPELINE_STATUS}"
echo "  Duration: ${PIPELINE_DURATION}s"
echo "  Output:   ${RUN_OUTPUT_DIR}"
if [[ ${#FAILED_STEPS[@]} -gt 0 ]]; then
    echo "  Failed:   ${FAILED_STEPS[*]}"
fi
if [[ -f "${RUN_OUTPUT_DIR}/run_report.md" ]]; then
    echo "  Report:   ${RUN_OUTPUT_DIR}/run_report.md"
fi

exit $([[ "$PIPELINE_STATUS" == "Finished" ]] && echo 0 || echo 1)
