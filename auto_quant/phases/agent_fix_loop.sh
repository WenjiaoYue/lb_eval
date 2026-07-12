#!/bin/bash
# agent_fix_loop.sh — Sourceable library for the agent-assisted fix loop.
#
# Provides:
#   agent_fix_loop <phase_name> <script_path> [args...]
#   save_lesson <phase> <error_context> <status> <solution_note>
#   search_lessons <phase> <error_text>
#   maybe_compact_lessons
#   push_lessons_to_git
#
# Required environment:
#   RUN_OUTPUT_DIR    — base output dir for this run
#   LESSONS_DIR       — path to lessons/ directory (git tracked)
#   MAX_FIX_ATTEMPTS  — max agent retry attempts (default: 3)
#   MODEL_ID, SCHEME, METHOD — for lesson metadata

# Guard against double-source
[[ -n "${_AGENT_FIX_LOOP_SOURCED:-}" ]] && return 0
_AGENT_FIX_LOOP_SOURCED=1

MAX_FIX_ATTEMPTS="${MAX_FIX_ATTEMPTS:-10}"
LESSONS_DIR="${LESSONS_DIR:-${LB_EVAL_REPO_DIR:-$(dirname "$0")/../lessons}}"

# ═══════════════════════════════════════════════════════════════════
# cleanup_stale_gpu_procs — kill leftover phase worker processes that may still
# be holding GPU memory, then wait for VRAM to actually release.
#
# Root cause this solves: after a phase fails (timeout / crash / agent-killed parent),
# a child quantize.py/evaluate.py can be orphaned and keep ~all VRAM allocated. The
# next run is then STARVED and silently falls back to CPU (hours of wasted compute).
#
# Safety: we match ONLY our own phase script paths, kill each PID explicitly (never by
# name-broad signals), and never touch ourselves. Gated by CLEANUP_STALE_GPU (default on).
# ═══════════════════════════════════════════════════════════════════
cleanup_stale_gpu_procs() {
    [ "${CLEANUP_STALE_GPU:-true}" = "true" ] || return 0

    local self_pid=$$
    local patterns=("phases/quantize.py" "phases/evaluate.py")
    local killed=0 pat pid comm

    for pat in "${patterns[@]}"; do
        # pgrep only LISTS pids; killing is done explicitly per-PID below.
        # Restrict to actual python worker processes: a bare -f match also hits our own
        # shell / command-substitution subshells (their cmdline contains the pattern
        # string) and the harness itself. Filtering comm=python* avoids killing them.
        for pid in $(pgrep -f "${pat}" 2>/dev/null || true); do
            [ "${pid}" = "${self_pid}" ] && continue
            kill -0 "${pid}" 2>/dev/null || continue
            comm=$(ps -o comm= -p "${pid}" 2>/dev/null | tr -d ' ')
            case "${comm}" in
                python|python3|python3.*) ;;
                *) continue ;;
            esac
            log_warn "Stale GPU worker still alive: PID=${pid} (${pat}) — terminating"
            kill "${pid}" 2>/dev/null || true
            killed=$((killed + 1))
        done
    done

    # Escalate any survivors after a grace period.
    if [ "${killed}" -gt 0 ]; then
        sleep 3
        for pat in "${patterns[@]}"; do
            for pid in $(pgrep -f "${pat}" 2>/dev/null || true); do
                [ "${pid}" = "${self_pid}" ] && continue
                kill -0 "${pid}" 2>/dev/null || continue
                comm=$(ps -o comm= -p "${pid}" 2>/dev/null | tr -d ' ')
                case "${comm}" in
                    python|python3|python3.*) ;;
                    *) continue ;;
                esac
                log_warn "PID=${pid} survived SIGTERM — sending SIGKILL"
                kill -9 "${pid}" 2>/dev/null || true
            done
        done
    fi

    # Wait for VRAM to actually free up (best-effort; needs nvidia-smi).
    command -v nvidia-smi >/dev/null 2>&1 || { [ "${killed}" -gt 0 ] && sleep 2; return 0; }
    local min_free_mb="${MIN_FREE_VRAM_MB:-2048}"
    local waited=0 max_wait="${GPU_FREE_WAIT_SEC:-30}" free_mb
    while [ "${waited}" -lt "${max_wait}" ]; do
        free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        [[ "${free_mb}" =~ ^[0-9]+$ ]] || break
        if [ "${free_mb}" -ge "${min_free_mb}" ]; then
            [ "${killed}" -gt 0 ] && log_ok "GPU VRAM released (${free_mb}MB free)"
            return 0
        fi
        log_info "Waiting for VRAM to free (${free_mb}MB free, need ${min_free_mb}MB)..."
        sleep 3
        waited=$((waited + 3))
    done
    return 0
}

# Stable location of this library and the shared error taxonomy, so the harness can
# REUSE the exact same deterministic classifier the post-mortem diagnosis uses.
_AFL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ERROR_ANALYSIS_DIR="${ERROR_ANALYSIS_DIR:-${_AFL_DIR}/../error_analysis}"

# ═══════════════════════════════════════════════════════════════════
# taxonomy_classify — L1 deterministic classification, REUSING error_analysis/taxonomy.py
#   (the same classify_error() the post-mortem diagnosis uses — single source of truth).
#   Reads an error-log file; prints:
#     line 1           : the taxonomy category token (or "unknown")
#     lines 2..N       : a ready-to-embed "prior" block for the agent prompt
#   This is a FAST, high-precision fast-path — it is NOT expected to cover every error.
#   Long-tail coverage is the agent's job (L2); unknowns fall back to text similarity.
# ═══════════════════════════════════════════════════════════════════
taxonomy_classify() {
    local errfile="$1"
    python3 - "$errfile" "${ERROR_ANALYSIS_DIR}" <<'PY' 2>/dev/null || echo "unknown"
import sys, os
errfile, ea_dir = sys.argv[1], sys.argv[2]
sys.path.insert(0, ea_dir)
try:
    from taxonomy import classify_error
except Exception:
    print("unknown"); sys.exit(0)
try:
    text = open(errfile, encoding="utf-8", errors="replace").read()
except OSError:
    text = ""
cat, info = classify_error(text)
print(cat)
desc = info.get("description", "")
guide = info.get("root_cause_guide", "")
if isinstance(guide, (list, tuple)):
    guide = " ".join(guide)
hints = info.get("workaround_hints", []) or []
print("- Category (pattern-based, MAY BE WRONG — verify or override): %s" % cat)
if desc:  print("- Description: %s" % desc)
if guide: print("- Root-cause guide: %s" % guide)
if hints: print("- Workaround hints: %s" % "; ".join(hints))
PY
}

# ═══════════════════════════════════════════════════════════════════
# logs_are_similar — L1.5 deterministic FALLBACK for drift when neither attempt got a
#   confident category (both "unknown"). Works on ARBITRARY error text with zero
#   enumeration: denoise (strip timestamps/HTTP/progress/paths, normalize numbers) then
#   compare with difflib. Exit 0 = same error, 1 = different, 2 = cannot tell.
# ═══════════════════════════════════════════════════════════════════
logs_are_similar() {
    python3 - "$1" "$2" "${DRIFT_SIM:-0.90}" <<'PY' 2>/dev/null
import sys, re, difflib
def denoise(p):
    try:
        t = open(p, encoding="utf-8", errors="replace").read()
    except OSError:
        return ""
    out = []
    for ln in t.splitlines():
        if re.search(r'HTTP Request|HTTP/1\.1|Client Error|Downloading|it/s\]|\|\s*\d+/\d+|Config was last written|allowlist contains|WARNING logging', ln):
            continue
        s = re.sub(r'^\S*\d{4}-\d\d-\d\dT[\d:.]+Z?\s*', '', ln)
        s = re.sub(r'\b\d{1,2}:\d{2}:\d{2}\b', '', s)
        s = re.sub(r'\[[A-Z]+\]', '', s)
        s = re.sub(r'0x[0-9a-fA-F]+', '0xADDR', s)
        s = re.sub(r'/[^\s:]+/', '/PATH/', s)
        s = re.sub(r'\d+\.\d+\s?[GMK]i?B', 'SIZE', s)
        s = re.sub(r'line \d+', 'line N', s)
        s = re.sub(r'\d+', 'N', s)
        s = s.strip()
        if s:
            out.append(s)
    return "\n".join(out)
a, b, thr = denoise(sys.argv[1]), denoise(sys.argv[2]), float(sys.argv[3])
if not a or not b:
    sys.exit(2)
r = difflib.SequenceMatcher(None, a, b).ratio()
sys.stderr.write("[drift] denoised similarity=%.3f (threshold=%.2f)\n" % (r, thr))
sys.exit(0 if r >= thr else 1)
PY
}

# ═══════════════════════════════════════════════════════════════════
# extract_progress — deepest quantized layer index seen in a log (else -1). Used as a
#   "real progress" override: if the re-run got FURTHER than before, it is NOT drift
#   even when the error class repeats.
# ═══════════════════════════════════════════════════════════════════
extract_progress() {
    local n
    n=$(grep -oE 'layers\.[0-9]+' "$1" 2>/dev/null | grep -oE '[0-9]+' | sort -n | tail -1)
    printf '%s' "${n:--1}"
}

# ═══════════════════════════════════════════════════════════════════
# agent_fix_loop — run a phase script, retry with agent on failure
# ═══════════════════════════════════════════════════════════════════
agent_fix_loop() {
    local phase_name="$1"
    local script_path="$2"
    shift 2
    local script_args=("$@")

    local max_attempts="${MAX_FIX_ATTEMPTS}"
    local attempt=0
    local prev_eff_class=""      # error class (agent's, else taxonomy's) from the previous attempt
    local prev_errfile=""        # previous attempt's error-tail file (similarity fallback)
    local drift_count=0          # consecutive attempts stuck on the same error class
    local max_progress=-1        # deepest quant layer reached so far (progress override)
    local phase_log="${RUN_OUTPUT_DIR}/logs/${phase_name}.log"
    local fix_log_dir="${RUN_OUTPUT_DIR}/logs/agent_fixes/${phase_name}"
    mkdir -p "$(dirname "${phase_log}")" "${fix_log_dir}"

    # Reuse ONE agent session across all attempts for this phase so the agent keeps
    # memory of what it already tried and does not repeat failed fixes.
    local fix_session_id="fix_${phase_name}_$$_$(date +%s)"

    # Snapshot whether CUDA was available BEFORE the fix loop. If it was, a fix that
    # loses CUDA is a regression — we must refuse to silently quantize on CPU.
    local cuda_was_available=false
    if python3 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        cuda_was_available=true
        log_info "CUDA available at start — GPU will be enforced across fix attempts"
    fi

    # First execution (deterministic script). Clear any leftover GPU workers first so
    # a leak from a prior phase/run can't starve this one onto CPU.
    cleanup_stale_gpu_procs
    log_step "Phase: ${phase_name}"
    bash "${script_path}" "${script_args[@]}" 2>&1 | tee "${phase_log}"
    local exit_code=${PIPESTATUS[0]}

    if [ $exit_code -eq 0 ]; then
        log_ok "${phase_name} succeeded"
        return 0
    fi

    log_warn "${phase_name} failed (exit=${exit_code}), entering agent fix loop"

    # Fix loop
    while [ $attempt -lt $max_attempts ]; do
        attempt=$((attempt + 1))
        log_step "Agent fix attempt ${attempt}/${max_attempts} for ${phase_name}"

        # 1. Extract error context and persist it to a per-attempt file (so drift can
        #    compare attempt N vs N-1 by FILE — never a file against itself).
        local error_tail errfile
        errfile="${fix_log_dir}/errtail_${attempt}.txt"
        error_tail=$(tail -100 "${phase_log}")
        printf '%s\n' "${error_tail}" > "${errfile}"

        # 2. L1 deterministic classification (REUSED taxonomy) → category + prior block.
        #    The category seeds drift detection; the prior block makes the AGENT start
        #    smarter (it gets the pattern-based guess + root-cause guide + hints, and is
        #    told it MAY BE WRONG and should verify/override).
        local classout cur_taxo_cat prior_block cur_progress
        classout=$(taxonomy_classify "${errfile}")
        cur_taxo_cat=$(printf '%s\n' "${classout}" | head -1)
        prior_block=$(printf '%s\n' "${classout}" | tail -n +2)
        cur_progress=$(extract_progress "${errfile}")
        log_info "L1 taxonomy class: ${cur_taxo_cat} (progress=layer ${cur_progress})"

        # 3. Load all lessons for agent context
        local lessons=""
        if [ -d "${LESSONS_DIR}" ]; then
            lessons=$(load_all_lessons 2>/dev/null || true)
        fi
        if [ -n "${lessons}" ]; then
            log_info "Loaded lessons for agent (let agent decide relevance)"
        else
            log_info "No lessons available"
        fi

        # 4. Build agent prompt (now seeded with the L1 taxonomy prior)
        local fix_prompt
        fix_prompt=$(build_fix_prompt "${phase_name}" "${error_tail}" "${lessons}" "${attempt}" "${prior_block}")

        # 5. Save prompt for audit
        local prompt_file="${fix_log_dir}/prompt_${attempt}.txt"
        printf '%s\n' "${fix_prompt}" > "${prompt_file}"

        # 6. Call the configured agent (same session across attempts → retains memory)
        local agent_log="${fix_log_dir}/attempt_${attempt}.log"
        # Mark a baseline so we can capture exactly what the agent changes (patch).
        local _patch_marker="${fix_log_dir}/.patch_marker_${attempt}"
        : > "${_patch_marker}"

        # Hybrid escalation: use the cheap PRIMARY backend (open-claw) for the first
        # AGENT_ESCALATE_AFTER attempts; if those fail, escalate to the stronger
        # ESCALATED backend (copilot) for the harder, non-routine case. Gated by
        # AGENT_ESCALATE=1; otherwise the single AGENT_BACKEND is used as-is.
        local _saved_backend="${AGENT_BACKEND:-openclaw}"
        if [ "${AGENT_ESCALATE:-0}" = "1" ]; then
            if [ "${attempt}" -gt "${AGENT_ESCALATE_AFTER:-2}" ]; then
                AGENT_BACKEND="${AGENT_ESCALATED:-copilot}"
                log_warn "Hybrid escalation: attempt ${attempt} > ${AGENT_ESCALATE_AFTER:-2} → switching to '${AGENT_BACKEND}' (harder case)"
            else
                AGENT_BACKEND="${AGENT_PRIMARY:-openclaw}"
                log_info "Hybrid: attempt ${attempt} → primary backend '${AGENT_BACKEND}'"
            fi
        fi

        # MULTI_AGENT=1 → two-agent flow (diagnoser → fixer); else single agent.
        if [ "${MULTI_AGENT:-0}" = "1" ]; then
            run_multiagent_fix "${fix_prompt}" "${agent_log}" "${fix_session_id}" "${attempt}" || true
        else
            run_agent_fix "${fix_prompt}" "${agent_log}" "${fix_session_id}" || true
        fi

        # 6·patch. Capture the agent's edits as a unified diff (bug-fix patch), so the
        # lesson stores WHAT CHANGED — not just a text summary. Sets LAST_PATCH_FILE.
        capture_fix_patch "${attempt}" "${_patch_marker}"
        # Restore the configured backend for non-agent code paths / next iteration logic.
        AGENT_BACKEND="${_saved_backend}"


        # Capture the agent's FULL structured diagnosis (analysis + fix) as JSON so every
        # lesson we write below carries the agent's ROOT_CAUSE / COMPONENT / EVIDENCE /
        # FIX_TIER — not just a grep'd fix line. Feeds L3 self-learning.
        local agent_analysis_json
        agent_analysis_json=$(extract_agent_analysis "${agent_log}")

        # 6b. Early stop: agent declared this failure UNFIXABLE → don't waste retries
        if grep -aiE 'VERDICT:[[:space:]*]*UNFIXABLE' "${agent_log}" >/dev/null 2>&1; then
            local unfix_reason
            unfix_reason=$(extract_agent_field "${agent_log}" "UNFIXABLE_REASON")
            unfix_reason="${unfix_reason:-declared UNFIXABLE by agent}"
            log_warn "Agent verdict: UNFIXABLE (${unfix_reason}). Aborting fix loop."
            save_lesson "${phase_name}" "${error_tail}" "unfixable" "UNFIXABLE: ${unfix_reason}" "${agent_analysis_json}"
            return 1
        fi

        # 6a. Drift / progress detection — 3-layer signal:
        #   PRIMARY  : the AGENT's semantic ERROR_CLASS (covers the long tail / new errors)
        #   FALLBACK : the L1 taxonomy category when the agent didn't emit a usable class
        #   TIE-BREAK: denoised text similarity when BOTH classes are unknown/missing
        #   OVERRIDE : deeper quant layer than before  → real progress, never drift
        #   FAIL-SAFE: if we cannot tell, CONTINUE (a false abort is the expensive failure)
        # We record the agent's class into the lesson (self-learning: recurring unknowns
        # can later be promoted into the taxonomy).
        local agent_class eff_class
        agent_class=$(extract_agent_field "${agent_log}" "ERROR_CLASS" | awk '{print $1}' \
            | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9_')
        if [ -n "${agent_class}" ] && [ "${agent_class}" != "unknown" ]; then
            eff_class="${agent_class}"      # PRIMARY: trust the agent's semantic label
        else
            eff_class="${cur_taxo_cat}"     # FALLBACK: deterministic taxonomy label
        fi
        log_info "Effective error class: ${eff_class} (agent='${agent_class:-none}', taxonomy='${cur_taxo_cat}')"

        if [ "${cur_progress}" -gt "${max_progress}" ] 2>/dev/null; then
            [ "${drift_count}" -gt 0 ] && log_info "Progress: reached layer ${cur_progress} (was ${max_progress}) — resetting drift"
            drift_count=0
        elif [ $attempt -gt 1 ]; then
            local same_error=""   # yes | no | "" (unknown)
            if [ -n "${eff_class}" ] && [ "${eff_class}" != "unknown" ] && [ -n "${prev_eff_class}" ] && [ "${prev_eff_class}" != "unknown" ]; then
                [ "${eff_class}" = "${prev_eff_class}" ] && same_error="yes" || same_error="no"
            elif [ -n "${prev_errfile}" ] && [ -f "${prev_errfile}" ]; then
                logs_are_similar "${errfile}" "${prev_errfile}"; local sim_rc=$?
                case "${sim_rc}" in 0) same_error="yes";; 1) same_error="no";; *) same_error="";; esac
            fi

            if [ "${same_error}" = "yes" ]; then
                drift_count=$((drift_count + 1))
                log_warn "Same error as previous attempt (class='${eff_class}', streak=${drift_count}/${DRIFT_THRESHOLD:-2})"
                if [ "${drift_count}" -ge "${DRIFT_THRESHOLD:-2}" ]; then
                    log_warn "Drift: error unchanged across ${drift_count} fixes. Aborting fix loop."
                    save_lesson "${phase_name}" "${error_tail}" "drift" "Stuck on '${eff_class}' for ${drift_count} attempts (agent_class='${agent_class:-none}')" "${agent_analysis_json}"
                    break
                fi
            elif [ "${same_error}" = "no" ]; then
                [ "${drift_count}" -gt 0 ] && log_info "Error changed ('${prev_eff_class}' → '${eff_class}') — fix made progress"
                drift_count=0
            fi
            # same_error == "" → FAIL-SAFE: neither abort nor reset; keep trying
        fi
        # Remember for the next iteration
        [ -n "${eff_class}" ] && [ "${eff_class}" != "unknown" ] && prev_eff_class="${eff_class}"
        prev_errfile="${errfile}"
        [ "${cur_progress}" -gt "${max_progress}" ] 2>/dev/null && max_progress="${cur_progress}"

        # 6c. GPU guard: a fix must NOT break CUDA. If GPU was available at start but is
        # now gone, refuse to silently fall back to a slow/OOM-prone CPU quantization run.
        # Feed the regression back so the agent restores CUDA on the next attempt.
        if [ "${cuda_was_available}" = "true" ] && [ "${REQUIRE_CUDA:-true}" = "true" ]; then
            if ! python3 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
                log_error "CUDA became UNAVAILABLE after agent fix (attempt ${attempt}) — refusing CPU re-run."
                {
                    echo "[harness] REGRESSION: torch.cuda.is_available() == False after your fix."
                    echo "[harness] This box HAS a GPU. Your fix broke CUDA — most likely a CPU-only torch"
                    echo "[harness] was installed, torch was reinstalled/downgraded, or CUDA_VISIBLE_DEVICES was cleared."
                    echo "[harness] RESTORE CUDA before anything else: reinstall the matching CUDA torch wheel,"
                    echo "[harness] unset/repair CUDA_VISIBLE_DEVICES, and verify: python3 -c 'import torch; assert torch.cuda.is_available()'"
                } | tee -a "${agent_log}"
                save_lesson "${phase_name}" "${error_tail}" "still_failing" "Fix broke CUDA (attempt ${attempt}); refused CPU re-run" "${agent_analysis_json}"
                phase_log="${agent_log}"
                continue
            fi
        fi

        # 6d. Cheap smoke test before the expensive full phase re-run.
        # run_smoke_test returns 0 if the smoke test passed OR none could be extracted
        # (fall back to the normal full re-run); non-zero only if an extracted test failed.
        if ! run_smoke_test "${agent_log}"; then
            log_warn "Smoke test failed after agent fix (attempt ${attempt}); skipping full re-run."
            save_lesson "${phase_name}" "${error_tail}" "still_failing" "Smoke test failed on attempt ${attempt}" "${agent_analysis_json}"
            phase_log="${agent_log}"
            continue
        fi

        # 7. Re-run phase script to verify
        # Clean up any orphaned GPU workers from the failed attempt (or from the agent's
        # own test runs) so this re-run isn't starved into a silent CPU fallback.
        cleanup_stale_gpu_procs
        log_info "Re-running ${phase_name} after agent fix..."
        local retry_log="${fix_log_dir}/retry_${attempt}.log"
        bash "${script_path}" "${script_args[@]}" 2>&1 | tee "${retry_log}"
        exit_code=${PIPESTATUS[0]}

        if [ $exit_code -eq 0 ]; then
            log_ok "${phase_name} fixed on attempt ${attempt}"
            # Extract agent's fix summary (first lines containing FIX_PLAN or actual commands)
            local fix_summary=""
            if [ -f "${agent_log}" ]; then
                fix_summary=$(grep -A3 "FIX_PLAN\|Fix applied\|Installing\|pip install\|Changing\|Setting" "${agent_log}" | head -5 | tr '\n' '; ')
            fi
            fix_summary="${fix_summary:-Agent fixed on attempt ${attempt}}"
            save_lesson "${phase_name}" "${error_tail}" "fixed" "${fix_summary}" "${agent_analysis_json}"
            return 0
        fi

        phase_log="${retry_log}"
        save_lesson "${phase_name}" "${error_tail}" "still_failing" "Attempt ${attempt} did not resolve" "${agent_analysis_json}"
    done

    log_error "${phase_name} failed after ${max_attempts} fix attempts"
    return 1
}

# ═══════════════════════════════════════════════════════════════════
# build_fix_prompt — construct the agent prompt for fixing a phase
# ═══════════════════════════════════════════════════════════════════
build_fix_prompt() {
    local phase="$1"
    local error="$2"
    local lessons="$3"
    local attempt="${4:-1}"
    local prior_block="${5:-}"

    local lessons_section=""
    if [ -n "${lessons}" ]; then
        lessons_section="## Historical Lessons (from past runs — decide which are relevant):
${lessons}
Review the lessons above and apply the most relevant fix for the current error."
    else
        lessons_section="## Historical Lessons:
No lessons available yet."
    fi

    local prior_section=""
    if [ -n "${prior_block}" ]; then
        prior_section="## Quick Classification (deterministic pattern match — a PRIOR, not the truth)
${prior_block}
Treat this as a starting hint. CONFIRM it against the traceback, and OVERRIDE it in your
ERROR_CLASS below if it is wrong or if the category is \`unknown\`.
"
    fi

    cat <<PROMPT
You are fixing a failed "${phase}" phase in the quantization pipeline.

## Error Output (last 100 lines):
${error}

${prior_section}
${lessons_section}

## MANDATORY PROTOCOL — fill this out BEFORE editing or installing anything

Use the \`error_analysis\` skill methodology: read the traceback BOTTOM-UP, locate the
EXACT file:line, then classify the failing component. You MUST print the block below
FIRST. Do NOT modify code or install packages until you have printed an EVIDENCE_RESULT
from a READ-ONLY command that actually supports your hypothesis. No guessing.

COMPONENT: <our_code|transformers|auto_round|torch|model_code|data|environment>
ERROR_CLASS: <ONE stable snake_case token naming THIS error's category. Reuse the taxonomy
             category shown in Quick Classification if it is correct; otherwise give a better
             existing token or a NEW snake_case name (e.g. shape_mismatch, meta_device_error,
             unrecognized_config_class). Use the SAME token every time the same underlying
             error recurs — this drives loop drift detection, so be consistent.>
ROOT_CAUSE_HYPOTHESIS: <one falsifiable sentence — the specific cause, NOT "maybe a version issue">
EVIDENCE_CMD: <a single read-only command that verifies the hypothesis>
EVIDENCE_RESULT: <paste the command's output>
VERDICT: <FIXABLE | UNFIXABLE>
UNFIXABLE_REASON: <required only if UNFIXABLE: e.g. multimodal-unsupported / corrupt weights / needs torch downgrade>
FIX_TIER: <config | upgrade | workaround | patch>   # always try the LOWEST tier that works
FIX_PLAN: <3 lines max — what you will change and why it fixes the ROOT CAUSE (not the symptom)>
SMOKE_TEST: <ONE fast command (NOT the full phase) that proves the fix works, e.g. a tokenizer/model load>

## Rules for this protocol:
- If VERDICT is UNFIXABLE: print the block and STOP. Do NOT attempt a fix. The pipeline will halt this phase (no wasted retries).
- Prefer the LOWEST FIX_TIER. Patching source code is a last resort.
- Escalate tiers only with evidence that the lower tier cannot work.
- After applying the fix, RUN your SMOKE_TEST yourself and show its output before finishing.
- GPU IS REQUIRED. This environment HAS CUDA and the re-run MUST run on GPU. Never force CPU
  (no \`device='cpu'\`, no \`device_map='cpu'\`, do not edit quantize.py to use CPU), never clear
  \`CUDA_VISIBLE_DEVICES\`, and never install a CPU-only torch. After any \`pip install\`, confirm
  CUDA still works: \`python3 -c "import torch; assert torch.cuda.is_available()"\`.
- This is attempt ${attempt}. Any earlier attempts are in your session history — do NOT repeat a fix that already failed; try a different hypothesis.

## Key Technique: Patching Model Custom Code

If the traceback shows files in \`~/.cache/huggingface/modules/transformers_modules/\`, that is the
MODEL'S CUSTOM CODE that was downloaded from HuggingFace. **YOU CAN AND SHOULD EDIT THESE FILES.**

Common fixes for model custom code:
- dtype mismatch (\`.float()\` mixed with bfloat16): Replace \`.float()\` with \`.to(other_tensor.dtype)\`
- Missing device: Add \`device=hidden_states.device\` to tensor creation
- Invalid regex: Fix the regex pattern in the model file
- Missing imports: Add the import or install the package

Example: If you see:
  File "/root/.cache/huggingface/modules/transformers_modules/Org/Model/hash/model.py", line 147
    h = h + torch.matmul(compressed[:, k:k+valid_len, :].float(), proj.t())
  RuntimeError: expected m1 and m2 to have the same dtype

Fix: Edit that file, change \`.float()\` to \`.to(proj.dtype)\`

## Constraints:
- Do NOT reinstall or downgrade torch (it will break CUDA).
- **CUDA MUST STAY WORKING.** The re-run quantizes on GPU. If your fix leaves the box on CPU
  (torch.cuda.is_available() == False), the pipeline will REJECT the CPU run as a failure.
  - Do NOT install a CPU-only torch wheel; if you must (re)install torch, use the matching CUDA wheel.
  - Do NOT set \`CUDA_VISIBLE_DEVICES=""\`; do NOT pass \`device='cpu'\` / \`device_map='cpu'\`.
  - Beware: \`pip install -U auto-round\`/\`transformers\` can pull a CPU torch — re-check CUDA after installing.
- Do NOT modify the evaluation tasks or expected output format
- Keep fixes minimal and targeted — change only what's needed
- If you need to install a package, use: pip install <package>
- Multimodal/VL models are NOT auto-rejected: AutoRound can quantize the LM backbone of VL/MLLM
  models (Qwen-VL, LLaVA, InternVL, Qwen3-VL). For image-processor / preprocessor_config / new-arch
  errors, try \`pip install -U auto-round transformers\` first. Only declare VERDICT: UNFIXABLE for a
  model with NO text-generation backbone (pure vision/audio encoder).
- Working directory: ${RUN_OUTPUT_DIR}
- Model: ${MODEL_ID}
PROMPT
}

# ═══════════════════════════════════════════════════════════════════
# run_agent_fix — dispatch the fix call to the configured agent backend.
#   Selected via AGENT_BACKEND (config.env / env): openclaw (default) | copilot
# ═══════════════════════════════════════════════════════════════════
run_agent_fix() {
    local backend
    backend=$(printf '%s' "${AGENT_BACKEND:-openclaw}" | tr '[:upper:]' '[:lower:]')
    case "${backend}" in
        copilot)
            run_copilot_fix "$@"
            ;;
        openclaw|"")
            run_openclaw_fix "$@"
            ;;
        *)
            log_warn "Unknown AGENT_BACKEND='${backend}', falling back to openclaw"
            run_openclaw_fix "$@"
            ;;
    esac
}

# ═══════════════════════════════════════════════════════════════════
# run_multiagent_fix — two-agent bug-analysis flow (P2), backend-agnostic.
#
#   Agent 1 (DIAGNOSER): analysis ONLY. Reads logs/code, does NOT edit anything.
#                        Emits COMPONENT / ROOT_CAUSE / FIX_PLAN / SMOKE_TEST.
#   Agent 2 (FIXER):     receives the diagnosis and actually applies the fix.
#
#   Both agents use the configured backend (openclaw/copilot) via run_agent_fix,
#   so this works for either. Enabled with MULTI_AGENT=1. The fixer's output goes
#   to the normal ${agent_log} so all downstream extraction/patch capture works
#   unchanged.
# ═══════════════════════════════════════════════════════════════════
run_multiagent_fix() {
    local base_prompt="$1"
    local log_file="$2"
    local session_id="$3"
    local attempt="${4:-1}"

    local diag_log="${fix_log_dir}/diagnose_${attempt}.log"

    local diagnose_prompt
    diagnose_prompt="ROLE: You are the DIAGNOSER in a two-agent repair team.
Analyze the failure below. DO NOT edit, install, or change ANYTHING — analysis only.
Output a concise structured diagnosis with these labeled lines:
COMPONENT: <auto_round|transformers|torch|model_code|lm_eval|infrastructure>
ROOT_CAUSE: <1-3 sentences, traced bottom-up from the traceback>
FIX_PLAN: <numbered, concrete steps the fixer should apply>
SMOKE_TEST: <a single shell/python command that will confirm the fix>

--- FAILURE CONTEXT ---
${base_prompt}"

    log_info "  [multi-agent] Agent 1/2: DIAGNOSER (session=${session_id}_diag)"
    run_agent_fix "${diagnose_prompt}" "${diag_log}" "${session_id}_diag" || true

    # Extract the diagnosis so we can hand it to the fixer.
    local diag_component diag_root diag_plan diag_smoke
    diag_component=$(extract_agent_field "${diag_log}" "COMPONENT")
    diag_root=$(extract_agent_field "${diag_log}" "ROOT_CAUSE")
    diag_plan=$(extract_agent_field "${diag_log}" "FIX_PLAN")
    diag_smoke=$(extract_agent_field "${diag_log}" "SMOKE_TEST")

    local fixer_prompt
    fixer_prompt="ROLE: You are the FIXER in a two-agent repair team.
A diagnoser already analyzed this failure. Apply the fix by EDITING files / installing
as needed, then verify with the smoke test. Prefer the LOWEST fix tier. Preserve CUDA.

--- DIAGNOSER'S ANALYSIS ---
COMPONENT: ${diag_component:-unknown}
ROOT_CAUSE: ${diag_root:-see full analysis below}
FIX_PLAN: ${diag_plan:-derive from root cause}
SMOKE_TEST: ${diag_smoke:-add your own}

(Full diagnoser log: ${diag_log})

--- ORIGINAL FAILURE CONTEXT ---
${base_prompt}

After applying the fix, emit the same structured fields (COMPONENT / ERROR_CLASS /
ROOT_CAUSE / FIX_TIER / FIX_PLAN / SMOKE_TEST) so the pipeline can record the lesson."

    log_info "  [multi-agent] Agent 2/2: FIXER (session=${session_id})"
    run_agent_fix "${fixer_prompt}" "${log_file}" "${session_id}" || true
    return 0
}

# ═══════════════════════════════════════════════════════════════════
# run_openclaw_fix — call OpenClaw agent with the fix prompt
# ═══════════════════════════════════════════════════════════════════
run_openclaw_fix() {
    local prompt="$1"
    local log_file="$2"
    local session_id_arg="${3:-}"

    if ! command -v openclaw >/dev/null 2>&1; then
        log_warn "openclaw not found, skipping agent fix"
        echo "openclaw not available" > "${log_file}"
        return 1
    fi

    local timeout="${AGENT_TIMEOUT:-600}"
    local session_id="${session_id_arg:-fix_${phase_name:-unknown}_$$_$(date +%s)}"
    local sessions_dir="${OPENCLAW_SESSIONS_DIR:-/root/.openclaw/agents/main/sessions}"
    local session_file="${sessions_dir}/${session_id}.jsonl"

    log_info "Calling openclaw agent (session=${session_id}, timeout=${timeout}s)..."
    log_info "  Session file: ${session_file}"

    # Background progress reporter — prints elapsed time + session size every 30s
    local _progress_pid=""
    (
        local _start=$SECONDS
        while true; do
            sleep 30
            local elapsed=$(( SECONDS - _start ))
            local session_lines=0
            [[ -f "${session_file}" ]] && session_lines=$(wc -l < "${session_file}" 2>/dev/null || echo 0)
            log_info "  [agent running ${elapsed}s] session: ${session_lines} messages"
        done
    ) &
    _progress_pid=$!

    timeout "${timeout}" openclaw agent --local \
        --session-id "${session_id}" \
        --message "${prompt}" \
        --timeout "${timeout}" \
        2>&1 | tee "${log_file}" || {
        local rc=$?
        if [ $rc -eq 124 ]; then
            echo "[TIMEOUT] Agent exceeded ${timeout}s" >> "${log_file}"
            log_warn "Agent timed out after ${timeout}s"
        fi
    }

    # Stop progress reporter
    if [[ -n "${_progress_pid}" ]]; then
        kill "${_progress_pid}" 2>/dev/null || true
        wait "${_progress_pid}" 2>/dev/null || true
    fi

    # Print session summary to auto.log
    if [[ -f "${session_file}" ]]; then
        local msg_count tool_count
        msg_count=$(grep -c '"type":"message"\|"type": "message"' "${session_file}" 2>/dev/null || echo 0)
        tool_count=$(grep -c '"tool_use"\|"tool_call"' "${session_file}" 2>/dev/null || echo 0)
        log_info "  Agent session complete: ${msg_count} messages, ${tool_count} tool calls"
    fi

    return 0
}

# ═══════════════════════════════════════════════════════════════════
# run_copilot_fix — call the GitHub Copilot CLI agent with the fix prompt
#
#   Drop-in replacement for run_openclaw_fix. Uses the agentic Copilot CLI
#   (`copilot -p "<prompt>"`) which can read files, edit code, run shell
#   commands and install packages autonomously.
#
#   Session memory: Copilot CLI keeps its own session store. To mirror
#   OpenClaw's single-session reuse (so the agent remembers what it already
#   tried across attempts), we pass `--continue` on every attempt after the
#   first for a given logical session id (tracked via a marker file).
#
#   Config (config.env / env):
#     COPILOT_BIN           binary name (default: copilot)
#     COPILOT_GITHUB_TOKEN  GitHub token for headless auth (fine-grained PAT with
#                           "Copilot Requests"; or GH_TOKEN / GITHUB_TOKEN). If
#                           empty, falls back to an interactive `copilot login`.
#     COPILOT_MODEL         optional --model override (Copilot backend model)
#     COPILOT_TOOL_ARGS     tool-permission args (default: --allow-all-tools)
#     AGENT_TIMEOUT         per-call timeout seconds (default: 600)
# ═══════════════════════════════════════════════════════════════════
run_copilot_fix() {
    local prompt="$1"
    local log_file="$2"
    local session_id_arg="${3:-}"

    local copilot_bin="${COPILOT_BIN:-copilot}"
    if ! command -v "${copilot_bin}" >/dev/null 2>&1; then
        log_warn "${copilot_bin} not found, skipping agent fix (install @github/copilot)"
        echo "copilot CLI not available" > "${log_file}"
        return 1
    fi

    # Headless auth: Copilot CLI reads COPILOT_GITHUB_TOKEN / GH_TOKEN / GITHUB_TOKEN
    # from its environment. config.env is plain-sourced (not auto-exported), so we
    # export the token here to make it visible to the copilot child process.
    # If none is set, the CLI falls back to a prior interactive `copilot login`.
    if [[ -n "${COPILOT_GITHUB_TOKEN:-}" ]]; then
        export COPILOT_GITHUB_TOKEN
    elif [[ -n "${GH_TOKEN:-}" ]]; then
        export GH_TOKEN
    elif [[ -n "${GITHUB_TOKEN:-}" ]]; then
        export GITHUB_TOKEN
    else
        log_info "  No COPILOT_GITHUB_TOKEN/GH_TOKEN set; relying on prior 'copilot login'"
    fi

    local timeout="${AGENT_TIMEOUT:-600}"
    local session_id="${session_id_arg:-fix_${phase_name:-unknown}_$$_$(date +%s)}"

    # Resume the same logical session on attempts after the first so the agent
    # retains memory of failed fixes (parity with OpenClaw's single session).
    local state_dir="${COPILOT_STATE_DIR:-${RUN_OUTPUT_DIR}/logs/copilot_state}"
    mkdir -p "${state_dir}"
    local marker="${state_dir}/${session_id}.started"
    local resume_args=()
    [[ -f "${marker}" ]] && resume_args+=(--continue)

    # Optional model override. NOTE: Copilot CLI uses its own model backend and
    # generally cannot be pointed at an arbitrary provider (e.g. minimax).
    local model_args=()
    [[ -n "${COPILOT_MODEL:-}" ]] && model_args+=(--model "${COPILOT_MODEL}")

    # Tool permissions — allow autonomous execution by default.
    # shellcheck disable=SC2206
    local tool_args=(${COPILOT_TOOL_ARGS:---allow-all-tools})

    log_info "Calling Copilot CLI agent (session=${session_id}, resume=${resume_args[*]:-no}, timeout=${timeout}s)..."

    # Background progress reporter — prints elapsed time + log size every 30s
    local _progress_pid=""
    (
        local _start=$SECONDS
        while true; do
            sleep 30
            local elapsed=$(( SECONDS - _start ))
            local lines=0
            [[ -f "${log_file}" ]] && lines=$(wc -l < "${log_file}" 2>/dev/null || echo 0)
            log_info "  [agent running ${elapsed}s] log: ${lines} lines"
        done
    ) &
    _progress_pid=$!

    NO_COLOR=1 timeout "${timeout}" "${copilot_bin}" \
        "${resume_args[@]}" \
        "${model_args[@]}" \
        "${tool_args[@]}" \
        -p "${prompt}" \
        2>&1 | tee "${log_file}" || {
        local rc=$?
        if [ $rc -eq 124 ]; then
            echo "[TIMEOUT] Agent exceeded ${timeout}s" >> "${log_file}"
            log_warn "Agent timed out after ${timeout}s"
        fi
    }

    # Stop progress reporter
    if [[ -n "${_progress_pid}" ]]; then
        kill "${_progress_pid}" 2>/dev/null || true
        wait "${_progress_pid}" 2>/dev/null || true
    fi

    # Mark the session so subsequent attempts resume it.
    touch "${marker}"

    log_info "  Agent turn complete (log: $(wc -l < "${log_file}" 2>/dev/null || echo 0) lines)"
    return 0
}

# ═══════════════════════════════════════════════════════════════════
# extract_agent_field — pull a labeled single-line field from agent output
#   Tolerates markdown bold (**FIELD:**) and leading/trailing whitespace.
# ═══════════════════════════════════════════════════════════════════
extract_agent_field() {
    local log="$1"
    local field="$2"
    [ -f "${log}" ] || return 0
    grep -aiE "${field}:" "${log}" 2>/dev/null \
        | head -1 \
        | sed -E "s/.*${field}:[[:space:]]*//I" \
        | sed -E 's/\*+//g; s/^[[:space:]]*//; s/[[:space:]]*$//; s/`//g'
}

# ═══════════════════════════════════════════════════════════════════
# extract_agent_analysis — capture the agent's WHOLE structured diagnosis block
#   (COMPONENT / ERROR_CLASS / ROOT_CAUSE / EVIDENCE / VERDICT / FIX_TIER / FIX_PLAN)
#   as a compact JSON object, so the lesson stores the agent's ANALYSIS — not just a
#   grep'd fix line. Multiline field values (e.g. FIX_PLAN) are captured up to the next
#   known label. Prints "{}" if the log is missing/empty.
# ═══════════════════════════════════════════════════════════════════
extract_agent_analysis() {
    local agent_log="$1"
    [ -f "${agent_log}" ] || { echo '{}'; return 0; }
    AGENT_LOG_PATH="${agent_log}" python3 <<'PYEOF'
import os, re, json

try:
    log = open(os.environ["AGENT_LOG_PATH"], encoding="utf-8", errors="replace").read()
except OSError:
    print("{}"); raise SystemExit

LABELS = ["COMPONENT", "ERROR_CLASS", "ROOT_CAUSE_HYPOTHESIS", "EVIDENCE_CMD",
          "EVIDENCE_RESULT", "VERDICT", "UNFIXABLE_REASON", "FIX_TIER", "FIX_PLAN",
          "SMOKE_TEST"]


def block(name, maxlen=400):
    others = "|".join(l for l in LABELS if l != name)
    m = re.search(rf'^\s*{name}\s*:\s*(.*?)(?=^\s*(?:{others})\s*:|\Z)',
                  log, re.MULTILINE | re.DOTALL)
    if not m:
        return ""
    val = re.sub(r'`', '', m.group(1))
    val = re.sub(r'\*+', '', val)
    val = re.sub(r'\s+', ' ', val).strip()
    # Drop unfilled placeholders like "<one falsifiable sentence ...>"
    if val.startswith('<') and val.endswith('>'):
        return ""
    return val[:maxlen]


err_class = block("ERROR_CLASS", 60)
if err_class:
    err_class = re.sub(r'[^a-z0-9_]', '', err_class.split()[0].lower()) if err_class.split() else ""

out = {
    "component": block("COMPONENT", 60),
    "error_class": err_class,
    "root_cause": block("ROOT_CAUSE_HYPOTHESIS", 400),
    "evidence": block("EVIDENCE_RESULT", 300),
    "verdict": block("VERDICT", 20),
    "fix_tier": block("FIX_TIER", 40),
    "fix_plan": block("FIX_PLAN", 400),
}
print(json.dumps({k: v for k, v in out.items() if v}, ensure_ascii=False))
PYEOF
}

# ═══════════════════════════════════════════════════════════════════
# run_smoke_test — run the agent's suggested SMOKE_TEST for cheap verification
#   Returns 0 if the smoke test passed OR no runnable test could be extracted
#   (caller then falls back to the normal full phase re-run).
#   Returns non-zero ONLY when an extracted command actually ran and failed.
# ═══════════════════════════════════════════════════════════════════
run_smoke_test() {
    local agent_log="$1"
    local cmd
    cmd=$(extract_agent_field "${agent_log}" "SMOKE_TEST")

    # Empty, placeholder (<...>), or missing → fall back to full re-run
    if [ -z "${cmd}" ] || printf '%s' "${cmd}" | grep -q '<'; then
        return 0
    fi
    # Only run things that look like an actual command; otherwise fall back
    case "${cmd}" in
        python3*|python*|pip*|uv*|bash*|./*) : ;;
        *) return 0 ;;
    esac

    log_info "Running agent smoke test: ${cmd}"
    if timeout "${SMOKE_TEST_TIMEOUT:-180}" bash -c "${cmd}" >>"${agent_log}" 2>&1; then
        log_ok "Smoke test passed — proceeding to full re-run"
        return 0
    fi
    return 1
}

# ═══════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════
# capture_fix_patch — record WHAT the agent changed as a unified diff.
#
#   Two complementary sources (both backends: openclaw / copilot):
#     1. git repos in PATCH_CAPTURE_DIRS → `git diff` (tracked-file edits)
#     2. any file modified after the per-attempt marker → snapshot diff via
#        a stored baseline copy is too heavy, so we record the file list plus,
#        for small text files, their current content as an "added" hunk.
#
#   Config:
#     PATCH_CAPTURE_DIRS  colon-separated dirs to watch. Default: auto-round
#                         source (if present), the phase scripts dir, and the
#                         run output dir. HF cache edits are captured when the
#                         cache dir is included here.
#     PATCH_MAX_BYTES     cap on stored patch size (default 200000).
#
#   Output: ${fix_log_dir}/patch_${attempt}.diff ; sets global LAST_PATCH_FILE.
# ═══════════════════════════════════════════════════════════════════
LAST_PATCH_FILE=""
capture_fix_patch() {
    local attempt="$1"
    local marker="$2"
    LAST_PATCH_FILE=""

    local out="${fix_log_dir}/patch_${attempt}.diff"
    local max_bytes="${PATCH_MAX_BYTES:-200000}"

    # Default watch set: auto-round source (common edit target), phase scripts, run dir.
    local default_dirs=()
    [[ -n "${AUTO_ROUND_SRC_DIR:-}" && -d "${AUTO_ROUND_SRC_DIR}" ]] && default_dirs+=("${AUTO_ROUND_SRC_DIR}")
    [[ -d "${_AFL_DIR:-}" ]] && default_dirs+=("${_AFL_DIR}")
    [[ -n "${RUN_OUTPUT_DIR:-}" && -d "${RUN_OUTPUT_DIR}" ]] && default_dirs+=("${RUN_OUTPUT_DIR}")
    local dirs_spec="${PATCH_CAPTURE_DIRS:-$(IFS=:; echo "${default_dirs[*]}")}"

    {
        echo "# Fix patch — attempt ${attempt}"
        echo "# Phase: ${phase_name:-unknown}  Model: ${MODEL_ID:-unknown}  Backend: ${AGENT_BACKEND:-openclaw}"
        echo "# Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo
    } > "${out}"

    local IFS=':'
    local d had_content=0
    for d in ${dirs_spec}; do
        [[ -z "${d}" || ! -d "${d}" ]] && continue

        # 1. git-tracked edits
        if git -C "${d}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
            local gdiff
            gdiff=$(git -C "${d}" diff 2>/dev/null || true)
            if [[ -n "${gdiff}" ]]; then
                { echo "### git diff: ${d}"; echo '```diff'; echo "${gdiff}"; echo '```'; echo; } >> "${out}"
                had_content=1
            fi
        fi

        # 2. non-git edits: files modified after the marker (skip logs/binaries)
        local changed
        changed=$(find "${d}" -type f -newer "${marker}" \
            ! -path '*/.git/*' ! -path '*/logs/*' ! -name '*.log' \
            ! -name '*.jsonl' ! -name '.patch_marker_*' 2>/dev/null | head -50 || true)
        if [[ -n "${changed}" ]]; then
            { echo "### files modified after agent run in ${d}:"; echo "${changed}" | sed 's/^/#   /'; echo; } >> "${out}"
            had_content=1
        fi
    done

    if [[ "${had_content}" -eq 0 ]]; then
        echo "# (no file changes detected — fix may be env/config only)" >> "${out}"
    fi

    # Enforce size cap
    if [[ -f "${out}" ]]; then
        local sz
        sz=$(wc -c < "${out}" 2>/dev/null || echo 0)
        if [[ "${sz}" -gt "${max_bytes}" ]]; then
            head -c "${max_bytes}" "${out}" > "${out}.trunc" && mv "${out}.trunc" "${out}"
            echo -e "\n# [truncated at ${max_bytes} bytes]" >> "${out}"
        fi
        LAST_PATCH_FILE="${out}"
        log_info "  Captured fix patch → ${out}"
    fi
}

# ═══════════════════════════════════════════════════════════════════
# save_lesson — persist a lesson to the JSONL file
# ═══════════════════════════════════════════════════════════════════
save_lesson() {
    local phase="$1"
    local error_context="$2"
    local status="$3"
    local solution_note="$4"
    local agent_analysis="${5:-}"   # optional: agent's structured diagnosis as JSON
                                    # (or a bare snake_case class token, for back-compat)

    local lessons_file="${LESSONS_DIR}/${phase}.jsonl"
    mkdir -p "${LESSONS_DIR}"

    # Pass error_context via env var (not stdin, which conflicts with heredoc)
    LESSON_ERROR_CONTEXT="${error_context}" LESSON_TAXONOMY_DIR="${ERROR_ANALYSIS_DIR}" LESSON_AGENT_ANALYSIS="${agent_analysis}" LESSON_PATCH_FILE="${LAST_PATCH_FILE:-}" python3 - "${phase}" "${status}" "${solution_note}" "${MODEL_ID:-unknown}" "${SCHEME:-W4A16}" "${METHOD:-RTN}" "${lessons_file}" <<'PYEOF'
import json
import sys
import os
import datetime
import re

phase = sys.argv[1]
status = sys.argv[2]
solution_note = sys.argv[3]
model_id = sys.argv[4]
scheme = sys.argv[5]
method = sys.argv[6]
lessons_file = sys.argv[7]

error_context = os.environ.get("LESSON_ERROR_CONTEXT", "")

# Reuse the shared taxonomy: denoise + deterministic classification. This is the SAME
# classifier the drift detector and post-mortem diagnosis use, so a lesson's category is
# consistent across the whole pipeline. Degrade gracefully if the import fails.
sys.path.insert(0, os.environ.get("LESSON_TAXONOMY_DIR", ""))
try:
    from taxonomy import _strip_noise, classify_error
except Exception:
    def _strip_noise(t):
        return t

    def classify_error(t):
        return "unknown", {}

# Strip a leading timestamp / log-level prefix so signatures are stable across runs
# (e.g. "15:51:56 [ERROR] Quantization failed: X" and the same error an hour later
# must produce the SAME signature so dedup works).
_PREFIX_RE = re.compile(
    r'^\s*'
    r'(?:\d{4}-\d{2}-\d{2}[T ])?'              # optional ISO date
    r'(?:\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)?'      # optional HH:MM:SS(.ms)
    r'(?:\s*[Zz]|\s*[+-]\d{2}:?\d{2})?'        # optional timezone
    r'\s*(?:\[[A-Za-z]+\]|[A-Z]{3,}:)?\s*'     # optional [ERROR] / ERROR:
)


def _clean(line):
    return _PREFIX_RE.sub('', line).strip()


# Python's real fault is the LAST exception line of a traceback, not the first line that
# merely mentions "error". Prefer the deepest concrete exception; then a wrapper line that
# actually carries a message; then the last meaningful denoised line.
_EXC_RE = re.compile(r'\b([A-Za-z_][\w.]*(?:Error|Exception|Warning)|OSError)\b\s*:\s*\S')
_WRAP_RE = re.compile(r'\b(?:failed|error)\b\s*[:\-]\s*(\S.+)', re.I)


def extract_signature(text):
    denoised = _strip_noise(text) or text
    lines = [l for l in denoised.splitlines() if l.strip()]
    exc = [_clean(l) for l in lines if _EXC_RE.search(_clean(l))]
    if exc:
        return exc[-1][:150]
    for l in reversed(lines):
        c = _clean(l)
        m = _WRAP_RE.search(c)
        if m and m.group(1).strip():
            return c[:150]
    return _clean(lines[-1])[:150] if lines else "unknown error"


error_signature = extract_signature(error_context)

# Persist the deterministic category at write time -> enables coverage measurement and
# L3 self-learning (promoting recurring "unknown" categories into the taxonomy later).
try:
    error_category = classify_error(error_context)[0]
except Exception:
    error_category = "unknown"

# The agent's semantic ERROR_CLASS (may be a NEW category the taxonomy doesn't know yet).
# This is the raw material for L3: when taxonomy says "unknown" but the agent consistently
# assigns the same label to a recurring error, promote_lessons.py can learn a signature.
# Arg is a JSON blob of the agent's whole diagnosis (or a bare class token for back-compat).
_raw_analysis = os.environ.get("LESSON_AGENT_ANALYSIS", "").strip()
agent = {}
if _raw_analysis:
    try:
        parsed = json.loads(_raw_analysis)
        if isinstance(parsed, dict):
            agent = parsed
    except ValueError:
        # Back-compat: a bare "error_class" token rather than JSON
        agent = {"error_class": _raw_analysis}

agent_category = re.sub(r'[^a-z0-9_]', '', str(agent.get("error_class", "")).strip().lower())
agent_root_cause = str(agent.get("root_cause", ""))[:400]
agent_component = str(agent.get("component", ""))[:60]
agent_evidence = str(agent.get("evidence", ""))[:300]
agent_fix_tier = str(agent.get("fix_tier", ""))[:40]
# Prefer the agent's FIX_PLAN as the solution when the caller's note is thin/placeholder.
agent_fix_plan = str(agent.get("fix_plan", ""))[:400]
if agent_fix_plan and (not solution_note or len(solution_note) < 15):
    solution_note = agent_fix_plan

# Extract keywords from the cleaned signature
words = re.findall(r'[a-zA-Z]{4,}', error_signature.lower())
keywords = list(dict.fromkeys(words))[:5]  # unique, ordered

# Full traceback (last 50 lines, denoised so 404/progress chatter doesn't crowd it out)
traceback_lines = (_strip_noise(error_context) or error_context).strip().splitlines()[-50:]
error_traceback = "\n".join(traceback_lines)

# Bug-fix patch (unified diff of what the agent changed), captured by capture_fix_patch().
patch_file = os.environ.get("LESSON_PATCH_FILE", "").strip()
patch_text = ""
patch_has_changes = False
if patch_file and os.path.isfile(patch_file):
    try:
        patch_text = open(patch_file, encoding="utf-8", errors="replace").read()
    except OSError:
        patch_text = ""
    # A patch is "real" if it contains an actual diff/file-change section.
    patch_has_changes = ("```diff" in patch_text) or ("files modified after" in patch_text)
# Keep the lesson compact: store a bounded excerpt + a path to the full patch.
patch_excerpt = patch_text[:6000]

lesson = {
    "id": f"lesson-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "phase": phase,
    "error_signature": error_signature,
    "error_category": error_category,
    "agent_category": agent_category,
    "agent_root_cause": agent_root_cause,
    "agent_component": agent_component,
    "agent_evidence": agent_evidence,
    "fix_tier": agent_fix_tier,
    "error_traceback": error_traceback,
    "error_keywords": keywords,
    "model": model_id,
    "scheme": scheme,
    "method": method,
    "solution": solution_note,
    "patch": patch_excerpt,
    "patch_file": patch_file,
    "patch_has_changes": patch_has_changes,
    "status": status,
    "verified_count": 1,
    "source_tasks": [f"{model_id}_{scheme}_{method}"],
}

with open(lessons_file, "a") as f:
    f.write(json.dumps(lesson, ensure_ascii=False) + "\n")

print(f"[lesson] Saved: [{status}] {error_signature[:80]}")
print(f"[lesson]   Solution: {solution_note}")
PYEOF
}

# ═══════════════════════════════════════════════════════════════════
# load_all_lessons — load all lessons as text for agent to decide relevance
# ═══════════════════════════════════════════════════════════════════
load_all_lessons() {
    [ ! -d "${LESSONS_DIR}" ] && return 0

    python3 - "${LESSONS_DIR}" <<'PYEOF'
import json
import sys
from pathlib import Path

lessons_dir = Path(sys.argv[1])
lessons = []

for fpath in sorted(lessons_dir.glob("*.jsonl")):
    try:
        with open(fpath) as f:
            for line in f:
                if not line.strip():
                    continue
                lesson = json.loads(line)
                # Only load actionable lessons (fixed/verified/seed) plus known-unfixable
                # verdicts so the agent can stop early on a previously-hopeless error.
                if lesson.get("status") in ("fixed", "seed", "verified", "unfixable"):
                    lessons.append(lesson)
    except (FileNotFoundError, json.JSONDecodeError):
        continue

# Deduplicate by error_signature
seen = set()
unique = []
for les in lessons:
    sig = les.get("error_signature", "")
    if sig not in seen:
        seen.add(sig)
        unique.append(les)

# Sort by verified_count (most reliable first), cap at 10 to avoid huge prompts
unique.sort(key=lambda x: x.get("verified_count", 0), reverse=True)
for i, les in enumerate(unique[:10], 1):
    verified = les.get("verified_count", 0)
    phase = les.get("phase", "?")
    sig = les.get("error_signature", "")[:120]
    solution = les.get("solution", "")
    notes = les.get("notes", "")
    print(f"Lesson {i} [phase={phase}, verified={verified}x]:")
    print(f"  Error: {sig}")
    print(f"  Solution: {solution}")
    if notes:
        print(f"  Notes: {notes}")
    print()
PYEOF
}

# ═══════════════════════════════════════════════════════════════════
# maybe_compact_lessons — compact if > 50 entries
# ═══════════════════════════════════════════════════════════════════
maybe_compact_lessons() {
    local compact_script="${LESSONS_DIR}/compact_lessons.py"
    [ ! -f "${compact_script}" ] && return 0

    for f in "${LESSONS_DIR}"/*.jsonl; do
        [ ! -f "$f" ] && continue
        local count
        count=$(wc -l < "$f")
        if [ "$count" -gt 50 ]; then
            log_info "Compacting lessons (${count} entries in $(basename "$f"))..."
            python3 "${compact_script}" "${LESSONS_DIR}"
            break
        fi
    done
}

# ═══════════════════════════════════════════════════════════════════
# push_lessons_to_git — commit and push lessons
# ═══════════════════════════════════════════════════════════════════
push_lessons_to_git() {
    maybe_compact_lessons

    local lessons_dir="${LESSONS_DIR:-}"
    [ -z "${lessons_dir}" ] && return 0
    [ ! -d "${lessons_dir}" ] && return 0

    # Check if any lessons exist to push
    local has_lessons=false
    for f in "${lessons_dir}"/*.jsonl; do
        [ -f "$f" ] && has_lessons=true && break
    done
    [ "${has_lessons}" = false ] && return 0

    # Need GIT_TOKEN and GIT_REPO to push
    if [[ -z "${GIT_TOKEN:-}" || -z "${GIT_REPO:-}" ]]; then
        log_warn "push_lessons: GIT_TOKEN or GIT_REPO not set, skipping"
        return 0
    fi

    local branch="${GIT_BRANCH:-main}"
    local auth_url="${GIT_REPO/https:\/\//https://x-access-token:${GIT_TOKEN}@}"
    local tmp_clone="${RUN_OUTPUT_DIR}/.lessons_push_tmp"

    # Clone fresh (shallow, only the branch we need)
    rm -rf "${tmp_clone}"
    log_info "push_lessons: cloning repo for lessons push..."
    if ! git clone --depth 1 --branch "${branch}" "${auth_url}" "${tmp_clone}" 2>/dev/null; then
        log_warn "push_lessons: git clone failed"
        return 0
    fi

    # Copy lessons into the clone
    mkdir -p "${tmp_clone}/auto_quant/lessons"
    cp -f "${lessons_dir}"/*.jsonl "${tmp_clone}/auto_quant/lessons/" 2>/dev/null || true

    # Commit and push
    cd "${tmp_clone}"
    git config user.name "${GIT_USER_NAME:-auto-pipeline}"
    git config user.email "${GIT_USER_EMAIL:-auto@pipeline.local}"
    git add --force auto_quant/lessons/ 2>/dev/null || true

    if ! git diff --cached --quiet auto_quant/lessons/ 2>/dev/null; then
        git commit -m "lessons: update from ${MODEL_ID:-unknown} ${SCHEME:-} ${METHOD:-}" || true
        git push origin "${branch}" 2>/dev/null && log_ok "push_lessons: pushed successfully" || log_warn "push_lessons: git push failed"
    else
        log_info "push_lessons: no changes to push"
    fi

    cd - > /dev/null
    rm -rf "${tmp_clone}"
}
