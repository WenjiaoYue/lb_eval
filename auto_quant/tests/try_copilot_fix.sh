#!/bin/bash
# try_copilot_fix.sh — Local smoke test for the agent backend switch.
#
# Exercises the REAL dispatch path: config.env (AGENT_BACKEND + token)
#   → run_agent_fix → run_copilot_fix / run_openclaw_fix → the CLI.
# It plants a deliberately broken Python script, hands the error to the
# configured agent, and checks the agent actually fixed the file.
#
# Usage:
#   bash auto_quant/tests/try_copilot_fix.sh
#   AGENT_BACKEND=copilot bash auto_quant/tests/try_copilot_fix.sh   # override
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTO_QUANT_DIR="$(dirname "${SCRIPT_DIR}")"
PHASES_DIR="${AUTO_QUANT_DIR}/phases"

# --- Minimal logging shims (normally provided by auto.sh) ---
log_info()  { echo -e "[info]  $*"; }
log_warn()  { echo -e "[warn]  $*"; }
log_ok()    { echo -e "[ ok ]  $*"; }
log_error() { echo -e "[err ]  $*"; }
log_step()  { echo -e "\n═══════ $* ═══════\n"; }

# --- Load user config (AGENT_BACKEND, COPILOT_* / token, timeouts) ---
if [[ -f "${AUTO_QUANT_DIR}/config.env" ]]; then
    set -a; source "${AUTO_QUANT_DIR}/config.env"; set +a
fi

# --- Sandbox working dir so we never touch the real pipeline output ---
RUN_OUTPUT_DIR="$(mktemp -d /tmp/try_copilot_fix.XXXXXX)"
export RUN_OUTPUT_DIR
mkdir -p "${RUN_OUTPUT_DIR}/logs"
phase_name="smoke"
fix_log_dir="${RUN_OUTPUT_DIR}/logs"
AGENT_TIMEOUT="${AGENT_TIMEOUT:-180}"

# --- Pull in the library (defines run_agent_fix / run_copilot_fix / ...) ---
source "${PHASES_DIR}/agent_fix_loop.sh"

# --- Plant a broken script: NameError (resutl vs result) ---
BROKEN="${RUN_OUTPUT_DIR}/broken.py"
cat > "${BROKEN}" <<'PY'
def main():
    result = 6 * 7
    print("ANSWER:", resutl)   # typo: resutl -> result

if __name__ == "__main__":
    main()
PY

log_step "Backend = ${AGENT_BACKEND:-openclaw}"
log_info "Broken script: ${BROKEN}"
log_info "Sandbox: ${RUN_OUTPUT_DIR}"

# Capture the real error the agent must fix
ERR_LOG="${RUN_OUTPUT_DIR}/logs/error.log"
if python3 "${BROKEN}" >"${ERR_LOG}" 2>&1; then
    log_error "Script unexpectedly passed before fix — abort"; exit 1
fi
log_info "Reproduced failure:"; sed 's/^/    /' "${ERR_LOG}"

# Build a fix prompt and call the configured agent via the dispatcher
PROMPT="A Python script fails. Fix the bug IN PLACE by editing the file, then stop.

File: ${BROKEN}
Error:
$(cat "${ERR_LOG}")

Requirements:
- Edit ${BROKEN} directly to fix the root cause.
- Do NOT create new files. Keep the logic (compute 6*7 and print it).
- After editing, the script must run cleanly and print 'ANSWER: 42'."

AGENT_LOG="${RUN_OUTPUT_DIR}/logs/agent.log"
log_step "Calling agent to fix it..."
if [ "${MULTI_AGENT:-0}" = "1" ]; then
    log_info "MULTI_AGENT=1 → two-agent flow (diagnoser → fixer)"
    run_multiagent_fix "${PROMPT}" "${AGENT_LOG}" "try_copilot_${phase_name}" 1 || true
else
    run_agent_fix "${PROMPT}" "${AGENT_LOG}" "try_copilot_${phase_name}" || true
fi

# Verify
log_step "Verifying fix..."
if python3 "${BROKEN}" 2>&1 | tee "${RUN_OUTPUT_DIR}/logs/after.log" | grep -q "ANSWER: 42"; then
    log_ok "Agent fixed the script. PASS"
    echo "Fixed file:"; sed 's/^/    /' "${BROKEN}"
    exit 0
else
    log_error "Script still broken after agent run. FAIL"
    echo "Agent log tail:"; tail -20 "${AGENT_LOG}" 2>/dev/null | sed 's/^/    /'
    exit 1
fi
