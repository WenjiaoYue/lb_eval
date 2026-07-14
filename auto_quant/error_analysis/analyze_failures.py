#!/usr/bin/env python3
"""Analyze failed pipeline runs, classify errors, and submit reports.

This script:
1. Scans lb_eval/results for failed runs (quantize/evaluate logs with errors)
2. Uses openclaw agent (with error_analysis skill) to analyze each failure
3. Saves structured diagnosis JSON alongside the run
4. Optionally uploads diagnosis summary to GitHub (lb_eval repo)
5. Optionally submits a community discussion to the leaderboard Space

Usage:
    # Analyze all failed runs (dry run - no uploads)
    python3 analyze_failures.py --results-dir /path/to/lb_eval/results

    # Analyze and upload to GitHub
    python3 analyze_failures.py --results-dir /path/to/lb_eval/results --push-github

    # Analyze and submit to leaderboard community
    python3 analyze_failures.py --results-dir /path/to/lb_eval/results --submit-community

    # Analyze a single run directory
    python3 analyze_failures.py --run-dir /path/to/results/org/model/run_xxx

Environment:
    GITHUB_TOKEN        — for pushing diagnosis to lb_eval repo
    HF_TOKEN            — for submitting community discussions
    OPENCLAW_TIMEOUT    — agent timeout in seconds (default: 120)
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add parent for taxonomy import
sys.path.insert(0, str(Path(__file__).resolve().parent))
from taxonomy import TAXONOMY, classify_error

SCRIPT_DIR = Path(__file__).resolve().parent
LB_EVAL_DIR = SCRIPT_DIR.parent.parent  # lb_eval root
LEADERBOARD_SPACE = "Intel/low-bit-leaderboard"
NOTIFY_USER = "lvkaokao"


def _extract_json_object(text: str) -> dict | None:
    """Extract the largest valid JSON object from text.

    Handles:
    - JSON wrapped in ```json ... ``` code blocks
    - Bare JSON objects in text
    - Nested JSON with arbitrary depth
    - Truncated JSON (attempts repair by closing open strings/braces)
    """
    import re as _re

    # Strategy 1: Find JSON inside ```json ... ``` blocks
    for m in _re.finditer(r"```(?:json)?\s*\n?(\{.+\})\s*\n?```", text, _re.DOTALL):
        try:
            parsed = json.loads(m.group(1))
            if isinstance(parsed, dict) and len(parsed) >= 5:
                return parsed
        except json.JSONDecodeError:
            pass

    # Strategy 2: Bracket-counting extraction
    best = None
    i = 0
    while i < len(text):
        if text[i] == '{':
            depth = 0
            in_string = False
            escape_next = False
            j = i
            while j < len(text):
                ch = text[j]
                if escape_next:
                    escape_next = False
                    j += 1
                    continue
                if in_string:
                    if ch == '\\':
                        escape_next = True
                    elif ch == '"':
                        in_string = False
                    j += 1
                    continue
                if ch == '"':
                    in_string = True
                elif ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = text[i:j + 1]
                        try:
                            parsed = json.loads(candidate)
                            if isinstance(parsed, dict):
                                if best is None or len(parsed) > len(best):
                                    best = parsed
                        except json.JSONDecodeError:
                            pass
                        break
                j += 1
        i += 1

    # Check if we found the actual diagnosis (must have "category" key)
    if best and "category" in best and len(best) >= 5:
        return best

    # Strategy 3: Handle truncated JSON (agent hit output limit)
    # Find the largest { that looks like the start of a diagnosis JSON
    for m in _re.finditer(r'```(?:json)?\s*\n?(\{)', text):
        start = m.start(1)
        fragment = text[start:]
        repaired = _repair_truncated_json(fragment)
        if repaired and len(repaired) > len(best or {}):
            best = repaired

    # Also try without code fence
    if not best or len(best) < 5:
        for m in _re.finditer(r'\{\s*"category"', text):
            fragment = text[m.start():]
            repaired = _repair_truncated_json(fragment)
            if repaired and len(repaired) > len(best or {}):
                best = repaired

    return best


def _repair_truncated_json(fragment: str) -> dict | None:
    """Attempt to repair a truncated JSON string by closing open structures."""
    # First try as-is
    try:
        parsed = json.loads(fragment)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Try progressively closing: strip trailing incomplete value, close string/braces
    # Remove trailing incomplete string value
    import re as _re
    cleaned = fragment.rstrip()

    # If ends mid-string, close the string
    # Find if we're inside a string by counting unescaped quotes
    in_string = False
    escape = False
    depth = 0
    for ch in cleaned:
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string:
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1

    repair = cleaned
    if in_string:
        repair += '"'
    # Remove trailing comma or colon with incomplete value
    repair = _re.sub(r',\s*$', '', repair)
    repair = _re.sub(r':\s*$', ': null', repair)
    # Close remaining braces
    for _ in range(depth):
        repair += '}'

    try:
        parsed = json.loads(repair)
        if isinstance(parsed, dict) and len(parsed) >= 3:
            return parsed
    except json.JSONDecodeError:
        pass

    return None


# ─── Utility Functions ───────────────────────────────────────────────

def find_failed_runs(results_dir: Path) -> list[dict]:
    """Find all runs with failed quantize or evaluate phases."""
    failed = []

    for run_dir in sorted(results_dir.rglob("run_*")):
        if not run_dir.is_dir():
            continue
        logs_dir = run_dir / "logs"
        if not logs_dir.exists():
            continue

        # Skip if already diagnosed
        if list(run_dir.glob("failure_diagnosis_*.json")):
            continue

        for log_name in ("quantize.log", "evaluate.log"):
            log_path = logs_dir / log_name
            if not log_path.exists():
                continue
            # Check if log contains errors
            try:
                tail = log_path.read_text(errors="replace")[-5000:]
                if re.search(r"error|exception|traceback|failed", tail, re.IGNORECASE):
                    # Verify it actually failed (not just warnings)
                    last_lines = tail.strip().split("\n")[-10:]
                    last_text = "\n".join(last_lines)
                    if re.search(r"Error|Exception|FAILED|Traceback|Killed", last_text):
                        phase = log_name.replace(".log", "")
                        # Extract model info from path
                        parts = run_dir.parts
                        # results/org/model-name/run_xxx
                        results_idx = next(
                            (i for i, p in enumerate(parts) if p == "results"), -1
                        )
                        if results_idx >= 0 and results_idx + 2 < len(parts):
                            org = parts[results_idx + 1]
                            model = parts[results_idx + 2]
                        else:
                            org = "unknown"
                            model = run_dir.parent.name

                        failed.append({
                            "run_dir": run_dir,
                            "log_path": log_path,
                            "phase": phase,
                            "org": org,
                            "model": model,
                            "run_id": run_dir.name,
                        })
            except OSError:
                continue

    return failed


def extract_error_context(log_path: Path, max_chars: int = 8000) -> str:
    """Extract relevant error context from a log file."""
    try:
        content = log_path.read_text(errors="replace")
    except OSError:
        return "[ERROR] Could not read log file"

    # Find the last traceback
    lines = content.split("\n")
    traceback_start = -1
    for i in range(len(lines) - 1, -1, -1):
        if "Traceback" in lines[i]:
            traceback_start = i
            break

    if traceback_start >= 0:
        # Include context before traceback + the full traceback
        start = max(0, traceback_start - 20)
        error_section = "\n".join(lines[start:])
        if len(error_section) > max_chars:
            error_section = error_section[-max_chars:]
        return error_section

    # No traceback found — return last N chars
    return content[-max_chars:]


def quick_classify(error_text: str) -> dict:
    """Quick classification without openclaw (pattern matching only)."""
    category, info = classify_error(error_text)
    return {
        "category": category,
        "description": info["description"],
        "retryable": info["retryable"],
        "fix_strategy": info["fix_strategy"],
        "workaround_hints": info["workaround_hints"],
    }


# ─── OpenClaw Agent Analysis ────────────────────────────────────────

def build_analysis_prompt(run_info: dict, error_context: str, quick_result: dict) -> str:
    """Build the prompt for openclaw agent to analyze the error."""
    return f"""You are a senior engineer analyzing a failed auto-quantization pipeline run.

⚠️ CRITICAL TIME CONSTRAINT: You have ~90 seconds total. Do NOT spend time on exhaustive investigation.
- Read the error log ONCE carefully
- Make at most 2-3 tool calls if needed (check a specific file/version)
- Then OUTPUT THE JSON IMMEDIATELY

If you cannot determine something, put your best guess with lower confidence. An 80% answer delivered on time is infinitely better than a perfect answer that times out.

## Run Information
- Model: {run_info['org']}/{run_info['model']}
- Phase: {run_info['phase']}
- Run ID: {run_info['run_id']}

## Quick Classification (pattern-based, may be wrong)
- Category: {quick_result['category']}
- Description: {quick_result['description']}

## Error Log (last section)
```
{error_context}
```

## Your Task — Root Cause Analysis

Analyze the traceback and determine:
1. **What** — The exact error and where it occurs
2. **Why** — Root cause (which component's fault: auto-round? transformers? model? environment?)
3. **Fix** — Concrete fix steps or explanation why not fixable

### Attribution Guidelines
Be specific about WHO is at fault:
- **auto-round code bug**: cite the auto_round file/function
- **transformers incompatibility**: which API changed? what version?
- **Model author fault**: corrupt tokenizer? invalid custom code? unsupported architecture?
- **Environment/infrastructure**: missing package? OOM? network?
- **PyTorch/kernel issue**: which op fails?

### Constraints
- Do NOT suggest downgrading PyTorch
- Do NOT modify evaluation tasks
- Prefer minimal, targeted fixes

## OUTPUT FORMAT

Output EXACTLY ONE JSON object (no other text before/after):
```json
{{
  "category": "<taxonomy_category or new_category_name>",
  "phase": "{run_info['phase']}",
  "key_error": "<exact final error line from log>",
  "root_cause": "<1-3 sentence explanation of the actual root cause>",
  "traceback_analysis": "<explain the call chain: A calls B calls C, fault is at B because...>",
  "fault_attribution": {{
    "component": "auto_round|transformers|tokenizers|torch|pytorch_kernel|model_code|model_data|lm_eval|infrastructure|unknown",
    "specific_module": "<e.g., auto_round.calib_dataset or transformers.tokenization_utils_tokenizers>",
    "specific_function": "<e.g., collate_batch or TokenizerFast.from_file>",
    "fault_type": "code_bug|api_change|corrupt_data|missing_dep|resource_limit|unsupported_arch|network",
    "responsible_party": "auto_round_devs|transformers_devs|model_author|infra_team|pytorch_devs"
  }},
  "retryable": true|false,
  "fix_available": true|false,
  "suggested_fix": "<concrete fix: commands or code change>",
  "fix_verification": "<command to verify fix works>",
  "workaround": "<alternative approach if primary fix is risky>",
  "affected_component": "auto_round|transformers|tokenizers|torch|lm_eval|model|infrastructure",
  "severity": "critical|high|medium|low",
  "confidence": 0.0-1.0,
  "versions_involved": {{
    "auto_round": "<version or unknown>",
    "transformers": "<version or unknown>",
    "torch": "<version or unknown>"
  }},
  "community_summary": "<2-3 sentence summary: what failed, why, what to do>"
}}
```

REMEMBER: Output the JSON NOW. Do not do more research. Use what you already know from the error log above.
"""


def run_openclaw_analysis(prompt: str, session_id: str, timeout: int = 120) -> tuple[dict | None, Path | None]:
    """Run openclaw agent to analyze the error.
    
    Returns (parsed_diagnosis, session_file_path) or (None, None).
    The session_file_path points to the raw JSONL for archival.
    """
    # Resolve openclaw binary
    nvm_dir = os.environ.get("NVM_DIR", os.path.expanduser("~/.nvm"))
    env = os.environ.copy()
    node_bin = Path(nvm_dir) / "versions" / "node"
    if node_bin.exists():
        v22_dirs = list(node_bin.glob("v22.*"))
        if v22_dirs:
            def _ver_key(p):
                parts = p.name.lstrip("v").split(".")
                return tuple(int(x) for x in parts if x.isdigit())
            v22_dirs.sort(key=_ver_key, reverse=True)
            env["PATH"] = f"{v22_dirs[0]}/bin:{env.get('PATH', '')}"

    openclaw_bin = shutil.which("openclaw", path=env.get("PATH"))
    if not openclaw_bin:
        print("  [WARN] openclaw not found, using quick classification only")
        return None, None

    # Configure git credential to avoid interactive prompts in containers
    git_token = os.environ.get("GIT_TOKEN") or os.environ.get("GITHUB_TOKEN", "")
    if git_token:
        # Set credential helper so agent's git operations don't hang
        env["GIT_ASKPASS"] = "/bin/echo"
        env["GIT_TERMINAL_PROMPT"] = "0"
        # Store credential for github.com
        subprocess.run(
            ["git", "config", "--global", "credential.helper", "store"],
            capture_output=True, env=env,
        )
        cred_file = Path.home() / ".git-credentials"
        cred_line = f"https://{git_token}@github.com\n"
        if not cred_file.exists() or cred_line not in cred_file.read_text():
            with open(cred_file, "a") as f:
                f.write(cred_line)
    else:
        # No token — disable interactive git prompts to avoid hanging
        env["GIT_TERMINAL_PROMPT"] = "0"

    # Save agent output to a log file for traceability
    agent_log_dir = Path(os.environ.get("RUN_OUTPUT_DIR", "/tmp")) / "logs"
    agent_log_dir.mkdir(parents=True, exist_ok=True)
    agent_log_path = agent_log_dir / f"error_analysis_agent_{session_id}.log"

    try:
        with open(agent_log_path, "w") as log_fh:
            proc = subprocess.Popen(
                [
                    openclaw_bin, "agent", "--local",
                    "--session-id", session_id,
                    "--message", prompt,
                    "--timeout", str(timeout),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            # Stream output to both log file and stdout (like tee)
            for line in proc.stdout:
                sys.stdout.write(f"    │ {line}")
                sys.stdout.flush()
                log_fh.write(line)
            proc.wait(timeout=timeout + 30)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        if 'proc' in locals():
            proc.kill()
        return None, None

    # Extract response from trajectory
    sessions_dir = Path.home() / ".openclaw" / "agents" / "main" / "sessions"
    session_file = sessions_dir / f"{session_id}.jsonl"
    if not session_file.exists():
        return None, None

    # Extract assistant text and thinking responses
    texts = []
    thinking_texts = []
    try:
        for line in session_file.read_text().strip().split("\n"):
            entry = json.loads(line)
            if entry.get("type") != "message":
                continue
            # message field may be a dict or a string representation of a dict
            msg = entry.get("message", {})
            if isinstance(msg, str):
                try:
                    msg = json.loads(msg)
                except json.JSONDecodeError:
                    import ast
                    try:
                        msg = ast.literal_eval(msg)
                    except (ValueError, SyntaxError):
                        continue
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text = part.get("text", "").strip()
                        if text:
                            texts.append(text)
                    elif part.get("type") == "thinking":
                        thinking = part.get("thinking", "").strip()
                        if thinking:
                            thinking_texts.append(thinking)
    except (json.JSONDecodeError, OSError):
        pass

    # Try to parse JSON from text responses — find the diagnosis JSON
    full_text = "\n".join(texts)
    diagnosis = _extract_json_object(full_text)
    if diagnosis:
        return diagnosis, session_file

    # Also check thinking blocks for JSON (agent may output there)
    if thinking_texts:
        full_thinking = "\n".join(thinking_texts)
        diagnosis = _extract_json_object(full_thinking)
        if diagnosis:
            return diagnosis, session_file

    # Fallback: try trajectory assistantTexts
    traj_file = sessions_dir / f"{session_id}.trajectory.jsonl"
    if traj_file.exists():
        try:
            for line in traj_file.read_text().strip().split("\n"):
                event = json.loads(line)
                if event.get("type") == "model.completed":
                    for text in event.get("data", {}).get("assistantTexts", []):
                        result = _extract_json_object(text)
                        if result:
                            return result, session_file
        except (json.JSONDecodeError, OSError):
            pass

    # Agent timed out but has thinking — try a fast follow-up to summarize
    if thinking_texts:
        partial_analysis = "\n\n".join(thinking_texts)
        return {"_agent_thinking": partial_analysis}, session_file

    # Agent ran but couldn't extract anything — still return session file
    return None, session_file


def run_copilot_analysis(prompt: str, session_id: str, timeout: int = 120) -> tuple[dict | None, Path | None]:
    """Run the GitHub Copilot CLI to analyze the error (drop-in for run_openclaw_analysis).

    Unlike OpenClaw (which persists a session JSONL we parse), the Copilot CLI prints
    the assistant's final answer to stdout. We capture stdout to a log file and extract
    the diagnosis JSON object from it.

    Auth: reads COPILOT_GITHUB_TOKEN / GH_TOKEN / GITHUB_TOKEN from the environment,
    or a prior interactive `copilot login`.

    Returns (parsed_diagnosis, log_file_path) or (None, log_file_path/None).
    """
    copilot_bin = shutil.which(os.environ.get("COPILOT_BIN", "copilot"))
    if not copilot_bin:
        print("  [WARN] copilot CLI not found, using quick classification only")
        return None, None

    env = os.environ.copy()
    # Ensure the token (if only present as a config var) is visible to the child.
    for _tok in ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
        if env.get(_tok):
            env[_tok] = env[_tok]
            break

    agent_log_dir = Path(os.environ.get("RUN_OUTPUT_DIR", "/tmp")) / "logs"
    agent_log_dir.mkdir(parents=True, exist_ok=True)
    agent_log_path = agent_log_dir / f"error_analysis_agent_{session_id}.log"

    tool_args = os.environ.get("COPILOT_TOOL_ARGS", "--allow-all-tools").split()
    model_args = []
    if os.environ.get("COPILOT_MODEL"):
        model_args = ["--model", os.environ["COPILOT_MODEL"]]

    cmd = [copilot_bin, *tool_args, *model_args, "-p", prompt]
    env["NO_COLOR"] = "1"

    try:
        with open(agent_log_path, "w") as log_fh:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            for line in proc.stdout:
                sys.stdout.write(f"    │ {line}")
                sys.stdout.flush()
                log_fh.write(line)
            proc.wait(timeout=timeout + 30)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        if 'proc' in locals():
            proc.kill()
        return None, agent_log_path if agent_log_path.exists() else None

    # Extract the diagnosis JSON from the captured stdout.
    try:
        output_text = agent_log_path.read_text()
    except OSError:
        return None, agent_log_path

    diagnosis = _extract_json_object(output_text)
    if diagnosis:
        return diagnosis, agent_log_path

    return None, agent_log_path


def run_agent_analysis(prompt: str, session_id: str, timeout: int = 120,
                       backend: str | None = None) -> tuple[dict | None, Path | None]:
    """Dispatch failure analysis to the configured agent backend.

    Selected via AGENT_BACKEND (env / config.env): openclaw (default) | copilot.
    A `backend` override lets callers force a specific backend (used by the
    hybrid escalation path).
    """
    backend = (backend or os.environ.get("AGENT_BACKEND", "openclaw")).strip().lower()
    if backend == "copilot":
        return run_copilot_analysis(prompt, session_id, timeout=timeout)
    if backend not in ("openclaw", ""):
        print(f"  [WARN] Unknown AGENT_BACKEND='{backend}', falling back to openclaw")
    return run_openclaw_analysis(prompt, session_id, timeout=timeout)


def _summarize_agent_thinking(
    run_info: dict, error_context: str, thinking: str, timeout: int = 60
) -> dict | None:
    """Quick second call to openclaw: summarize partial thinking into structured JSON.
    
    No tool calls needed — just format existing analysis.
    """
    # Truncate thinking if too long (keep most recent analysis)
    if len(thinking) > 8000:
        thinking = thinking[-8000:]

    summary_prompt = f"""You previously analyzed a pipeline failure but ran out of time before producing output.
Below is your internal analysis (thinking). Produce the final structured JSON diagnosis based on it.

## Context
- Model: {run_info['org']}/{run_info['model']}
- Phase: {run_info['phase']}
- Key error: {error_context.strip().split(chr(10))[-1][:200]}

## Your Previous Analysis (thinking blocks)
{thinking}

## Task
Based on your analysis above, output EXACTLY ONE JSON object. Do NOT use any tools. Do NOT do further research. Just synthesize what you already figured out:

```json
{{
  "category": "<best category based on your analysis>",
  "phase": "{run_info['phase']}",
  "key_error": "<exact error from the log>",
  "root_cause": "<1-3 sentences: what you determined is the root cause>",
  "traceback_analysis": "<your call chain analysis from the thinking above>",
  "fault_attribution": {{
    "component": "auto_round|transformers|tokenizers|torch|model_code|model_data|lm_eval|infrastructure",
    "specific_module": "<module you identified>",
    "specific_function": "<function you identified>",
    "fault_type": "code_bug|api_change|corrupt_data|missing_dep|resource_limit|unsupported_arch",
    "responsible_party": "auto_round_devs|transformers_devs|model_author|infra_team|pytorch_devs"
  }},
  "retryable": true|false,
  "fix_available": true|false,
  "suggested_fix": "<fix based on your analysis>",
  "fix_verification": "<verification command>",
  "workaround": "<workaround>",
  "affected_component": "<component>",
  "severity": "critical|high|medium|low",
  "confidence": 0.0-1.0,
  "versions_involved": {{"auto_round": "<>", "transformers": "<>", "torch": "<>"}},
  "community_summary": "<2-3 sentence summary>"
}}
```

OUTPUT THE JSON NOW. No tools. No research. Just format your existing analysis.
"""

    session_id = f"summary_{run_info['org']}_{int(time.time())}"
    # Use the configured agent backend but with a shorter timeout
    result, _ = run_agent_analysis(summary_prompt, session_id, timeout=timeout)
    # If result has _agent_thinking, it means it timed out again — give up
    if result and "_agent_thinking" not in result:
        return result
    return None


# ─── Report Generation & Submission ─────────────────────────────────

def save_diagnosis(run_dir: Path, phase: str, diagnosis: dict) -> Path:
    """Save diagnosis JSON to the run directory."""
    out_path = run_dir / f"failure_diagnosis_{phase}.json"
    # Don't save raw_error_log in JSON (too large, redundant with log file)
    save_data = {k: v for k, v in diagnosis.items() if k != "raw_error_log"}
    out_path.write_text(json.dumps(save_data, indent=2, ensure_ascii=False))
    return out_path


def save_failure_analysis_md(run_dir: Path, phase: str, diagnosis: dict, run_info: dict) -> Path:
    """Save human-readable failure analysis as Markdown (pushed to GitHub with results)."""
    model_name = f"{run_info['org']}/{run_info['model']}"
    quant_scheme = diagnosis.get("quant_scheme", _get_quant_scheme(run_info))
    category = diagnosis.get("category", "unknown")

    # Show the deterministic taxonomy label too when it disagrees with the agent's — a
    # disagreement is itself a useful signal (agent refined it, or agent may be off).
    pattern_category = diagnosis.get("pattern_category", "")
    pattern_line = ""
    if pattern_category and pattern_category != category:
        pattern_line = f"\n**Pattern Match:** `{pattern_category}` (deterministic)"

    fault_attr = diagnosis.get("fault_attribution", {})
    traceback_analysis = diagnosis.get("traceback_analysis", "")

    # Build attribution table
    attribution_section = ""
    if fault_attr and fault_attr.get("component") != "unknown":
        attribution_section = f"""
## Fault Attribution

| Field | Value |
|-------|-------|
| Component | `{fault_attr.get('component', 'unknown')}` |
| Module | `{fault_attr.get('specific_module', 'N/A')}` |
| Function | `{fault_attr.get('specific_function', 'N/A')}` |
| Fault Type | `{fault_attr.get('fault_type', 'unknown')}` |
| Responsible | `{fault_attr.get('responsible_party', 'unknown')}` |
"""

    # Build traceback analysis section
    traceback_section = ""
    if traceback_analysis:
        traceback_section = f"""
## Traceback Analysis

{traceback_analysis}
"""

    # Build fix section
    fix_section = ""
    suggested_fix = diagnosis.get("suggested_fix", "")
    if suggested_fix:
        fix_section = f"""
## Suggested Fix

{suggested_fix}
"""
        fix_verification = diagnosis.get("fix_verification", "")
        if fix_verification:
            fix_section += f"""
### Verification Command

```bash
{fix_verification}
```
"""

    workaround = diagnosis.get("workaround", "")
    if workaround and workaround != "None":
        fix_section += f"""
## Workaround

{workaround}
"""

    md_content = f"""# Failure Analysis: {model_name}

**Quantization Scheme:** `{quant_scheme}`
**Failed Phase:** `{phase}`
**Error Category:** `{category}`{pattern_line}
**Severity:** `{diagnosis.get('severity', 'unknown')}`
**Confidence:** `{diagnosis.get('confidence', 'N/A')}`
**Retryable:** `{diagnosis.get('retryable', 'unknown')}`
**Analyzed:** `{diagnosis.get('analyzed_at', 'N/A')}`

## Root Cause

{diagnosis.get('root_cause', 'Not determined')}
{attribution_section}
{traceback_section}

## Key Error

```
{diagnosis.get('key_error', 'Unknown')}
```

## Versions

| Package | Version |
|---------|---------|
| auto-round | `{diagnosis.get('versions_involved', {}).get('auto_round', 'unknown')}` |
| transformers | `{diagnosis.get('versions_involved', {}).get('transformers', 'unknown')}` |
| torch | `{diagnosis.get('versions_involved', {}).get('torch', 'unknown')}` |
{fix_section}
---
*Auto-generated by error_analysis pipeline.*
"""

    out_path = run_dir / "failure_analysis.md"
    out_path.write_text(md_content)
    return out_path


def push_to_github(
    diagnosis_files: list[Path],
    repo_dir: Path | None = None,
    org: str | None = None,
    artifact_name: str | None = None,
    run_id: str | None = None,
) -> bool:
    """Copy diagnosis files to the lb_eval git repo and push.

    The repo structure is:  repo_dir/results/{org}/{artifact_name}/run_{timestamp}/
    We find the matching run directory in the repo and copy files there.
    """
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GIT_TOKEN")
    if not token:
        print("  [SKIP] GITHUB_TOKEN/GIT_TOKEN not set, skipping git push")
        return False

    # Determine git repo root
    git_root = repo_dir
    if not git_root:
        # Fallback: walk up from the first diagnosis file
        if not diagnosis_files:
            return False
        git_root = diagnosis_files[0].parent
        while git_root != git_root.parent:
            if (git_root / ".git").exists():
                break
            git_root = git_root.parent
        else:
            print("  [WARN] Not a git repository")
            return False
    
    if not (git_root / ".git").exists():
        # repo_dir might not be the root — walk up
        _root = git_root
        while _root != _root.parent:
            if (_root / ".git").exists():
                git_root = _root
                break
            _root = _root.parent
        else:
            print("  [WARN] Not a git repository: {git_root}")
            return False

    try:
        # Find the target run directory inside the repo
        target_run_dir = None
        if org and artifact_name and run_id and run_id.startswith("run_"):
            # Exact path: results/org/artifact_name/run_id
            target_run_dir = git_root / "results" / org / artifact_name / run_id
            if not target_run_dir.exists():
                target_run_dir = None  # Doesn't exist, fall through

        if not target_run_dir and org and artifact_name:
            # Find the latest run_* dir (most recently created by upload_results_github.py)
            model_dir = git_root / "results" / org / artifact_name
            if model_dir.exists():
                run_dirs = sorted(
                    [d for d in model_dir.glob("run_*") if d.is_dir()],
                    reverse=True,
                )
                if run_dirs:
                    target_run_dir = run_dirs[0]
        
        if not target_run_dir:
            # Last resort: if files are already inside the repo, use their location
            for f in diagnosis_files:
                try:
                    f.relative_to(git_root)
                    target_run_dir = f.parent
                    break
                except ValueError:
                    continue

        if not target_run_dir:
            print("  [WARN] Cannot determine target run directory in repo")
            return False
        
        target_run_dir.mkdir(parents=True, exist_ok=True)

        # Prevent interactive git prompts (critical in containers)
        push_env = os.environ.copy()
        push_env["GIT_TERMINAL_PROMPT"] = "0"
        push_env["GIT_ASKPASS"] = "/bin/echo"

        # Ensure git config is set (required in containers / CI)
        git_user_name = os.environ.get("GIT_USER_NAME", "auto-pipeline")
        git_user_email = os.environ.get("GIT_USER_EMAIL", "auto@pipeline.local")
        subprocess.run(
            ["git", "config", "user.name", git_user_name],
            cwd=git_root, capture_output=True, env=push_env,
        )
        subprocess.run(
            ["git", "config", "user.email", git_user_email],
            cwd=git_root, capture_output=True, env=push_env,
        )

        # Copy diagnosis files to target run dir (if not already there)
        copied_paths = []
        for f in diagnosis_files:
            dest = target_run_dir / f.name
            try:
                f.relative_to(git_root)
                # File is already in repo — just git add it
                copied_paths.append(f)
            except ValueError:
                # File is outside repo — copy it in
                shutil.copy2(f, dest)
                copied_paths.append(dest)

        # Git add all
        for f in copied_paths:
            try:
                rel = f.relative_to(git_root)
            except ValueError:
                continue
            subprocess.run(
                ["git", "add", str(rel)],
                cwd=git_root, check=True, capture_output=True, env=push_env,
            )

        # Check if there are staged changes
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=git_root, capture_output=True, env=push_env,
        )
        if result.returncode == 0:
            print("  [SKIP] No changes to commit")
            return True

        # Commit
        msg = f"Add failure analysis for {org}/{artifact_name}\n\nAutomated error analysis by pipeline."
        subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=git_root, check=True, capture_output=True, env=push_env,
        )

        # Build authenticated push URL (like upload_results_github.py does)
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=git_root, capture_output=True, text=True, env=push_env,
        )
        remote_url = result.stdout.strip() if result.returncode == 0 else ""
        
        # Construct auth URL: https://x-access-token:TOKEN@github.com/...
        if remote_url and token:
            import urllib.parse
            if remote_url.startswith("https://"):
                # Remove any existing credentials from URL
                parsed = urllib.parse.urlparse(remote_url)
                auth_url = f"https://x-access-token:{token}@{parsed.hostname}{parsed.path}"
            else:
                auth_url = remote_url
        else:
            auth_url = "origin"

        # Pull rebase to handle remote changes, then push
        subprocess.run(
            ["git", "pull", "--rebase", "--autostash", auth_url, "main"],
            cwd=git_root, capture_output=True, timeout=60, env=push_env,
        )

        # Push with authenticated URL
        subprocess.run(
            ["git", "push", auth_url, "HEAD:main"],
            cwd=git_root, check=True, capture_output=True, env=push_env,
        )
        print(f"  [OK] Pushed {len(copied_paths)} file(s) to {target_run_dir.relative_to(git_root)}")
        return True

    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else str(e.stderr or "")
        print(f"  [WARN] Git push failed: {stderr[:200]}")
        return False


def _extract_full_error_log(log_path: Path) -> str:
    """Extract the full error section from log (untruncated, for community posting)."""
    try:
        content = log_path.read_text(errors="replace")
    except OSError:
        return "[ERROR] Could not read log file"

    lines = content.split("\n")

    # Find the last traceback — include everything from 30 lines before to end
    traceback_start = -1
    for i in range(len(lines) - 1, -1, -1):
        if "Traceback" in lines[i]:
            traceback_start = i
            break

    if traceback_start >= 0:
        start = max(0, traceback_start - 30)
        return "\n".join(lines[start:])

    # No traceback — return last 200 lines
    return "\n".join(lines[-200:])


def _get_quant_scheme(run_info: dict) -> str:
    """Extract quantization scheme from request.json or model name."""
    # Try request.json
    request_path = run_info["run_dir"] / "request.json"
    if request_path.exists():
        try:
            req = json.loads(request_path.read_text())
            scheme = req.get("quant_scheme", "")
            if scheme:
                return scheme
        except (json.JSONDecodeError, OSError):
            pass

    # Fall back to parsing model name (e.g., "Model-AutoRound-W4A16-RTN")
    model_name = run_info.get("model", "")
    import re as _re
    m = _re.search(r"(W\d+A\d+(?:-\w+)?)", model_name)
    if m:
        return m.group(1)
    return "Unknown"


def submit_community_discussion(
    run_info: dict, diagnosis: dict, hf_token: str | None = None
) -> bool:
    """Submit a community discussion to the leaderboard Space.

    Title format: model_name | quant_scheme
    Body: complete raw error log only (no agent analysis — that goes to failure_analysis.md)
    """
    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        # Try HF_TOKENS (comma-separated, take first)
        hf_tokens = os.environ.get("HF_TOKENS", "")
        if hf_tokens:
            token = hf_tokens.split(",")[0].strip()
    if not token:
        # Try reading from huggingface cache
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            token = token_path.read_text().strip()
    if not token:
        print("  [SKIP] HF_TOKEN not set, skipping community submission")
        return False

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("  [SKIP] huggingface_hub not installed")
        return False

    # Build title: model | scheme
    model_name = f"{run_info['org']}/{run_info['model']}"
    quant_scheme = _get_quant_scheme(run_info)
    title = f"{model_name} | {quant_scheme}"
    if len(title) > 100:
        max_model_len = 100 - len(quant_scheme) - 5
        title = f"{model_name[:max_model_len]}... | {quant_scheme}"

    # Full raw error log (NOT truncated)
    raw_error_log = diagnosis.get("raw_error_log") or _extract_full_error_log(run_info["log_path"])

    # Build body — only error log + basic metadata
    phase = run_info.get("phase", "unknown")
    category = diagnosis.get("category", "unknown")

    body = f"""## Pipeline Failure Report

**Model:** `{model_name}`
**Quantization Scheme:** `{quant_scheme}`
**Failed Phase:** `{phase}`
**Run ID:** `{run_info['run_id']}`
**Error Category:** `{category}`

---

### Full Error Log

```
{raw_error_log}
```

---
*Auto-generated by error_analysis pipeline. cc @{NOTIFY_USER}*
"""

    try:
        api = HfApi(token=token)
        discussion = api.create_discussion(
            repo_id=LEADERBOARD_SPACE,
            repo_type="space",
            title=title,
            description=body,
        )
        print(f"  [OK] Community discussion created: {discussion.url}")
        return True
    except Exception as e:
        print(f"  [WARN] Community submission failed: {e}")
        return False


# ─── Main Entry Point ────────────────────────────────────────────────

def analyze_single_run(run_info: dict, use_agent: bool = True, timeout: int = 120) -> dict | None:
    """Analyze a single failed run."""
    print(f"\n  Analyzing: {run_info['org']}/{run_info['model']} [{run_info['phase']}]")

    # Extract error context (truncated for agent prompt) and full log (for community)
    error_context = extract_error_context(run_info["log_path"])
    raw_error_log = _extract_full_error_log(run_info["log_path"])

    # Quick classification (pattern matching) — use full log for better accuracy
    quick_result = quick_classify(raw_error_log)
    print(f"    Quick: {quick_result['category']} (retryable={quick_result['retryable']})")

    # Deep analysis with the configured agent backend
    diagnosis = None
    session_file = None
    if use_agent:
        session_id = f"diag_{run_info['org']}_{int(time.time())}"
        prompt = build_analysis_prompt(run_info, error_context, quick_result)

        # Hybrid escalation: try the cheap PRIMARY backend first; if it fails to
        # produce a usable diagnosis, escalate to the stronger ESCALATED backend.
        escalate = os.environ.get("AGENT_ESCALATE", "0") == "1"
        primary = os.environ.get("AGENT_PRIMARY", os.environ.get("AGENT_BACKEND", "openclaw"))
        escalated = os.environ.get("AGENT_ESCALATED", "copilot")

        first_backend = primary if escalate else None
        print(f"    Agent analyzing ({first_backend or os.environ.get('AGENT_BACKEND', 'openclaw')})...",
              end=" ", flush=True)
        diagnosis, session_file = run_agent_analysis(prompt, session_id, timeout=timeout,
                                                     backend=first_backend)

        def _usable(d):
            return bool(d) and "_agent_thinking" not in d and (
                d.get("root_cause") or d.get("suggested_fix") or d.get("category"))

        if escalate and not _usable(diagnosis) and escalated != primary:
            print(f"→ [primary '{primary}' inconclusive, escalating to '{escalated}'...]",
                  end=" ", flush=True)
            esc_session = f"diag_{run_info['org']}_{int(time.time())}_esc"
            esc_diag, esc_file = run_agent_analysis(prompt, esc_session, timeout=timeout,
                                                    backend=escalated)
            if _usable(esc_diag):
                diagnosis, session_file = esc_diag, esc_file

        if diagnosis and "_agent_thinking" in diagnosis:
            # Agent timed out but has partial thinking — ask openclaw to summarize
            thinking_content = diagnosis.pop("_agent_thinking")
            print("→ [timed out, summarizing thinking...]", end=" ", flush=True)
            summary_diagnosis = _summarize_agent_thinking(
                run_info, error_context, thinking_content, timeout=60
            )
            if summary_diagnosis:
                diagnosis = summary_diagnosis
                print(f"→ {diagnosis.get('category', '?')} (confidence={diagnosis.get('confidence', '?')})")
            else:
                diagnosis = None
                print("→ [summary also failed]")
        elif diagnosis:
            print(f"→ {diagnosis.get('category', '?')} (confidence={diagnosis.get('confidence', '?')})")
        else:
            print("→ [agent failed, using quick classification]")

    # Fall back to quick classification if agent failed
    if not diagnosis:
        # Map category to fault attribution for quick mode
        _CATEGORY_ATTRIBUTION = {
            "autoround_internal_error": ("auto_round", "code_bug", "auto_round_devs"),
            "transformers_incompatible": ("transformers", "api_change", "transformers_devs"),
            "tokenizer_error": ("tokenizers", "corrupt_data", "model_author"),
            "pytorch_cuda_error": ("torch", "resource_limit", "infra_team"),
            "dtype_mismatch": ("pytorch_kernel", "code_bug", "auto_round_devs"),
            "out_of_memory": ("infrastructure", "resource_limit", "infra_team"),
            "multimodal_unsupported": ("model_code", "unsupported_arch", "model_author"),
            "missing_dependency": ("infrastructure", "missing_dep", "infra_team"),
            "dataset_error": ("lm_eval", "corrupt_data", "auto_round_devs"),
            "eval_framework_error": ("lm_eval", "api_change", "auto_round_devs"),
            "network_error": ("infrastructure", "network", "infra_team"),
            "model_unavailable": ("model_data", "corrupt_data", "model_author"),
            "process_killed": ("infrastructure", "resource_limit", "infra_team"),
            "shape_mismatch": ("auto_round", "code_bug", "auto_round_devs"),
            "meta_device_error": ("auto_round", "code_bug", "auto_round_devs"),
            "device_mismatch": ("infrastructure", "code_bug", "infra_team"),
        }
        attr = _CATEGORY_ATTRIBUTION.get(quick_result["category"], ("unknown", "unknown", "unknown"))

        diagnosis = {
            "category": quick_result["category"],
            "phase": run_info["phase"],
            "key_error": error_context.strip().split("\n")[-1][:200],
            "root_cause": quick_result["description"],
            "traceback_analysis": "",
            "fault_attribution": {
                "component": attr[0],
                "specific_module": "N/A (quick classification only)",
                "specific_function": "N/A (quick classification only)",
                "fault_type": attr[1],
                "responsible_party": attr[2],
            },
            "retryable": quick_result["retryable"],
            "fix_available": quick_result["retryable"] or False,
            "suggested_fix": "; ".join(quick_result["workaround_hints"][:2]),
            "fix_verification": "",
            "workaround": quick_result["workaround_hints"][0] if quick_result["workaround_hints"] else "None",
            "affected_component": attr[0],
            "severity": "medium",
            "confidence": 0.6,
            "versions_involved": {},
            "community_summary": f"{quick_result['description']} in {run_info['phase']} phase for model {run_info['org']}/{run_info['model']}.",
        }

    # Add metadata
    diagnosis["analyzed_at"] = datetime.now(timezone.utc).isoformat()
    diagnosis["model"] = f"{run_info['org']}/{run_info['model']}"
    diagnosis["run_id"] = run_info["run_id"]
    diagnosis["raw_error_log"] = raw_error_log
    diagnosis["quant_scheme"] = _get_quant_scheme(run_info)
    # Always preserve the deterministic taxonomy label alongside the agent's category so the
    # pushed report carries BOTH a reliable pattern-based label and the agent's semantic one
    # (they cross-check each other; the deterministic label survives even if the agent errs).
    diagnosis["pattern_category"] = quick_result["category"]

    # Copy diagnostic session JSONL to run directory (matches auto.sh convention)
    copied_session = None
    if session_file and session_file.exists():
        basename = session_file.name
        if not basename.startswith("session_"):
            basename = f"session_{basename}"
        dest = run_info["run_dir"] / basename
        try:
            shutil.copy2(session_file, dest)
            copied_session = dest
            # Also format to markdown if formatter exists
            formatter = LB_EVAL_DIR / "auto_quant" / "format_sessions.py"
            if formatter.exists():
                subprocess.run(
                    ["python3", str(formatter), str(dest)],
                    capture_output=True, timeout=30,
                )
        except (OSError, subprocess.TimeoutExpired):
            pass

    return diagnosis, copied_session


def main():
    parser = argparse.ArgumentParser(
        description="Analyze failed pipeline runs and submit reports"
    )
    parser.add_argument(
        "--results-dir", type=Path,
        default=LB_EVAL_DIR / "results",
        help="Path to lb_eval/results directory",
    )
    parser.add_argument(
        "--run-dir", type=Path, default=None,
        help="Analyze a single run directory",
    )
    parser.add_argument(
        "--repo-dir", type=Path, default=None,
        help="Path to lb_eval git repo (for push). If not set, uses results_dir parent.",
    )
    parser.add_argument(
        "--org", type=str, default=None,
        help="Organization/owner name (for locating run in repo)",
    )
    parser.add_argument(
        "--artifact-name", type=str, default=None,
        help="Artifact name (e.g. model-AutoRound-W4A16-Tuning) for locating run in repo",
    )
    parser.add_argument(
        "--push-github", action="store_true",
        help="Push diagnosis files to GitHub",
    )
    parser.add_argument(
        "--submit-community", action="store_true",
        help="Submit discussions to leaderboard community",
    )
    parser.add_argument(
        "--no-agent", action="store_true",
        help="Skip openclaw agent (quick classification only)",
    )
    parser.add_argument(
        "--timeout", type=int, default=120,
        help="OpenClaw agent timeout per analysis (seconds)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max number of runs to analyze (0=all)",
    )
    args = parser.parse_args()

    # Find failed runs
    if args.run_dir:
        # Analyze single run
        if not args.run_dir.exists():
            print(f"Error: {args.run_dir} not found")
            sys.exit(1)
        # Determine phase from available logs
        logs_dir = args.run_dir / "logs"
        runs_to_analyze = []

        # Try to get org/model from request.json (works for pipeline output dirs)
        org, model = "unknown", "unknown"
        request_json = args.run_dir / "request.json"
        if request_json.exists():
            try:
                req = json.loads(request_json.read_text())
                model_id = req.get("model", "")
                if "/" in model_id:
                    org, model = model_id.split("/", 1)
                else:
                    model = model_id
            except (json.JSONDecodeError, OSError):
                pass

        # Fallback: infer from path (results/org/model/run_xxx)
        if org == "unknown":
            parts = args.run_dir.parts
            if len(parts) >= 3:
                org = parts[-3]
                model = parts[-2]

        for log_name in ("quantize.log", "evaluate.log", "setup_env.log"):
            log_path = logs_dir / log_name
            if log_path.exists():
                # Only include if the log actually has errors
                try:
                    tail = log_path.read_text(errors="replace")[-2000:]
                    if re.search(r"Error|Exception|Traceback|Killed|FAILED", tail):
                        runs_to_analyze.append({
                            "run_dir": args.run_dir,
                            "log_path": log_path,
                            "phase": log_name.replace(".log", ""),
                            "org": org,
                            "model": model,
                            "run_id": args.run_dir.name,
                        })
                except OSError:
                    continue
    else:
        print(f"Scanning {args.results_dir} for failed runs...")
        runs_to_analyze = find_failed_runs(args.results_dir)

    if not runs_to_analyze:
        print("No failed runs found (or all already diagnosed).")
        return

    if args.limit > 0:
        runs_to_analyze = runs_to_analyze[:args.limit]

    print(f"Found {len(runs_to_analyze)} failed run(s) to analyze.")

    # Analyze each run
    diagnosis_files = []
    community_submissions = []

    for run_info in runs_to_analyze:
        result = analyze_single_run(
            run_info, use_agent=not args.no_agent, timeout=args.timeout
        )
        if not result:
            continue
        diagnosis, session_path = result

        # Save diagnosis JSON
        out_path = save_diagnosis(run_info["run_dir"], run_info["phase"], diagnosis)
        diagnosis_files.append(out_path)
        print(f"    Saved: {out_path.name}")

        # Save failure_analysis.md (human-readable, goes to GitHub with results)
        md_path = save_failure_analysis_md(
            run_info["run_dir"], run_info["phase"], diagnosis, run_info
        )
        diagnosis_files.append(md_path)
        print(f"    Saved: {md_path.name}")

        # Include diagnostic session file (agent's full reasoning trace)
        if session_path and session_path.exists():
            diagnosis_files.append(session_path)
            # Also include formatted .md if it was generated
            session_md = session_path.with_suffix(".md")
            if session_md.exists():
                diagnosis_files.append(session_md)
            print(f"    Saved: {session_path.name} (agent session)")

        # Collect for community submission
        if args.submit_community:
            community_submissions.append((run_info, diagnosis))

    # Push to GitHub
    if args.push_github and diagnosis_files:
        print(f"\nPushing {len(diagnosis_files)} diagnosis file(s) to GitHub...")
        # Determine run_id from first analyzed run
        _run_id = runs_to_analyze[0]["run_id"] if runs_to_analyze else None
        push_to_github(
            diagnosis_files,
            repo_dir=args.repo_dir,
            org=args.org,
            artifact_name=args.artifact_name,
            run_id=_run_id,
        )

    # Submit community discussions
    if community_submissions:
        print(f"\nSubmitting {len(community_submissions)} community discussion(s)...")
        for run_info, diagnosis in community_submissions:
            submit_community_discussion(run_info, diagnosis)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Analysis Summary")
    print(f"{'='*60}")
    print(f"  Total analyzed: {len(diagnosis_files)}")

    # Category breakdown
    categories = {}
    for f in diagnosis_files:
        try:
            d = json.loads(f.read_text())
            cat = d.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
        except (json.JSONDecodeError, OSError):
            pass

    if categories:
        print(f"\n  {'Category':<30} {'Count':>5}")
        print(f"  {'─'*30} {'─'*5}")
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            print(f"  {cat:<30} {count:>5}")


if __name__ == "__main__":
    main()
