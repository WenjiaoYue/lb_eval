#!/usr/bin/env python3
"""extract_autoround_info.py — Mine + confirm useful auto-round information.

Aggregates the pipeline's accumulated knowledge and extracts everything relevant
to **auto-round** (bug fixes, issue analyses, and — via the recipe registry —
working quantization recipes), deduplicated and confirmed by recurrence.

Sources scanned:
  1. lessons/*.jsonl                      — fix-loop lessons (root cause, solution, patch)
  2. results/**/failure_diagnosis*.json   — post-mortem agent diagnoses
  3. recipes/recipes.jsonl                — working recipes (positive KB)

An entry is considered "auto-round related" when any component/attribution field
mentions auto_round / auto-round / autoround, or the error signature does.

Output:
  - autoround_issues.jsonl   (deduped, confirmed structured records)
  - autoround_report.md      (human-readable summary)

Usage:
  python3 extract_autoround_info.py --repo-dir /path/to/lb_eval
  python3 extract_autoround_info.py --repo-dir . --out-dir /tmp/ar --json
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

_AR_RE = re.compile(r"auto[\-_ ]?round", re.I)


def _is_autoround(*fields) -> bool:
    for f in fields:
        if f and _AR_RE.search(str(f)):
            return True
    return False


def _load_jsonl(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(obj)
        except json.JSONDecodeError:
            continue
    return out


def _norm_sig(text: str) -> str:
    """Normalize an error signature for dedup: drop digits/paths/hex."""
    t = str(text or "").lower()
    t = re.sub(r"0x[0-9a-f]+", "", t)
    t = re.sub(r"/\S+", "", t)          # paths
    t = re.sub(r"\d+", "", t)           # numbers
    t = re.sub(r"[^a-z ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:120]


def collect_lessons(repo: Path) -> list[dict]:
    records = []
    lessons_dir = repo / "auto_quant" / "lessons"
    for jf in sorted(lessons_dir.glob("*.jsonl")):
        for l in _load_jsonl(jf):
            if not _is_autoround(
                l.get("agent_component"), l.get("error_category"),
                l.get("agent_category"), l.get("error_signature"),
                l.get("agent_root_cause"), l.get("solution"),
            ):
                continue
            records.append({
                "source": "lesson",
                "source_file": str(jf.relative_to(repo)),
                "phase": l.get("phase"),
                "signature": l.get("error_signature", ""),
                "category": l.get("error_category") or l.get("agent_category"),
                "root_cause": l.get("agent_root_cause", ""),
                "solution": l.get("solution", ""),
                "fix_tier": l.get("fix_tier", ""),
                "status": l.get("status", ""),
                "has_patch": bool(l.get("patch_has_changes")),
                "patch_file": l.get("patch_file", ""),
                "model": l.get("model", ""),
                "scheme": l.get("scheme", ""),
                "method": l.get("method", ""),
                "timestamp": l.get("timestamp", ""),
            })
    return records


def collect_diagnoses(repo: Path) -> list[dict]:
    records = []
    results_dir = repo / "results"
    if not results_dir.exists():
        return records
    for jf in results_dir.rglob("failure_diagnosis*.json"):
        try:
            d = json.loads(jf.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(d, dict):
            continue
        fa = d.get("fault_attribution", {}) if isinstance(d.get("fault_attribution"), dict) else {}
        if not _is_autoround(
            d.get("affected_component"), d.get("category"), d.get("key_error"),
            fa.get("component"), fa.get("specific_module"), d.get("root_cause"),
        ):
            continue
        records.append({
            "source": "diagnosis",
            "source_file": str(jf.relative_to(repo)),
            "phase": d.get("phase"),
            "signature": d.get("key_error", ""),
            "category": d.get("category", ""),
            "root_cause": d.get("root_cause", ""),
            "solution": d.get("suggested_fix", "") or d.get("workaround", ""),
            "fix_tier": fa.get("fault_type", ""),
            "status": "fix_available" if d.get("fix_available") else "no_fix",
            "specific_module": fa.get("specific_module", ""),
            "specific_function": fa.get("specific_function", ""),
            "responsible_party": fa.get("responsible_party", ""),
            "versions": d.get("versions_involved", {}),
            "community_summary": d.get("community_summary", ""),
            "timestamp": d.get("timestamp", ""),
        })
    return records


def collect_recipes(repo: Path) -> list[dict]:
    recipes = _load_jsonl(repo / "auto_quant" / "recipes" / "recipes.jsonl")
    out = []
    for r in recipes:
        # auto-round recipes: export_format auto_round OR auto_round_ref pinned
        if not (_is_autoround(r.get("export_format")) or r.get("auto_round_ref")):
            continue
        out.append({
            "source": "recipe",
            "model": r.get("model", ""),
            "scheme": r.get("scheme", ""),
            "method": r.get("method", ""),
            "iters": r.get("iters", ""),
            "export_format": r.get("export_format", ""),
            "auto_round_ref": r.get("auto_round_ref", ""),
            "accuracy_mean": r.get("accuracy_mean"),
            "status": r.get("status", ""),
            "timestamp": r.get("timestamp", ""),
        })
    return out


def dedup_issues(records: list[dict]) -> list[dict]:
    """Group by normalized signature; confirm by recurrence (occurrences count)."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[_norm_sig(r.get("signature", "")) or "unknown"].append(r)

    merged = []
    for sig, items in groups.items():
        # Prefer an item that has a fix/solution as the representative.
        items.sort(key=lambda x: (bool(x.get("solution")), bool(x.get("has_patch")),
                                  x.get("timestamp", "")), reverse=True)
        rep = items[0]
        models = sorted({i.get("model", "") for i in items if i.get("model")})
        fixed = any(str(i.get("status", "")).lower() in ("fixed", "fix_available") for i in items)
        merged.append({
            "signature_norm": sig,
            "representative_signature": rep.get("signature", ""),
            "category": rep.get("category", ""),
            "root_cause": rep.get("root_cause", ""),
            "solution": rep.get("solution", ""),
            "fix_tier": rep.get("fix_tier", ""),
            "confirmed": fixed,
            "occurrences": len(items),
            "affected_models": models,
            "has_patch": any(i.get("has_patch") for i in items),
            "patch_files": [i["patch_file"] for i in items if i.get("patch_file")],
            "sources": sorted({i.get("source", "") for i in items}),
            "specific_module": rep.get("specific_module", ""),
            "specific_function": rep.get("specific_function", ""),
            "responsible_party": rep.get("responsible_party", ""),
            "community_summary": rep.get("community_summary", ""),
        })
    merged.sort(key=lambda x: (x["confirmed"], x["occurrences"]), reverse=True)
    return merged


def write_report(issues: list[dict], recipes: list[dict], out_md: Path) -> None:
    lines = []
    lines.append("# Auto-Round: Extracted & Confirmed Information\n")
    lines.append(f"> Generated: {datetime.now(timezone.utc).isoformat()}\n")
    lines.append(f"- Distinct auto-round issues: **{len(issues)}**")
    confirmed = sum(1 for i in issues if i["confirmed"])
    lines.append(f"- Confirmed (a fix/workaround exists): **{confirmed}**")
    lines.append(f"- Working auto-round recipes: **{len(recipes)}**\n")

    lines.append("## Issues (deduped, most-confirmed first)\n")
    if not issues:
        lines.append("_None found._\n")
    for i, iss in enumerate(issues, 1):
        badge = "✅ confirmed" if iss["confirmed"] else "⚠️ open"
        lines.append(f"### {i}. {badge} — {iss['representative_signature'][:100]}")
        lines.append(f"- Category: `{iss['category']}`  | occurrences: {iss['occurrences']}"
                     f"  | sources: {', '.join(iss['sources'])}")
        if iss["specific_module"]:
            loc = iss["specific_module"] + (f".{iss['specific_function']}" if iss["specific_function"] else "")
            lines.append(f"- Location: `{loc}`  | responsible: {iss.get('responsible_party', '')}")
        if iss["root_cause"]:
            lines.append(f"- Root cause: {iss['root_cause']}")
        if iss["solution"]:
            lines.append(f"- Fix/workaround: `{iss['solution']}`")
        if iss["has_patch"]:
            lines.append(f"- Patch available: {', '.join(iss['patch_files']) or 'yes'}")
        if iss["affected_models"]:
            lines.append(f"- Affected models ({len(iss['affected_models'])}): "
                         f"{', '.join(iss['affected_models'][:8])}"
                         + (" …" if len(iss['affected_models']) > 8 else ""))
        lines.append("")

    lines.append("## Working recipes (auto-round)\n")
    if not recipes:
        lines.append("_None recorded yet._\n")
    for r in recipes:
        lines.append(f"- `{r['model']}` → {r['scheme']}/{r['method']} "
                     f"iters={r.get('iters')} mean_acc={r.get('accuracy_mean')} "
                     f"(ref={r.get('auto_round_ref') or 'default'})")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Extract & confirm auto-round information.")
    p.add_argument("--repo-dir", default=".", help="lb_eval repo root")
    p.add_argument("--out-dir", default="", help="Output dir (default: <repo>/auto_quant/error_analysis)")
    p.add_argument("--json", action="store_true", help="Also print JSON summary to stdout")
    args = p.parse_args()

    repo = Path(args.repo_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (repo / "auto_quant" / "error_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    lessons = collect_lessons(repo)
    diagnoses = collect_diagnoses(repo)
    recipes = collect_recipes(repo)

    issues = dedup_issues(lessons + diagnoses)

    out_jsonl = out_dir / "autoround_issues.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for iss in issues:
            f.write(json.dumps(iss, ensure_ascii=False) + "\n")

    out_md = out_dir / "autoround_report.md"
    write_report(issues, recipes, out_md)

    print(f"[autoround] lessons scanned (AR-related): {len(lessons)}")
    print(f"[autoround] diagnoses scanned (AR-related): {len(diagnoses)}")
    print(f"[autoround] distinct issues: {len(issues)} | recipes: {len(recipes)}")
    print(f"[autoround] wrote {out_jsonl}")
    print(f"[autoround] wrote {out_md}")

    if args.json:
        print(json.dumps({
            "issues": issues,
            "recipes": recipes,
        }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
