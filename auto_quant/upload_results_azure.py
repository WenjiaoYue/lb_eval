#!/usr/bin/env python3
"""
Upload auto_quant result artifacts to Azure Blob Storage.

Replaces upload_results_github.py. Uses azure-storage-blob instead of git.

Blob layout:
  results/<org>/<artifact_name>/results_<timestamp>.json
  results/<org>/<artifact_name>/run_<timestamp>/quant_summary.json
  results/<org>/<artifact_name>/run_<timestamp>/accuracy.json
  results/<org>/<artifact_name>/run_<timestamp>/quantize.py
  results/<org>/<artifact_name>/run_<timestamp>/lm_eval_results/**
  results/<org>/<artifact_name>/run_<timestamp>/logs/**
  results/<org>/<artifact_name>/run_<timestamp>/session_*.jsonl
  results/<org>/<artifact_name>/run_<timestamp>/session_*.md
  status/<request_filename>

Install dependency:
  pip install azure-storage-blob
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def file_timestamp() -> str:
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())


def sanitize_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def looks_like_quantized_artifact(model_short: str) -> bool:
    lowered = model_short.lower()
    markers = (
        "-autoround-",
        "-gptq",
        "-awq",
        ".gguf",
        "-gguf",
        "llm-compressor",
    )
    return any(marker in lowered for marker in markers)


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def detect_artifact_name(model_id: str, scheme: str, quant_summary: dict | None) -> str:
    if quant_summary:
        hf_repo = quant_summary.get("hf_repo")
        if isinstance(hf_repo, str) and hf_repo.strip():
            return hf_repo.rstrip("/").rsplit("/", 1)[-1]

    model_short = model_id.split("/", 1)[-1] if "/" in model_id else model_id
    if looks_like_quantized_artifact(model_short):
        return sanitize_name(model_short)
    return sanitize_name(f"{model_short}-autoround-{scheme}")


def derive_pipeline_status(quant_summary: dict | None, accuracy: dict | None) -> str:
    """Derive overall pipeline status from quant_summary and accuracy results.

    Returns one of: "Finished", "Quant Failed", "Eval Failed", "Partial".
    """
    qs_status = (quant_summary or {}).get("status", "missing")
    acc_status = (accuracy or {}).get("status", "missing")

    if qs_status == "failed":
        return "Quant Failed"
    if acc_status == "failed":
        return "Eval Failed"

    # Check for any task with acc=0 (indicates evaluation failure)
    if isinstance(accuracy, dict):
        tasks = accuracy.get("tasks")
        if isinstance(tasks, dict):
            for task_name, task_val in tasks.items():
                acc_value = task_val if not isinstance(task_val, dict) else task_val.get("accuracy")
                try:
                    if acc_value is not None and float(acc_value) == 0.0:
                        return "Eval Failed"
                except (TypeError, ValueError):
                    pass

    if qs_status == "success" and acc_status == "success":
        return "Finished"
    if acc_status == "success":
        return "Finished"
    if qs_status == "success" and acc_status == "partial":
        return "Partial"
    return "Partial"


def upload_file(service, container: str, blob_path: str, local_path: Path) -> None:
    """Upload a single local file to blob_path in the given container."""
    blob_client = service.get_blob_client(container=container, blob=blob_path)
    with local_path.open("rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    print(f"[azure-upload]   uploaded: {blob_path}")


def upload_dir(service, container: str, blob_prefix: str, local_dir: Path) -> list[str]:
    """Recursively upload all files in local_dir under blob_prefix.

    Returns a list of blob paths that were uploaded.
    """
    uploaded: list[str] = []
    if not local_dir.exists() or not local_dir.is_dir():
        return uploaded
    for local_file in sorted(local_dir.rglob("*")):
        if not local_file.is_file():
            continue
        relative = local_file.relative_to(local_dir)
        blob_path = f"{blob_prefix}/{relative}".replace("\\", "/")
        upload_file(service, container, blob_path, local_file)
        uploaded.append(blob_path)
    return uploaded


def write_back_status_blob(
    service,
    container: str,
    request_filename: str,
    new_status: str,
    max_retries: int = 5,
) -> None:
    """Read-modify-write the status/<request_filename> blob using ETag optimistic locking.

    Retries on ResourceModifiedError (concurrent write detected).
    """
    if not request_filename:
        print("[azure-upload] No --request-filename provided; skipping status write-back")
        return

    from azure.core.exceptions import ResourceNotFoundError, ResourceModifiedError
    from azure.storage.blob import BlobClient
    from azure.storage.blob._models import MatchConditions

    blob_name = f"status/{request_filename}"
    blob_client: BlobClient = service.get_blob_client(container=container, blob=blob_name)

    for attempt in range(1, max_retries + 1):
        try:
            downloader = blob_client.download_blob()
            content = downloader.readall()
            etag = downloader.properties.etag
            data = json.loads(content.decode("utf-8"))
        except ResourceNotFoundError:
            print(f"[azure-upload] Status blob '{blob_name}' not found; skipping write-back")
            return
        except Exception as exc:
            print(f"[azure-upload] WARNING: could not read status blob '{blob_name}': {exc}")
            return

        old_status = data.get("status", "unknown")
        data["status"] = new_status
        new_content = (json.dumps(data, indent=4, ensure_ascii=False) + "\n").encode("utf-8")

        try:
            blob_client.upload_blob(
                new_content,
                overwrite=True,
                etag=etag,
                match_condition=MatchConditions.IfNotModified,
            )
            print(f"[azure-upload] Status write-back: {blob_name} ({old_status} -> {new_status})")
            return
        except ResourceModifiedError:
            print(f"[azure-upload] ETag conflict on attempt {attempt}/{max_retries}; retrying...")
            time.sleep(1)
        except Exception as exc:
            print(f"[azure-upload] WARNING: failed to write-back status blob '{blob_name}': {exc}")
            return

    print(f"[azure-upload] WARNING: status write-back failed after {max_retries} retries")


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload result artifacts to Azure Blob Storage")
    parser.add_argument("runtime_output_dir", help="Directory containing quant/eval runtime artifacts")
    parser.add_argument("model_id", help="Original model id, e.g. Qwen/Qwen3-0.6B")
    parser.add_argument("--pipeline", default="", help="Pipeline label: auto_quant or auto_eval")
    parser.add_argument("--scheme", default="W4A16", help="Quantization scheme label")
    parser.add_argument("--quant-num-gpus", default="", help="Quantization GPU count")
    parser.add_argument("--eval-num-gpus", default="", help="Evaluation GPU count")
    parser.add_argument(
        "--model-output-dir",
        default="",
        help="Optional model directory associated with this runtime output",
    )
    parser.add_argument(
        "--request-filename",
        default="",
        help="Original request JSON filename (e.g. Qwen3-0.6B_quant_request_False_W4A16_4bit_int4.json). "
             "Used to write back status and recorded in the aggregate JSON.",
    )
    parser.add_argument(
        "--azure-account-name",
        default=os.environ.get("AZURE_STORAGE_ACCOUNT", ""),
        help="Azure Storage account name (or set AZURE_STORAGE_ACCOUNT env var)",
    )
    parser.add_argument(
        "--azure-account-key",
        default=os.environ.get("AZURE_STORAGE_KEY", ""),
        help="Azure Storage account key (or set AZURE_STORAGE_KEY env var)",
    )
    parser.add_argument(
        "--azure-container",
        default=os.environ.get("AZURE_STORAGE_CONTAINER", "lb-eval-results"),
        help="Azure Blob Storage container name (default: lb-eval-results)",
    )
    # Keep --skip-github flag for backward compatibility with existing callers
    parser.add_argument(
        "--skip-github",
        action="store_true",
        help="(Ignored — kept for backward compatibility)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded but do not actually upload",
    )
    args = parser.parse_args()

    runtime_output_dir = Path(args.runtime_output_dir).resolve()
    if not runtime_output_dir.is_dir():
        print(f"ERROR: runtime output directory not found: {runtime_output_dir}")
        return 1

    model_output_dir = None
    if args.model_output_dir.strip():
        model_output_dir = Path(args.model_output_dir).resolve()

    if not args.dry_run:
        account_name = args.azure_account_name.strip()
        account_key = args.azure_account_key.strip()
        if not account_name or not account_key:
            print("ERROR: Azure Storage credentials are required. "
                  "Set AZURE_STORAGE_ACCOUNT and AZURE_STORAGE_KEY environment variables "
                  "or pass --azure-account-name / --azure-account-key.")
            return 1

        try:
            from azure.storage.blob import BlobServiceClient
        except ImportError:
            print("ERROR: azure-storage-blob is not installed. "
                  "Run: pip install azure-storage-blob")
            return 1

        service = BlobServiceClient(
            account_url=f"https://{account_name}.blob.core.windows.net",
            credential=account_key,
        )
        # Ensure container exists
        container = args.azure_container
        try:
            service.create_container(container)
            print(f"[azure-upload] Created container: {container}")
        except Exception as exc:
            from azure.core.exceptions import ResourceExistsError
            if isinstance(exc, ResourceExistsError):
                pass
            else:
                print(f"[azure-upload] WARNING: could not create container: {exc}")
    else:
        service = None
        container = args.azure_container

    quant_summary_path = runtime_output_dir / "quant_summary.json"
    summary_path = runtime_output_dir / "summary.json"
    accuracy_path = runtime_output_dir / "accuracy.json"
    quantize_script_path = runtime_output_dir / "quantize.py"
    legacy_quantize_script_path = runtime_output_dir / "quantize_script.py"
    lm_eval_results_dir = runtime_output_dir / "lm_eval_results"
    logs_dir = runtime_output_dir / "logs"
    quant_summary = load_json(quant_summary_path) or load_json(summary_path)
    accuracy = load_json(accuracy_path)

    artifact_name = detect_artifact_name(args.model_id, args.scheme, quant_summary)
    org = args.model_id.split("/", 1)[0] if "/" in args.model_id else "unknown"
    timestamp = file_timestamp()

    run_blob_prefix = f"results/{org}/{artifact_name}/run_{timestamp}"
    aggregate_blob_path = f"results/{org}/{artifact_name}/results_{timestamp}.json"

    print(f"[azure-upload] container       : {container}")
    print(f"[azure-upload] model id        : {args.model_id}")
    print(f"[azure-upload] artifact name   : {artifact_name}")
    print(f"[azure-upload] result org      : {org}")
    print(f"[azure-upload] run blob prefix : {run_blob_prefix}")

    uploaded_blobs: list[str] = []

    def _upload_file(local_path: Path, blob_name: str) -> None:
        if not local_path.exists() or not local_path.is_file():
            return
        if args.dry_run:
            print(f"[azure-upload] (dry-run) would upload: {blob_name}")
        else:
            upload_file(service, container, blob_name, local_path)
        uploaded_blobs.append(blob_name)

    def _upload_dir(local_dir: Path, blob_prefix: str) -> None:
        if not local_dir.exists() or not local_dir.is_dir():
            return
        if args.dry_run:
            for local_file in sorted(local_dir.rglob("*")):
                if local_file.is_file():
                    relative = local_file.relative_to(local_dir)
                    blob_name = f"{blob_prefix}/{relative}".replace("\\", "/")
                    print(f"[azure-upload] (dry-run) would upload: {blob_name}")
                    uploaded_blobs.append(blob_name)
        else:
            blobs = upload_dir(service, container, blob_prefix, local_dir)
            uploaded_blobs.extend(blobs)

    # Upload individual artifact files
    _upload_file(quant_summary_path, f"{run_blob_prefix}/quant_summary.json")
    _upload_file(summary_path, f"{run_blob_prefix}/summary.json")
    _upload_file(accuracy_path, f"{run_blob_prefix}/accuracy.json")
    if quantize_script_path.is_file():
        _upload_file(quantize_script_path, f"{run_blob_prefix}/quantize.py")
    elif legacy_quantize_script_path.is_file():
        _upload_file(legacy_quantize_script_path, f"{run_blob_prefix}/quantize.py")
    _upload_dir(lm_eval_results_dir, f"{run_blob_prefix}/lm_eval_results")
    _upload_dir(logs_dir, f"{run_blob_prefix}/logs")
    for path in sorted(runtime_output_dir.glob("session_*.jsonl")):
        _upload_file(path, f"{run_blob_prefix}/{path.name}")
    for path in sorted(runtime_output_dir.glob("session_*.md")):
        _upload_file(path, f"{run_blob_prefix}/{path.name}")

    if not uploaded_blobs:
        print("[azure-upload] No artifacts found to upload.")
        return 0

    # Build and upload aggregate JSON
    aggregate = {
        "pipeline": args.pipeline or ("auto_quant" if quant_summary else "auto_eval"),
        "model_id": args.model_id,
        "artifact_name": artifact_name,
        "request_filename": args.request_filename or None,
        "generated_at": utc_now(),
        "source_runtime_dir": str(runtime_output_dir),
        "source_model_dir": str(model_output_dir) if model_output_dir else None,
        "run_blob_prefix": run_blob_prefix,
        "quant_summary": quant_summary,
        "accuracy": accuracy,
        "uploaded_blobs": uploaded_blobs,
    }
    if aggregate["pipeline"] == "auto_quant":
        aggregate["quant_num_gpus"] = args.quant_num_gpus or (
            str(quant_summary.get("quant_num_gpus") or quant_summary.get("num_gpus"))
            if isinstance(quant_summary, dict) and (quant_summary.get("quant_num_gpus") or quant_summary.get("num_gpus")) is not None
            else None
        )
        aggregate["eval_num_gpus"] = args.eval_num_gpus or (
            str(accuracy.get("eval_num_gpus") or accuracy.get("num_gpus"))
            if isinstance(accuracy, dict) and (accuracy.get("eval_num_gpus") or accuracy.get("num_gpus")) is not None
            else None
        )
    else:
        aggregate["num_gpus"] = args.eval_num_gpus or args.quant_num_gpus or (
            str(accuracy.get("num_gpus"))
            if isinstance(accuracy, dict) and accuracy.get("num_gpus") is not None
            else None
        )

    aggregate_content = (json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if args.dry_run:
        print(f"[azure-upload] (dry-run) would upload: {aggregate_blob_path}")
    else:
        blob_client = service.get_blob_client(container=container, blob=aggregate_blob_path)
        blob_client.upload_blob(aggregate_content, overwrite=True)
        print(f"[azure-upload]   uploaded: {aggregate_blob_path}")
    uploaded_blobs.append(aggregate_blob_path)

    # Write-back status to the status blob
    pipeline_status = derive_pipeline_status(quant_summary, accuracy)
    print(f"[azure-upload] Derived pipeline status: {pipeline_status}")
    if not args.dry_run:
        write_back_status_blob(service, container, args.request_filename, pipeline_status)
    else:
        if args.request_filename:
            print(f"[azure-upload] (dry-run) would write-back status/{args.request_filename}: {pipeline_status}")

    print("[azure-upload] Uploaded blobs:")
    for blob_path in uploaded_blobs:
        print(f"  - {blob_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
