from __future__ import annotations

import argparse
import inspect
import os
import subprocess
from pathlib import Path

from _model_registry import (
    ensure_dirs,
    get_cache_env,
    get_model,
    get_model_dir,
    get_source_repo_id,
    load_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download a model into /home/dataset-local/data/zos_download/model_support.")
    parser.add_argument("--config", default=None, help="Path to models.yaml")
    parser.add_argument("--model", required=True, help="Model alias from configs/models.yaml")
    parser.add_argument("--source", choices=["hf", "mop"], default="hf", help="Download source")
    parser.add_argument("--repo-id", help="Override repo id for the selected source")
    parser.add_argument("--revision", help="Optional revision / branch / tag")
    parser.add_argument("--max-workers", type=int, default=8, help="Download worker count")
    parser.add_argument("--dry-run", action="store_true", help="Only show what would happen")
    parser.add_argument(
        "--command-only",
        action="store_true",
        help="Print the underlying command and exit without downloading",
    )
    return parser


def print_plan(source: str, repo_id: str, target_dir: str) -> None:
    print(f"source    : {source}")
    print(f"repo_id   : {repo_id}")
    print(f"target_dir: {target_dir}")


def download_hf(repo_id: str, target_dir: Path, revision: str | None, max_workers: int) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is not installed. Run: python -m pip install -r requirements/requirements-server.txt"
        ) from exc

    kwargs = {
        "repo_id": repo_id,
        "local_dir": str(target_dir),
        "max_workers": max_workers,
        "token": os.getenv("HF_TOKEN"),
    }
    if revision:
        kwargs["revision"] = revision
    snapshot_download(**kwargs)


def try_modelscope_python_api(repo_id: str, target_dir: Path, revision: str | None) -> bool:
    try:
        from modelscope import snapshot_download as ms_snapshot_download
    except ImportError:
        return False

    sig = inspect.signature(ms_snapshot_download)
    kwargs: dict[str, object] = {}
    if "local_dir" in sig.parameters:
        kwargs["local_dir"] = str(target_dir)
    if "cache_dir" in sig.parameters and "local_dir" not in sig.parameters:
        kwargs["cache_dir"] = str(target_dir.parent)
    if "revision" in sig.parameters and revision:
        kwargs["revision"] = revision
    token = os.getenv("MODELSCOPE_API_TOKEN")
    if "api_token" in sig.parameters and token:
        kwargs["api_token"] = token

    ms_snapshot_download(repo_id, **kwargs)
    return True


def download_modelscope(repo_id: str, target_dir: Path, revision: str | None) -> None:
    if try_modelscope_python_api(repo_id, target_dir, revision):
        return

    command = ["modelscope", "download", "--model", repo_id, "--local_dir", str(target_dir)]
    if revision:
        command.extend(["--revision", revision])
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise SystemExit(
            "Neither the ModelScope Python API nor the `modelscope` CLI is available. "
            "Install it with: python -m pip install -r requirements/requirements-server.txt"
        ) from exc


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    model_cfg = get_model(config, args.model)
    repo_id = get_source_repo_id(model_cfg, args.source, args.repo_id)
    target_dir_str = get_model_dir(config, model_cfg, args.source)

    env_hint = get_cache_env(config)
    for key, value in env_hint.items():
        os.environ.setdefault(key, value)

    print_plan(args.source, repo_id, target_dir_str)
    print("")
    print("env exports:")
    for key, value in env_hint.items():
        print(f"  export {key}={value}")
    print("")

    if args.source == "hf":
        command_preview = (
            f"hf download {repo_id} --local-dir {target_dir_str}"
            + (f" --revision {args.revision}" if args.revision else "")
        )
    else:
        command_preview = (
            f"modelscope download --model {repo_id} --local_dir {target_dir_str}"
            + (f" --revision {args.revision}" if args.revision else "")
        )
    print(f"command hint: {command_preview}")

    if args.dry_run or args.command_only:
        return

    target_dir = Path(target_dir_str)
    ensure_dirs([target_dir.parent] + [Path(value) for value in env_hint.values()])

    if args.source == "hf":
        download_hf(repo_id, target_dir, args.revision, args.max_workers)
    else:
        download_modelscope(repo_id, target_dir, args.revision)

    print("")
    print(f"Download finished: {target_dir}")


if __name__ == "__main__":
    main()
