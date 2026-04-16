from __future__ import annotations

import argparse
import os
import site
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path


def extract_old_data_segments(value: str) -> list[str]:
    segments: list[str] = []
    for part in value.split(os.pathsep):
        if part == "/data" or part.startswith("/data/"):
            segments.append(part)
    return segments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check whether writable paths stay inside the project root.")
    parser.add_argument(
        "--project-root",
        default="/home/dataset-local/data/zos_download/model_support",
        help="Expected project root on the server",
    )
    parser.add_argument(
        "--allow-prefix",
        action="append",
        default=[],
        help="Additional allowed path prefixes, e.g. /home/dataset-local/data/zos_download/conda/envs/model_support",
    )
    return parser


def normalize(path_str: str) -> str:
    return str(Path(path_str).resolve())


def is_allowed(path_str: str, allowed_roots: list[str]) -> bool:
    try:
        normalized = normalize(path_str)
    except Exception:  # noqa: BLE001
        normalized = path_str
    return any(normalized.startswith(root) for root in allowed_roots)


def collect_paths() -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    env_keys = [
        "MODEL_SUPPORT_ROOT",
        "HOME",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_XET_CACHE",
        "HF_ASSETS_CACHE",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "MODELSCOPE_CACHE",
        "VLLM_CACHE_ROOT",
        "VLLM_CONFIG_ROOT",
        "VLLM_ASSETS_CACHE",
        "VLLM_RPC_BASE_PATH",
        "PIP_CACHE_DIR",
        "PIP_SRC",
        "UV_CACHE_DIR",
        "TMPDIR",
        "TEMP",
        "TMP",
        "TORCH_HOME",
        "TORCH_EXTENSIONS_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "TRITON_HOME",
        "CUDA_CACHE_PATH",
    ]
    for key in env_keys:
        value = os.getenv(key)
        if value:
            items.append((f"env:{key}", value))

    items.append(("python:sys.prefix", sys.prefix))
    items.append(("python:base_prefix", sys.base_prefix))
    items.append(("python:purelib", sysconfig.get_path("purelib") or ""))
    items.append(("python:platlib", sysconfig.get_path("platlib") or ""))
    items.append(("python:tempdir", tempfile.gettempdir()))

    for idx, path_str in enumerate(site.getsitepackages()):
        items.append((f"python:sitepackages[{idx}]", path_str))

    seen = {name.removeprefix("env:") for name, _ in items if name.startswith("env:")}
    for key, value in os.environ.items():
        if key in seen:
            continue
        for segment in extract_old_data_segments(value):
            items.append((f"env:suspicious:{key}", segment))

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "cache", "dir"],
            check=True,
            capture_output=True,
            text=True,
        )
        items.append(("pip:cache_dir", result.stdout.strip()))
    except Exception as exc:  # noqa: BLE001
        items.append(("pip:cache_dir", f"<unavailable: {exc}>"))

    return items


def main() -> None:
    args = build_parser().parse_args()
    allowed_roots = [normalize(args.project_root)] + [normalize(path) for path in args.allow_prefix]
    rows = collect_paths()

    problems: list[tuple[str, str]] = []
    print("Allowed roots:")
    for root in allowed_roots:
        print(f"  - {root}")
    print("")
    print("Path check:")
    for name, value in rows:
        if value.startswith("<unavailable:"):
            print(f"  [WARN] {name}: {value}")
            continue
        ok = is_allowed(value, allowed_roots)
        status = "OK" if ok else "OUTSIDE"
        print(f"  [{status}] {name}: {value}")
        if not ok:
            problems.append((name, value))

    print("")
    if problems:
        print("Conclusion: there are writable/runtime paths outside the allowed roots.")
        print("If you need strict project-only isolation, use a project-local virtualenv under /home/dataset-local/data/zos_download/model_support/.venv.")
        raise SystemExit(2)

    print("Conclusion: the checked paths are inside the allowed roots.")


if __name__ == "__main__":
    main()
