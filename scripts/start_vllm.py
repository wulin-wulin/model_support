from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path

from _model_registry import (
    ensure_dirs,
    get_cache_env,
    get_model,
    get_model_dir,
    load_config,
    merge_vllm_settings,
)


KEY_ENV_VARS = [
    "MODEL_SUPPORT_ROOT",
    "HOME",
    "HF_HOME",
    "HF_HUB_CACHE",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "MODELSCOPE_CACHE",
    "VLLM_CACHE_ROOT",
    "VLLM_CONFIG_ROOT",
    "VLLM_ASSETS_CACHE",
    "VLLM_RPC_BASE_PATH",
    "PIP_CACHE_DIR",
    "UV_CACHE_DIR",
    "TMPDIR",
    "TORCH_HOME",
    "TORCH_EXTENSIONS_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "TRITON_HOME",
    "CUDA_CACHE_PATH",
    "TORCHINDUCTOR_FORCE_DISABLE_CACHES",
    "TORCH_COMPILE_FORCE_DISABLE_CACHES",
    "TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE",
    "TORCHINDUCTOR_BUNDLED_AUTOTUNE_REMOTE_CACHE",
    "TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE",
    "VLLM_API_KEY",
    "CUDA_VISIBLE_DEVICES",
]

TORCH_COMPILE_CACHE_ENV_OVERRIDES = {
    "TORCHINDUCTOR_FORCE_DISABLE_CACHES": "1",
    "TORCH_COMPILE_FORCE_DISABLE_CACHES": "1",
    "TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE": "0",
    "TORCHINDUCTOR_BUNDLED_AUTOTUNE_REMOTE_CACHE": "0",
    "TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE": "0",
}

PURGE_CACHE_DIR_KEYS = [
    "vllm_cache_root",
    "torchinductor_cache_dir",
    "triton_cache_dir",
    "triton_home",
]


def get_installed_package_version(name: str) -> str | None:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None


def find_env_refs_under_old_data_root(env: dict[str, str]) -> list[tuple[str, str]]:
    matches: list[tuple[str, str]] = []
    for key, value in sorted(env.items()):
        for part in value.split(os.pathsep):
            if part == "/data" or part.startswith("/data/"):
                matches.append((key, part))
    return matches


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and optionally run a vLLM serve command.")
    parser.add_argument("--config", default=None, help="Path to models.yaml")
    parser.add_argument("--model", required=True, help="Model alias from configs/models.yaml")
    parser.add_argument("--source", choices=["hf", "mop"], default="hf", help="Model source dir")
    parser.add_argument("--model-path", help="Override the resolved local model path")
    parser.add_argument("--host", help="Override host, default from config")
    parser.add_argument("--port", type=int, help="Override port, default from config")
    parser.add_argument("--tensor-parallel-size", type=int, help="Override TP size")
    parser.add_argument("--max-model-len", type=int, help="Override max model length")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        help="Override gpu_memory_utilization",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Force eager execution and disable the CUDA graph path",
    )
    parser.add_argument(
        "--max-cudagraph-capture-size",
        type=int,
        help="Cap CUDA graph capture size for startup stability on some models",
    )
    parser.add_argument("--max-num-seqs", type=int, help="Override max_num_seqs")
    parser.add_argument(
        "--served-model-name",
        help="Override served model name exposed through /v1/models",
    )
    parser.add_argument(
        "--language-model-only",
        action="store_true",
        help="Add --language-model-only even if not present in config",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Append raw vLLM arguments, e.g. --extra-arg=\"--enable-auto-tool-choice\"",
    )
    parser.add_argument(
        "--disable-torch-compile-caches",
        action="store_true",
        help="Disable Torch compile/autotune caches for this run to avoid stale cache path issues",
    )
    parser.add_argument(
        "--purge-compile-caches",
        action="store_true",
        help="Delete configured vLLM/Torch/Triton compile cache directories before launch",
    )
    parser.add_argument(
        "--skip-preflight-checks",
        action="store_true",
        help="Skip lightweight local validation such as multimodal processor loading",
    )
    parser.add_argument("--print-only", action="store_true", help="Print command and exit")
    parser.add_argument("--run", action="store_true", help="Run the generated command")
    return parser


def get_compile_cache_dirs(config: dict) -> list[Path]:
    caches = config["paths"]["caches"]
    return [Path(str(caches[key])) for key in PURGE_CACHE_DIR_KEYS]


def purge_cache_dirs(paths: list[Path], allowed_root: Path) -> None:
    resolved_root = allowed_root.resolve(strict=False)
    for path in paths:
        resolved_path = path.resolve(strict=False)
        if resolved_path == resolved_root or not resolved_path.is_relative_to(resolved_root):
            raise ValueError(
                f"Refusing to purge cache dir outside server_root: {resolved_path} (server_root={resolved_root})"
            )
        if path.exists():
            print(f"Purging cache dir: {path}")
            shutil.rmtree(path)


def load_processor_class_name(model_path: Path) -> str | None:
    for filename in ["processor_config.json", "preprocessor_config.json"]:
        path = model_path / filename
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        processor_class = data.get("processor_class")
        if isinstance(processor_class, str) and processor_class:
            return processor_class
    return None


def run_preflight_checks(
    args: argparse.Namespace,
    model_cfg: dict,
    model_path: Path,
) -> list[str]:
    issues: list[str] = []
    if args.skip_preflight_checks:
        return issues
    if model_cfg.get("modality") != "multimodal" or args.language_model_only:
        return issues
    if not model_path.exists():
        issues.append(f"Model path does not exist: {model_path}")
        return issues

    processor_class_name = load_processor_class_name(model_path)
    transformers_version = get_installed_package_version("transformers") or "unknown"
    vllm_version = get_installed_package_version("vllm") or "unknown"

    try:
        from transformers import AutoProcessor
        from transformers.processing_utils import ProcessorMixin
    except Exception as exc:
        issues.append(f"Failed to import transformers processor utilities: {exc}")
        return issues

    try:
        processor = AutoProcessor.from_pretrained(str(model_path))
    except Exception as exc:
        issue = f"AutoProcessor.from_pretrained({model_path}) failed: {exc}"
        if processor_class_name == "Glm46VProcessor":
            issue += (
                " The model config expects Glm46VProcessor; the official GLM-4.6V model card "
                "requires transformers>=5.0.0rc0."
            )
            if vllm_version == "0.19.0":
                issue += " Current vLLM appears to be 0.19.0; vLLM 0.19.1 upgraded to Transformers v5.5.4."
        issues.append(issue)
        return issues

    if not isinstance(processor, ProcessorMixin):
        issue = (
            "AutoProcessor returned "
            f"{type(processor).__name__} instead of a ProcessorMixin "
            f"(transformers={transformers_version}, vllm={vllm_version})."
        )
        if processor_class_name == "Glm46VProcessor":
            issue += (
                " This usually means the current transformers build is too old for GLM-4.6V. "
                "The official GLM-4.6V model card requires transformers>=5.0.0rc0, and "
                "vLLM 0.19.1 upgraded to Transformers v5.5.4."
            )
        issues.append(issue)

    return issues


def build_command(args: argparse.Namespace, config: dict, model_cfg: dict) -> tuple[list[str], dict[str, str]]:
    vllm_cfg = merge_vllm_settings(config, model_cfg)
    model_path_str = args.model_path if args.model_path else get_model_dir(config, model_cfg, args.source)

    command: list[str] = ["vllm", "serve", model_path_str]
    command.extend(["--host", args.host or str(vllm_cfg["host"])])
    command.extend(["--port", str(args.port or vllm_cfg["default_port"])])
    command.extend(
        [
            "--served-model-name",
            args.served_model_name or str(vllm_cfg["served_model_name"]),
        ]
    )
    command.extend(
        [
            "--tensor-parallel-size",
            str(args.tensor_parallel_size or vllm_cfg["tensor_parallel_size"]),
        ]
    )
    command.extend(["--max-model-len", str(args.max_model_len or vllm_cfg["max_model_len"])])
    command.extend(
        [
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization or vllm_cfg["gpu_memory_utilization"]),
        ]
    )
    command.extend(["--max-num-seqs", str(args.max_num_seqs or vllm_cfg["max_num_seqs"])])
    if args.max_cudagraph_capture_size is not None:
        command.extend(["--max-cudagraph-capture-size", str(args.max_cudagraph_capture_size)])
    command.extend(["--generation-config", str(vllm_cfg["generation_config"])])

    if vllm_cfg.get("enable_prefix_caching"):
        command.append("--enable-prefix-caching")
    if vllm_cfg.get("enforce_eager") or args.enforce_eager:
        command.append("--enforce-eager")
    if vllm_cfg.get("reasoning_parser"):
        command.extend(["--reasoning-parser", str(vllm_cfg["reasoning_parser"])])
    if vllm_cfg.get("limit_mm_per_prompt"):
        command.extend(["--limit-mm-per-prompt", str(vllm_cfg["limit_mm_per_prompt"])])
    if vllm_cfg.get("mm_processor_kwargs"):
        command.extend(["--mm-processor-kwargs", str(vllm_cfg["mm_processor_kwargs"])])
    if vllm_cfg.get("allowed_local_media_path"):
        command.extend(["--allowed-local-media-path", str(vllm_cfg["allowed_local_media_path"])])
    if vllm_cfg.get("tool_call_parser"):
        command.extend(["--tool-call-parser", str(vllm_cfg["tool_call_parser"])])
    if vllm_cfg.get("enable_auto_tool_choice"):
        command.append("--enable-auto-tool-choice")
    if vllm_cfg.get("language_model_only") or args.language_model_only:
        command.append("--language-model-only")

    for raw_arg in vllm_cfg.get("extra_args", []):
        command.extend(shlex.split(str(raw_arg)))
    for raw_arg in args.extra_arg:
        command.extend(shlex.split(raw_arg))

    env = os.environ.copy()
    env_defaults = get_cache_env(config)
    env.update(env_defaults)
    if args.disable_torch_compile_caches:
        env.update(TORCH_COMPILE_CACHE_ENV_OVERRIDES)

    return command, env


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    model_cfg = get_model(config, args.model)
    model_path = Path(args.model_path) if args.model_path else Path(get_model_dir(config, model_cfg, args.source))

    command, env = build_command(args, config, model_cfg)

    print("Resolved command:")
    print("  " + " ".join(shlex.quote(part) for part in command))
    print("")
    print("Key env:")
    for key in KEY_ENV_VARS:
        if key in env:
            print(f"  {key}={env[key]}")

    old_data_refs = find_env_refs_under_old_data_root(env)
    if old_data_refs:
        print("")
        print("WARNING: final environment still contains paths under /data:")
        for key, value in old_data_refs[:20]:
            print(f"  {key}={value}")
        if len(old_data_refs) > 20:
            print(f"  ... and {len(old_data_refs) - 20} more")

    preflight_issues = run_preflight_checks(args, model_cfg, model_path)
    if preflight_issues:
        print("")
        print("Preflight issues:")
        for issue in preflight_issues:
            print(f"  - {issue}")
        if args.run:
            raise SystemExit(
                "Refusing to launch vLLM because preflight checks failed. "
                "Use --skip-preflight-checks only if you have already verified the environment."
            )

    if args.print_only or not args.run:
        return

    env_defaults = get_cache_env(config)
    server_root = Path(str(config["paths"]["server_root"]))
    if args.purge_compile_caches:
        purge_cache_dirs(get_compile_cache_dirs(config), server_root)
    ensure_dirs([model_path.parent] + [Path(value) for value in env_defaults.values()])

    subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
