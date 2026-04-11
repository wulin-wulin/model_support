from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "models.yaml"


def load_config(config_path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_models(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return config["models"]


def get_model(config: dict[str, Any], alias: str) -> dict[str, Any]:
    models = get_models(config)
    if alias not in models:
        known = ", ".join(sorted(models))
        raise KeyError(f"Unknown model alias '{alias}'. Known aliases: {known}")
    return models[alias]


def get_source_repo_id(
    model_cfg: dict[str, Any],
    source: str,
    override_repo_id: str | None = None,
) -> str:
    if override_repo_id:
        return override_repo_id
    repo_id = model_cfg["repo_ids"].get(source)
    if not repo_id:
        raise ValueError(
            f"Model '{model_cfg['display_name']}' does not define a '{source}' repo id. "
            "Please pass --repo-id explicitly."
        )
    return repo_id


def get_model_dir(
    config: dict[str, Any],
    model_cfg: dict[str, Any],
    source: str,
) -> str:
    base_dir = PurePosixPath(config["paths"]["models"][source])
    return str(base_dir / model_cfg["local_dir_name"])


def get_cache_env(config: dict[str, Any]) -> dict[str, str]:
    caches = config["paths"]["caches"]
    return {
        "HOME": str(config["paths"]["home_dir"]),
        "HF_HOME": str(caches["hf_home"]),
        "HF_HUB_CACHE": str(caches["hf_hub_cache"]),
        "HF_XET_CACHE": str(caches["hf_xet_cache"]),
        "HF_ASSETS_CACHE": str(caches["hf_assets_cache"]),
        "XDG_CACHE_HOME": str(caches["xdg_cache_home"]),
        "XDG_CONFIG_HOME": str(caches["xdg_config_home"]),
        "XDG_DATA_HOME": str(caches["xdg_data_home"]),
        "MODELSCOPE_CACHE": str(caches["modelscope_cache"]),
        "VLLM_CACHE_ROOT": str(caches["vllm_cache_root"]),
        "VLLM_CONFIG_ROOT": str(caches["vllm_config_root"]),
        "VLLM_ASSETS_CACHE": str(caches["vllm_assets_cache"]),
        "VLLM_RPC_BASE_PATH": str(caches["vllm_rpc_base_path"]),
        "PIP_CACHE_DIR": str(caches["pip_cache_dir"]),
        "PIP_SRC": str(caches["pip_src"]),
        "UV_CACHE_DIR": str(caches["uv_cache_dir"]),
        "TMPDIR": str(caches["tmp_dir"]),
        "TEMP": str(caches["tmp_dir"]),
        "TMP": str(caches["tmp_dir"]),
        "TORCH_HOME": str(caches["torch_home"]),
        "TORCH_EXTENSIONS_DIR": str(caches["torch_extensions_dir"]),
        "TORCHINDUCTOR_CACHE_DIR": str(caches["torchinductor_cache_dir"]),
        "TRITON_CACHE_DIR": str(caches["triton_cache_dir"]),
        "TRITON_HOME": str(caches["triton_home"]),
        "CUDA_CACHE_PATH": str(caches["cuda_cache_path"]),
    }


def ensure_dirs(paths: list[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def merge_vllm_settings(config: dict[str, Any], model_cfg: dict[str, Any]) -> dict[str, Any]:
    merged = dict(config.get("defaults", {}))
    merged.update(model_cfg.get("vllm", {}))
    return merged


def format_table(rows: list[list[str]]) -> str:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines: list[str] = []
    for idx, row in enumerate(rows):
        padded = [cell.ljust(widths[i]) for i, cell in enumerate(row)]
        lines.append("  ".join(padded))
        if idx == 0:
            lines.append("  ".join("-" * width for width in widths))
    return "\n".join(lines)
