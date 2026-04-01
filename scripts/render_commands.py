from __future__ import annotations

import argparse
import json

from _model_registry import (
    format_table,
    get_model,
    get_model_dir,
    get_models,
    load_config,
    merge_vllm_settings,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render model download and serve hints.")
    parser.add_argument("--config", default=None, help="Path to models.yaml")
    parser.add_argument("--list", action="store_true", help="List all model aliases")
    parser.add_argument("--model", help="Render commands for a single model alias")
    parser.add_argument(
        "--source",
        choices=["hf", "mop"],
        default="hf",
        help="Preferred source when rendering concrete commands",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print model details as JSON instead of human-readable text",
    )
    return parser


def render_list(config: dict) -> str:
    rows = [["alias", "display_name", "port", "tp", "model_dir"]]
    for alias, model_cfg in sorted(get_models(config).items()):
        vllm_cfg = merge_vllm_settings(config, model_cfg)
        model_dir = get_model_dir(config, model_cfg, "hf")
        rows.append(
            [
                alias,
                model_cfg["display_name"],
                str(vllm_cfg["default_port"]),
                str(vllm_cfg["tensor_parallel_size"]),
                str(model_dir),
            ]
        )
    return format_table(rows)


def render_one(config: dict, alias: str, source: str) -> str:
    model_cfg = get_model(config, alias)
    vllm_cfg = merge_vllm_settings(config, model_cfg)
    model_dir = get_model_dir(config, model_cfg, source)
    lines = [
        f"alias: {alias}",
        f"display_name: {model_cfg['display_name']}",
        f"source: {source}",
        f"model_dir: {model_dir}",
        f"served_model_name: {vllm_cfg['served_model_name']}",
        f"default_port: {vllm_cfg['default_port']}",
        "",
        "download:",
        f"  python scripts/download_model.py --model {alias} --source {source} --dry-run",
        f"  python scripts/download_model.py --model {alias} --source {source}",
        "",
        "serve:",
        f"  python scripts/start_vllm.py --model {alias} --source {source} --print-only",
        f"  python scripts/start_vllm.py --model {alias} --source {source} --run",
        "",
        "smoke_test:",
        (
            f"  python scripts/smoke_test_openai.py "
            f"--base-url http://127.0.0.1:{vllm_cfg['default_port']}/v1 "
            f"--model {vllm_cfg['served_model_name']} "
            "--prompt \"请用三句话介绍你自己。\""
        ),
        "",
        "load_test_20_concurrency:",
        (
            f"  python scripts/load_test_openai.py "
            f"--base-url http://127.0.0.1:{vllm_cfg['default_port']}/v1 "
            f"--model {vllm_cfg['served_model_name']} "
            "--concurrency 20 --requests 60"
        ),
        "",
        "notes:",
    ]
    for note in model_cfg.get("notes", []):
        lines.append(f"  - {note}")
    return "\n".join(lines)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    if args.json and args.model:
        model_cfg = get_model(config, args.model)
        payload = dict(model_cfg)
        payload["resolved_model_dir"] = str(get_model_dir(config, model_cfg, args.source))
        payload["resolved_vllm"] = merge_vllm_settings(config, model_cfg)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.list or not args.model:
        print(render_list(config))
        if not args.list and not args.model:
            print("\nUse --model <alias> to render concrete commands.")
        return

    print(render_one(config, args.model, args.source))


if __name__ == "__main__":
    main()

