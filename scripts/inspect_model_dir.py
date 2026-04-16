from __future__ import annotations

import argparse
from pathlib import Path


TEXT_EXTENSIONS = {
    ".json",
    ".jsonl",
    ".txt",
    ".md",
    ".yaml",
    ".yml",
    ".py",
    ".ini",
    ".cfg",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect a local model directory for symlinks and suspicious absolute path references.",
    )
    parser.add_argument("model_dir", help="Model directory to inspect")
    parser.add_argument(
        "--pattern",
        default="/data",
        help="Absolute path prefix to flag, default: /data",
    )
    parser.add_argument(
        "--max-findings",
        type=int,
        default=200,
        help="Maximum number of findings to print",
    )
    return parser


def is_text_candidate(path: Path) -> bool:
    return path.suffix.lower() in TEXT_EXTENSIONS


def inspect_model_dir(model_dir: Path, pattern: str, max_findings: int) -> int:
    findings: list[str] = []

    if not model_dir.exists():
        print(f"[ERROR] model_dir does not exist: {model_dir}")
        return 2
    if not model_dir.is_dir():
        print(f"[ERROR] model_dir is not a directory: {model_dir}")
        return 2

    resolved_root = model_dir.resolve()
    print(f"model_dir      : {model_dir}")
    print(f"resolved_root  : {resolved_root}")
    print(f"pattern        : {pattern}")
    print("")

    for path in sorted(model_dir.rglob("*")):
        if len(findings) >= max_findings:
            break

        try:
            if path.is_symlink():
                target = path.readlink()
                resolved = path.resolve(strict=False)
                if pattern in str(target) or pattern in str(resolved):
                    findings.append(
                        f"[SYMLINK] {path} -> {target} (resolved: {resolved})",
                    )
                continue

            resolved = path.resolve(strict=False)
            if pattern in str(resolved):
                findings.append(f"[REALPATH] {path} -> {resolved}")
                continue

            if path.is_file() and is_text_candidate(path):
                try:
                    content = path.read_text(encoding="utf-8", errors="ignore")
                except OSError as exc:
                    findings.append(f"[READ-ERROR] {path}: {exc}")
                    continue
                if pattern in content:
                    findings.append(f"[CONTENT] {path}")
        except OSError as exc:
            findings.append(f"[STAT-ERROR] {path}: {exc}")

    if findings:
        print("findings:")
        for finding in findings:
            print(f"  {finding}")
        if len(findings) >= max_findings:
            print("")
            print(f"Stopped after {max_findings} findings.")
        return 1

    print("No suspicious references found.")
    return 0


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(inspect_model_dir(Path(args.model_dir), args.pattern, args.max_findings))


if __name__ == "__main__":
    main()
