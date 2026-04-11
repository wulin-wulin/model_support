from __future__ import annotations

import argparse
from pathlib import Path

from openai import OpenAI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke test an OpenAI-compatible vLLM endpoint.")
    parser.add_argument("--base-url", required=True, help="Base URL, e.g. http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY", help="OpenAI-compatible API key")
    parser.add_argument("--model", required=True, help="Served model name")
    parser.add_argument("--prompt", required=True, help="User prompt")
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--image-url", help="Optional remote image URL")
    parser.add_argument("--image-path", help="Optional local image path")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    return parser


def build_content(args: argparse.Namespace) -> list[dict]:
    content: list[dict] = []
    image_url = args.image_url
    if args.image_path:
        image_url = Path(args.image_path).resolve().as_uri()
    if image_url:
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    content.append({"type": "text", "text": args.prompt})
    return content


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": build_content(args)},
    ]

    response = client.chat.completions.create(
        model=args.model,
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()

