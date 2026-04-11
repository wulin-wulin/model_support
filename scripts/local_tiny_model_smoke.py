from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local tiny-model smoke test with transformers.")
    parser.add_argument(
        "--model-id",
        default="HuggingFaceTB/SmolLM2-135M-Instruct",
        help="A very small instruction-tuned model suitable for local smoke testing",
    )
    parser.add_argument(
        "--output-dir",
        default="local_artifacts/models/smollm2-135m-instruct",
        help="Directory to store the downloaded tiny model",
    )
    parser.add_argument("--prompt", default="请用两句话解释什么是 vLLM。")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    try:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing local dependencies. Run: python -m pip install -r requirements/requirements-local.txt"
        ) from exc

    output_dir = Path(args.output_dir).resolve()
    hf_home = output_dir.parents[2] / "huggingface-cache"
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))

    print(f"Downloading tiny model to: {output_dir}")
    model_path = snapshot_download(repo_id=args.model_id, local_dir=str(output_dir), max_workers=4)
    model_path = Path(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Loading from: {model_path}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    model.to(device)
    model.eval()

    messages = [{"role": "user", "content": args.prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        encoded = tokenizer(args.prompt, return_tensors="pt").input_ids
    encoded = encoded.to(device)

    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - start

    new_tokens = output_ids[0][encoded.shape[-1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print("")
    print(f"generation_time_s: {elapsed:.2f}")
    print("response:")
    print(text.strip())


if __name__ == "__main__":
    main()

