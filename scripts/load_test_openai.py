from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from dataclasses import dataclass

from openai import AsyncOpenAI


@dataclass
class Result:
    ok: bool
    latency_s: float
    error: str | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple concurrency test for an OpenAI-compatible endpoint.")
    parser.add_argument("--base-url", required=True, help="Base URL, e.g. http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY", help="OpenAI-compatible API key")
    parser.add_argument("--model", required=True, help="Served model name")
    parser.add_argument("--prompt", default="请用一句话说明你支持什么能力。")
    parser.add_argument("--requests", type=int, default=60, help="Total request count")
    parser.add_argument("--concurrency", type=int, default=20, help="Max concurrent requests")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    return parser


async def run_once(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
) -> Result:
    async with semaphore:
        start = time.perf_counter()
        try:
            await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return Result(ok=True, latency_s=time.perf_counter() - start)
        except Exception as exc:  # noqa: BLE001
            return Result(ok=False, latency_s=time.perf_counter() - start, error=str(exc))


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    idx = max(0, min(len(values) - 1, round((len(values) - 1) * pct)))
    return sorted(values)[idx]


async def amain(args: argparse.Namespace) -> None:
    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)
    semaphore = asyncio.Semaphore(args.concurrency)
    started = time.perf_counter()
    tasks = [
        run_once(
            client=client,
            model=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            semaphore=semaphore,
        )
        for _ in range(args.requests)
    ]
    results = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - started

    latencies = [result.latency_s for result in results if result.ok]
    failures = [result for result in results if not result.ok]
    print(f"requests      : {args.requests}")
    print(f"concurrency   : {args.concurrency}")
    print(f"success       : {len(latencies)}")
    print(f"failures      : {len(failures)}")
    print(f"wall_time_s   : {elapsed:.2f}")
    print(f"throughput_rps: {len(latencies) / elapsed:.2f}" if elapsed else "throughput_rps: inf")
    if latencies:
        print(f"latency_avg_s : {statistics.mean(latencies):.2f}")
        print(f"latency_p50_s : {percentile(latencies, 0.50):.2f}")
        print(f"latency_p95_s : {percentile(latencies, 0.95):.2f}")
        print(f"latency_max_s : {max(latencies):.2f}")
    if failures:
        print("")
        print("sample_failures:")
        for failure in failures[:5]:
            print(f"  - {failure.error}")


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()

