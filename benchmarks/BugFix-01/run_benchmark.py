#!/usr/bin/env python3
"""BugFix-01: run the same prompt against 4 local Ollama coding models."""

from __future__ import annotations

import json
import time
from pathlib import Path

import ollama

BENCHMARK_DIR = Path(__file__).resolve().parent
PROMPT_PATH = BENCHMARK_DIR / "prompt" / "prompt.txt"
RESPONSES_DIR = BENCHMARK_DIR / "responses"
RESULTS_DIR = BENCHMARK_DIR / "results"

MODELS = [
    "qwen3.6:35b-a3b-coding-nvfp4",
    "qwen3-coder-fast:30b",
    "north-mini-code-1.0:mlx-nvfp4",
    "glm-4.7-flash:latest",
]


def slug(model: str) -> str:
    return model.replace(":", "_").replace("/", "_").replace(".", "_")


def main() -> None:
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []

    for model in MODELS:
        print(f"\n=== Running {model} ===", flush=True)
        start = time.perf_counter()
        error = None
        content = ""
        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1},
            )
            content = response["message"]["content"]
        except Exception as exc:  # noqa: BLE001 — capture for benchmark log
            error = str(exc)
            content = f"[ERROR]\n{error}"
        elapsed_s = round(time.perf_counter() - start, 2)

        out_path = RESPONSES_DIR / f"{slug(model)}.md"
        out_path.write_text(
            f"# Model: `{model}`\n\n"
            f"- Elapsed: **{elapsed_s}s**\n"
            f"- Error: `{error}`\n\n"
            f"## Response\n\n{content}\n",
            encoding="utf-8",
        )

        row = {
            "model": model,
            "elapsed_s": elapsed_s,
            "error": error,
            "response_path": str(out_path.relative_to(BENCHMARK_DIR)),
            "char_count": len(content),
        }
        summary_rows.append(row)
        print(f"Done in {elapsed_s}s -> {out_path.name}", flush=True)

    raw_summary = RESULTS_DIR / "raw_run_summary.json"
    raw_summary.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    print(f"\nWrote {raw_summary}", flush=True)


if __name__ == "__main__":
    main()
