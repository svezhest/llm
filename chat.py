#!/usr/bin/env python3
"""Streaming chat with Qwen3.6-35B-A3B (MLX, vision-language).

Examples:
  uv run python chat.py "Привет! Кто ты?"
  uv run python chat.py "Опиши картинку" --image photo.jpg
  uv run python chat.py "Solve 2x+3=7" --max-tokens 512 --temp 0.0
"""
import argparse
import sys
import time

DEFAULT_MODEL = "/Users/user/models/Qwen3.6-35B-A3B-UD-MLX-3bit"


def main():
    ap = argparse.ArgumentParser(description="Streaming chat with the local Qwen MLX model.")
    ap.add_argument("prompt", help="Prompt text.")
    ap.add_argument("--image", action="append", default=[],
                    help="Path/URL to an image (repeatable). Triggers vision mode.")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Model path or HF repo id.")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temp", type=float, default=0.7, help="Sampling temperature (0 = greedy).")
    ap.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False,
                    help="Reasoning pass. Off by default; use --enable-thinking to turn it on.")
    args = ap.parse_args()

    from mlx_vlm import load, stream_generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    print(f"loading {args.model} ...", file=sys.stderr, flush=True)
    t = time.time()
    model, processor = load(args.model)
    config = load_config(args.model)
    print(f"loaded in {time.time() - t:.1f}s\n", file=sys.stderr, flush=True)

    formatted = apply_chat_template(
        processor, config, args.prompt, num_images=len(args.image),
        enable_thinking=args.enable_thinking,
    )

    gen_tps = peak = 0.0
    for chunk in stream_generate(
        model, processor, formatted, args.image,
        max_tokens=args.max_tokens, temperature=args.temp,
    ):
        print(chunk.text, end="", flush=True)
        gen_tps = chunk.generation_tps
        peak = chunk.peak_memory
    print(f"\n\n[{gen_tps:.0f} tok/s, peak {peak:.1f} GB]", file=sys.stderr)


if __name__ == "__main__":
    main()
