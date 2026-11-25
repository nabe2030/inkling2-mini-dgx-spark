#!/usr/bin/env python3
"""Warm up Inkling 2's MoE Triton kernels on SM12x (DGX Spark / GB10).

On SM12x, vLLM runs Inkling 2's expert GEMMs as Triton kernels whose launch plan
(`max_ctas`) is a compile-time constant derived from the number of tokens T in the
step, so every new prompt length can trigger a recompilation
(see https://github.com/vllm-project/vllm/pull/69288).

This script enumerates every kernel variant reachable with a prompt of up to
--max-num-batched-tokens tokens, picks one representative length per variant, and
sends it to /v1/completions as a token-id prompt. Requests are sent one at a time, so
each prompt is prefilled in a single step. Run it while the server is idle.

Only the Python standard library is needed.

  python3 warmup_variants.py --dry-run          # list the variants, send nothing
  python3 warmup_variants.py                    # warm up http://localhost:8000
  python3 warmup_variants.py --verify 12        # warm up, then check random lengths
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

# (largest T, tile) pairs, as in expert_plan() of vLLM's ops/trtllm_bmm.py
# (vLLM nightly 647a768). Update these if a later vLLM changes the plan.
TILES = ((8, 8), (128, 16), (384, 32), (768, 64), (4096, 128), (1 << 31, 256))


def representative_lengths(budget, num_experts, top_k):
    """One prompt length per variant key, in ascending order."""
    rep = {}
    for t in range(1, budget + 1):
        tile = next(x for m, x in TILES if t <= m)
        rows = top_k * t
        e = min(num_experts, rows)
        max_ctas = e + (rows - e) // tile
        # Triton specializes the integer argument `tokens` on "== 1" and "multiple of
        # 16"; from T >= 2048 k_group switches to the TMA path.
        rep.setdefault((tile, max_ctas, t == 1, t % 16 == 0, t >= 2048), t)
    return sorted(rep.values())


def make_prompt(length, salt, vocab_start, vocab_span):
    """Pseudo-random ordinary token ids. The salt keeps prompts distinct between
    requests, so that prefix caching (if enabled) cannot skip the prefill."""
    return [vocab_start + (j * 7919 + length + salt) % vocab_span for j in range(length)]


def send(args, length, salt):
    """Send one request and return its wall time in seconds."""
    body = {
        "model": args.model,
        "prompt": make_prompt(length, salt, args.vocab_start, args.vocab_span),
        "max_tokens": 1,
        "temperature": 0,
    }
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    req = urllib.request.Request(args.base_url.rstrip("/") + "/v1/completions",
                                 data=json.dumps(body).encode(), headers=headers)
    start = time.time()
    with urllib.request.urlopen(req, timeout=args.timeout) as resp:
        resp.read()
    return time.time() - start


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000",
                    help="server root, without /v1 (default: %(default)s)")
    ap.add_argument("--model", default="inkling-2-mini",
                    help="served model name (default: %(default)s)")
    ap.add_argument("--max-num-batched-tokens", type=int, default=2048,
                    help="the server's max_num_batched_tokens; 2048 is vLLM's default on "
                         "GB10 (default: %(default)s)")
    ap.add_argument("--num-experts", type=int, default=128,
                    help="experts per MoE layer (default: %(default)s)")
    ap.add_argument("--top-k", type=int, default=8,
                    help="experts selected per token (default: %(default)s)")
    ap.add_argument("--vocab-start", type=int, default=1000,
                    help="lowest token id to use (default: %(default)s)")
    ap.add_argument("--vocab-span", type=int, default=150000,
                    help="how many token ids to draw from; keep special tokens out of "
                         "this range (default: %(default)s)")
    ap.add_argument("--timeout", type=float, default=600,
                    help="per-request timeout in seconds (default: %(default)s)")
    ap.add_argument("--slow-threshold", type=float, default=1.0,
                    help="count requests slower than this many seconds as likely "
                         "compilations (default: %(default)s)")
    ap.add_argument("--verify", type=int, default=0, metavar="N",
                    help="afterwards, send N random lengths twice each and compare the "
                         "first and second time")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed for the prompt salts and the --verify lengths "
                         "(default: random)")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""),
                    help="bearer token, if the server needs one (default: $OPENAI_API_KEY)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the number of variants and the lengths, send nothing")
    args = ap.parse_args()
    if args.max_num_batched_tokens < 1:
        ap.error("--max-num-batched-tokens must be >= 1")

    lengths = representative_lengths(args.max_num_batched_tokens, args.num_experts, args.top_k)
    print(f"variants: {len(lengths)} (T = 1..{args.max_num_batched_tokens})", flush=True)
    if args.dry_run:
        print("lengths:", " ".join(map(str, lengths)))
        return 0

    rng = random.Random(args.seed)
    t0 = time.time()
    slow = 0
    try:
        for i, length in enumerate(lengths):
            dt = send(args, length, rng.randrange(args.vocab_span))
            slow += dt > args.slow_threshold
            if i % 25 == 0 or i == len(lengths) - 1:
                print(f"{i + 1}/{len(lengths)} T={length} {dt:.2f}s "
                      f"elapsed={time.time() - t0:.0f}s slow={slow}", flush=True)
        print(f"done in {time.time() - t0:.0f}s, requests slower than "
              f"{args.slow_threshold:g}s: {slow}")
        for _ in range(args.verify):
            length = rng.randint(1, args.max_num_batched_tokens)
            first = send(args, length, rng.randrange(args.vocab_span))
            second = send(args, length, rng.randrange(args.vocab_span))
            print(f"verify T={length}: first {first:.2f}s, second {second:.2f}s", flush=True)
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode(errors='replace')[:500]}", file=sys.stderr)
        return 1
    except (urllib.error.URLError, OSError) as e:
        print(f"request failed: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
