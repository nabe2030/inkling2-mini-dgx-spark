#!/usr/bin/env python3
"""Build the throughput table from `vllm bench serve` result JSON files and the
/metrics snapshots taken before and after each point.

  python3 summarize_speed.py data/speed/off
"""
import glob
import json
import re
import sys


def read_metrics(path):
    m = {}
    for line in open(path):
        key, value = line.rsplit(" ", 1)
        m[re.sub(r"\{.*\}", "", key)] = float(value)
    return m


def main():
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip())
        return 0 if len(sys.argv) == 2 else 2
    d = sys.argv[1]
    print("| c | completed | output tok/s | TTFT median ms | TTFT mean ms | TPOT median ms | TPOT mean ms | output tokens | draft acceptance | mean acceptance length | prefix cache hits | preemptions |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c in (1, 2, 4, 8, 16, 32):
        files = glob.glob(f"{d}/*-c{c}.json")
        if not files:
            continue
        r = json.load(open(files[0]))
        before = read_metrics(f"{d}/metrics_before_c{c}.txt")
        after = read_metrics(f"{d}/metrics_after_c{c}.txt")

        def delta(k):
            return after.get(k, 0) - before.get(k, 0)

        acc = length = "-"
        if "vllm:spec_decode_num_draft_tokens_total" in after:
            draft = delta("vllm:spec_decode_num_draft_tokens_total")
            accepted = delta("vllm:spec_decode_num_accepted_tokens_total")
            drafts = delta("vllm:spec_decode_num_drafts_total")
            acc = f"{accepted / draft * 100:.1f}%" if draft else "-"
            length = f"{1 + accepted / drafts:.2f}" if drafts else "-"
        print(f"| {c} | {r['completed']}/{r['num_prompts']} | {r['output_throughput']:.2f} | "
              f"{r['median_ttft_ms']:.1f} | {r['mean_ttft_ms']:.1f} | {r['median_tpot_ms']:.2f} | "
              f"{r['mean_tpot_ms']:.2f} | {r['total_output_tokens']} | {acc} | {length} | "
              f"{delta('vllm:prefix_cache_hits_total'):.0f} | {delta('vllm:num_preemptions_total'):.0f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
