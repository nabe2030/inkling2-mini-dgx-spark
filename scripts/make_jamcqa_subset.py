#!/usr/bin/env python3
"""Pick the 300 JamC-QA test questions used in the README.

Seats are allocated to categories in proportion to their size (largest remainder, ties
broken by category name). Within a category the sorted qids are sampled with
random.Random(seed), categories being processed in name order. The output lists the
chosen qids in test-split order.

  pip install datasets
  python3 make_jamcqa_subset.py --output jamcqa_subset300_seed20261123.txt
"""
import argparse
import collections
import random


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default="jamcqa_subset300_seed20261123.txt")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261123)
    ap.add_argument("--revision", default=None,
                    help="dataset revision; ours was ec2f8a34b1cfb9dcd8c50583eb688ef2ee3fb671")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = list(load_dataset("sbintuitions/JamC-QA", "v1.0", split="test", revision=args.revision))
    by = collections.defaultdict(list)
    for r in ds:
        by[r["category"]].append(r["qid"])
    total = len(ds)
    quota = {c: args.n * len(v) / total for c, v in by.items()}
    alloc = {c: int(q) for c, q in quota.items()}
    rest = args.n - sum(alloc.values())
    for c in sorted(quota, key=lambda c: (-(quota[c] - alloc[c]), c))[:rest]:
        alloc[c] += 1
    rng = random.Random(args.seed)
    selected = []
    for c in sorted(by):
        selected += rng.sample(sorted(by[c]), alloc[c])
    order = {r["qid"]: i for i, r in enumerate(ds)}
    selected.sort(key=order.get)
    with open(args.output, "w") as f:
        f.write("\n".join(selected) + "\n")
    print(f"{len(selected)} of {total} questions written to {args.output}")


if __name__ == "__main__":
    main()
