#!/usr/bin/env python3
"""JamC-QA 4-choice harness for an OpenAI-compatible server, on a fixed list of questions.

3-shot (the first 3 dev questions), exact match on the choice index (the first 0-3 found
in the reply). The prompt is in Japanese, like the dataset, and is part of the measurement
conditions, so it is kept as is.

  --qid-file      question ids to run, one per line; they run in test-split order
  --extra-body    JSON merged into each request, e.g. {"reasoning_effort":"none"}
  --workers       concurrent requests (results are saved in dataset order)
  --force-direct  skip reasoning: render the chat template with /tokenize (extra-body is
                  passed as chat_template_kwargs), append the tokens that start an answer
                  in the few-shot turns, <|message_model|><|content_text|> (Inkling 2 token
                  ids 200001, 200004), and generate with /v1/completions, stopping at
                  <|end_message|> (200010)
  --save-text     also save the question and the reply/reasoning text

Set --max-tokens large enough: cut-off replies are counted as "truncated", not retried.

  pip install datasets httpx
  # headline number in the README (no reasoning)
  python3 jamcqa_bench_subset.py --qid-file ../data/jamcqa/jamcqa_subset300_seed20261123.txt \\
    --force-direct --extra-body '{"reasoning_effort":"none"}' --max-tokens 8 --workers 16 --output direct.json
  # reference (reasoning on, temperature 1.0)
  python3 jamcqa_bench_subset.py --qid-file ../data/jamcqa/jamcqa_subset300_seed20261123.txt \\
    --extra-body '{"reasoning_effort":"none"}' --max-tokens 8192 --temperature 1.0 --workers 16 --output reasoning.json
"""
import argparse, json, time, collections
from concurrent.futures import ThreadPoolExecutor

def build_prompt(question, choices):
    lines = [f"質問: {question}"]
    for i, c in enumerate(choices):
        lines.append(f"{i}: {c}")
    lines.append("回答（番号のみ）:")
    return "\n".join(lines)

def build_messages(question, choices, few_shot):
    messages = [{"role": "system", "content": "あなたは日本固有の知識問題に答えるアシスタントです。選択肢の番号のみを回答してください。"}]
    for ex in few_shot:
        messages.append({"role": "user", "content": build_prompt(ex["question"], ex["choices"])})
        messages.append({"role": "assistant", "content": str(ex["answer"])})
    messages.append({"role": "user", "content": build_prompt(question, choices)})
    return messages

def extract_answer(text):
    if not text:
        return -1
    for ch in text.strip():
        if ch in "0123":
            return int(ch)
    return -1

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-url", default="http://localhost:8000/v1/chat/completions")
    ap.add_argument("--model", default="inkling-2-mini")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--few-shot-n", type=int, default=3)
    ap.add_argument("--qid-file", required=True)
    ap.add_argument("--extra-body", default="{}")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--output", required=True)
    ap.add_argument("--force-direct", action="store_true")
    ap.add_argument("--shuffle-choices-seed", type=int, default=None,
                    help="shuffle the choices of each question with this seed (position-bias check)")
    ap.add_argument("--save-text", action="store_true")
    a = ap.parse_args()
    extra = json.loads(a.extra_body)
    import httpx
    from datasets import load_dataset
    ds = load_dataset("sbintuitions/JamC-QA", "v1.0", split="test")
    dev = load_dataset("sbintuitions/JamC-QA", "v1.0", split="dev")
    few = [{"question": e["question"], "choices": [e[f"choice{j}"] for j in range(4)], "answer": e["answer_index"]}
           for e in list(dev)[:a.few_shot_n]]
    want = [l.strip() for l in open(a.qid_file) if l.strip()]
    ws = set(want)
    items = [r for r in ds if r["qid"] in ws]
    assert len(items) == len(ws), (len(items), len(ws))
    client = httpx.Client(timeout=900.0)

    def perm_of(item):
        if a.shuffle_choices_seed is None:
            return [0, 1, 2, 3]
        import random
        rng = random.Random(f"{a.shuffle_choices_seed}:{item['qid']}")
        perm = list(range(4)); rng.shuffle(perm)
        return perm  # perm[new_pos] = old_pos

    def one(item):
        perm = perm_of(item)
        choices = [item[f"choice{j}"] for j in perm]
        body = {"model": a.model, "messages": build_messages(item["question"], choices, few),
                "max_tokens": a.max_tokens, "temperature": a.temperature, **extra}
        t0 = time.time()
        try:
            if a.force_direct:
                base = a.api_url.rsplit("/v1/", 1)[0]
                tk = client.post(base + "/tokenize", json={"model": a.model, "messages": body["messages"],
                                 "chat_template_kwargs": extra, "add_generation_prompt": False}).json()
                ids = tk["tokens"] + [200001, 200004]
                d = client.post(base + "/v1/completions", json={"model": a.model, "prompt": ids,
                                "max_tokens": a.max_tokens, "temperature": a.temperature,
                                "stop_token_ids": [200010], "skip_special_tokens": False}).json()
                d["choices"][0]["message"] = {"content": d["choices"][0].get("text", "")}
            else:
                d = client.post(a.api_url, json=body).json()
            ch = d["choices"][0]; msg = ch["message"]
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
            u = d.get("usage") or {}
            ctd = u.get("completion_tokens_details") or {}
            rec = dict(finish=ch.get("finish_reason", ""), content=content, reasoning=reasoning,
                       completion_tokens=u.get("completion_tokens"), reasoning_tokens=ctd.get("reasoning_tokens"),
                       prompt_tokens=u.get("prompt_tokens"), error="")
        except Exception as e:
            rec = dict(finish="error", content="", reasoning="", completion_tokens=None,
                       reasoning_tokens=None, prompt_tokens=None, error=str(e)[:300])
        rec["latency"] = round(time.time() - t0, 3)
        return rec

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        outs = list(ex.map(one, items))
    wall = time.time() - t0
    results = []; correct = parse_fail = truncated = errors = 0
    cat = collections.defaultdict(lambda: [0, 0]); pos = collections.Counter()
    for item, r in zip(items, outs):
        gold = perm_of(item).index(item["answer_index"]); pred = extract_answer(r["content"])
        ok = pred == gold
        correct += ok; parse_fail += pred == -1; truncated += r["finish"] == "length"; errors += r["finish"] == "error"
        cat[item["category"]][0] += ok; cat[item["category"]][1] += 1; pos[pred] += 1
        results.append({"qid": item["qid"], "category": item["category"],
                        "gold": gold, "pred": pred, "correct": ok, "reasoning_len_chars": len(r["reasoning"]),
                        "finish_reason": r["finish"], "completion_tokens": r["completion_tokens"],
                        "reasoning_tokens": r["reasoning_tokens"], "prompt_tokens": r["prompt_tokens"],
                        "latency": r["latency"], "error": r["error"]})
        if a.save_text:
            results[-1].update(question=item["question"], response=r["content"], reasoning=r["reasoning"])
    n = len(items)
    ct = sorted(x["completion_tokens"] for x in results if x["completion_tokens"] is not None)
    summary = {"model": a.model, "dataset": "sbintuitions/JamC-QA v1.0 test (subset)", "qid_file": a.qid_file,
               "total": n, "correct": correct, "accuracy": round(correct / n * 100, 2),
               "parse_fail": parse_fail, "parse_fail_rate": round(parse_fail / n * 100, 2),
               "truncated": truncated, "truncated_rate": round(truncated / n * 100, 2), "errors": errors,
               "max_tokens": a.max_tokens, "temperature": a.temperature, "extra_body": extra,
               "few_shot_n": a.few_shot_n, "workers": a.workers, "force_direct": a.force_direct, "shuffle_choices_seed": a.shuffle_choices_seed, "wall_sec": round(wall, 1),
               "completion_tokens_median": ct[len(ct) // 2] if ct else None, "completion_tokens_max": ct[-1] if ct else None,
               "pred_distribution": {str(k): v for k, v in sorted(pos.items())},
               "category_breakdown": {c: {"correct": v[0], "total": v[1], "accuracy": round(v[0] / v[1] * 100, 2)} for c, v in sorted(cat.items())},
               "results": results}
    json.dump(summary, open(a.output, "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
