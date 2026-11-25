# Data

## `speed/`

Throughput runs, DSpark off (`off/`) and on (`on/`), at 1, 2, 4, 8, 16 and 32 concurrent requests (200 ShareGPT prompts each).

- `inkling2mini-{off,on}-c{N}.json`: the result file of `vllm bench serve`, unmodified. `date` is in UTC.
- `metrics_{before,after}_c{N}.txt`: selected counters from the server's `/metrics` before and after each point (prefix cache, preemptions, speculative decoding, token totals).
- `summary_off.md`, `summary_on.md`: tables built from the files above with `scripts/summarize_speed.py`.

The ShareGPT dataset itself is not included.

## `jamcqa/`

- `jamcqa_subset300_seed20261123.txt`: the question ids of the 300-question subset, in test-split order (sha256 `2dd41f9104faf500ed81cbb93d0e4f9743916a0fca451f9962ba54be6a39dff8`). Made with `scripts/make_jamcqa_subset.py` from revision `ec2f8a34b1cfb9dcd8c50583eb688ef2ee3fb671`.
- `*.csv`: per-question results of Inkling 2 Mini, one row per question. Question text, choices and replies are not included.
- `*.summary.json`: totals, per-category accuracy and settings of the same run.

| Run | Reasoning | Temperature | max_tokens | DSpark | Correct | Truncated | Use |
|---|---|---:|---:|---|---:|---:|---|
| `direct` | off (workaround in the README) | 0 | 8 | off | 137 | 2 | headline number (45.67%) |
| `direct_shuffle42` | off | 0 | 8 | off | 139 | 3 | choices shuffled, position-bias check |
| `reasoning_temp1.0` | on, effort "none" | 1.0 | 8192 | off | 164 | 1 | reference (54.67%) |
| `reasoning_temp0_dspark-off_INVALID` | on, effort "none" | 0 | 4096 | off | 134 | 73 | not scored (truncations) |
| `reasoning_temp0_dspark-on_INVALID` | on, effort "none" | 0 | 4096 | on | 138 | 75 | not scored (truncations) |

CSV columns: `qid`, `category`, `gold` (index of the correct choice, as shown in the prompt; the shuffled position for `direct_shuffle42`), `pred` (-1 if no index was found), `correct`, `finish_reason`, `completion_tokens`, `reasoning_tokens`, `prompt_tokens`, `latency_s`. Runs used 16 concurrent requests, so `latency_s` is not a single-request latency.

## License of the JamC-QA-derived data

[JamC-QA](https://huggingface.co/datasets/sbintuitions/JamC-QA) is published by SB Intuitions under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The question ids, categories and correct-choice indices in `jamcqa/` come from it (changes: a 300-question selection, and the choice order shuffled in `direct_shuffle42`), so the files in `jamcqa/` are provided under CC BY-SA 4.0. Everything else in this repository is under Apache-2.0.
