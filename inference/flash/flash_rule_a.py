#!/usr/bin/env python3
"""Rule-A Flash driver: Flash attempts on a base model's least-sure tasks, requeued per task until settled.

Runs in the Flash runtime Python with the harness on PYTHONPATH (cwd = harness dir), started by flash_kaggle_run.py --pools.
The setup copies the harness's own main() (same tokenizer, settings, rows, sandbox core); each attempt is the harness's
run_attempt(row, k) and its records are written with pair_records, exactly as the harness CLI does, so seeds and records match a
normal run's sample k.

Queue: FIFO of (task, k). It starts with attempt 0 of every task in --keys order (least sure first); when an attempt ends, the task
is settled if every test input has a verified answer (check() passed) that equals any base candidate in --pools or that Flash gave
--repeat-stop times; otherwise (task, k+1) goes to the back of the queue. Nothing new starts after --deadline; the launcher kills
the process at the deadline, so attempts still running then are dropped. Standard library + the harness only.

usage (from flash_kaggle_run.py): python flash_rule_a.py --base-url ... --tokenizer ... --data rows.jsonl --out shard0.jsonl
    --setting S --seats 28 --pools pools.json --keys keys.json --deadline <unix> --python <sandbox python> --server-tokenizer M
"""
import argparse
import collections
import json
import os
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from scripts.grpo_arc2_code import image_multiturn_eval as H


def norm(text):
    """An answer grid as 'row\\nrow' of digits only (None if empty)."""
    if text is None:
        return None
    rows = ["".join(ch for ch in r if ch.isdigit()) for r in str(text).strip().split("\n")]
    rows = [r for r in rows if r]
    return "\n".join(rows) or None


def main():
    ap = argparse.ArgumentParser()
    for a in ("--base-url", "--tokenizer", "--data", "--out", "--setting", "--pools", "--keys"):
        ap.add_argument(a, required=True)
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--seats", type=int, default=28)
    ap.add_argument("--sandbox-workers", type=int, default=24)
    ap.add_argument("--python", default=None)
    ap.add_argument("--server-tokenizer", default=None)
    ap.add_argument("--deadline", type=float, default=0.0)
    ap.add_argument("--max-attempts", type=int, default=16)
    ap.add_argument("--repeat-stop", type=int, default=3)
    a = ap.parse_args()
    # the harness's own argument names/defaults for what run_attempt reads (image_multiturn_eval.main)
    args = argparse.Namespace(base_url=a.base_url, model=a.model, tokenizer=a.tokenizer, data=a.data, out=a.out, setting=a.setting,
                              shard=0, nshards=1, samples=a.max_attempts, concurrency=a.seats, sandbox_workers=a.sandbox_workers,
                              python=a.python, temperature=1.0, top_p=0.95, top_k=20, timeout=3 * 3600, limit=0, keys=a.keys,
                              save_messages=False, skip_done=None, require_source=None, probe=False, server_tokenizer=a.server_tokenizer)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    settings = json.load(open(os.path.join(os.path.dirname(os.path.abspath(H.__file__)), "direct_eval_settings.json")))
    setting = settings[args.setting]
    if setting.get("token_exact"):
        args.server_tok = AutoTokenizer.from_pretrained(args.server_tokenizer) if args.server_tokenizer else tok
    order = json.load(open(a.keys))
    order = order["keys"] if isinstance(order, dict) else order
    rows = [json.loads(l) for l in open(args.data)]
    rows = [r for r in rows if r["task_id"] in set(order)]
    rows = H.select_rows(rows, setting)
    by_task = {}
    for r in rows:
        by_task.setdefault(r["task_id"], []).append(r)
    core = H.ArcCodeCore({"num_workers": args.sandbox_workers, "timeout_s": setting.get("timeout_s", 10), "out_chars": setting.get("out_chars", 1000), "mem_mb": 6144,
                          "drop_timed_out": setting.get("drop_timed_out", False),
                          "parse_tool_calls": setting.get("parse_tool_calls", False),
                          **({"extra_imports": H.HYGIENE_IMPORTS, "recursion_limit": H.HYGIENE_RECURSION} if setting.get("sandbox_hygiene") else {}),
                          "max_runs": setting.get("max_runs", 24), "observation_style": "chat", "enable_thinking": True,
                          **({"python": args.python} if args.python else {})})
    pools = {k: {norm("\n".join("".join(str(int(c)) for c in row) for row in g)) for g, _ in v} for k, v in json.load(open(a.pools)).items()}
    ntest = {t: sum(len(r.get("pair_ids") or [H.row_id(r)]) for r in by_task[t]) for t in by_task}
    votes = collections.defaultdict(collections.Counter)      # "task_i" -> verified answer -> count
    settled, attempts = set(), collections.Counter()
    lock, out = threading.Lock(), open(args.out, "a")
    t0, n_done = time.time(), [0]

    def run(task, k):
        recs = []
        for row in by_task[task]:      # a task with <= 2 test inputs is one per_task row (one conversation)
            try:
                rec = H.run_attempt(args, tok, core, setting, row, k)
            except Exception as e:     # one attempt must never sink the run
                rec = {"puzzle_id": H.row_id(row), "sample": k, "setting": setting["name"], "status": f"error: {type(e).__name__}: {e}"[:200],
                       "gen_tokens": 0, "correct": None, "answered": False}
            recs += H.pair_records(rec, row)
        return task, k, recs

    queue = collections.deque((t, 0) for t in order if t in by_task)
    running = set()
    with ThreadPoolExecutor(max_workers=a.seats) as ex:
        while queue or running:
            while queue and len(running) < a.seats and (not a.deadline or time.time() < a.deadline):
                running.add(ex.submit(run, *queue.popleft()))
            if not running:
                break
            done, _ = wait(running, timeout=30, return_when=FIRST_COMPLETED)
            for f in done:
                running.discard(f)
                task, k, recs = f.result()
                attempts[task] += 1
                with lock:
                    for rec in recs:
                        out.write(json.dumps(rec) + "\n")
                        tid, idx = rec["puzzle_id"].split("#")
                        g = norm(rec.get("answer")) if rec.get("answered") and rec.get("auto_answer") else None
                        if g is not None:
                            key = f"{tid}_{idx}"
                            votes[key][g] += 1
                            if g in pools.get(key, set()) or votes[key][g] >= a.repeat_stop:
                                settled.add(key)
                    out.flush()
                n_done[0] += 1
                fin = all(f"{task}_{i}" in settled for i in range(ntest[task]))
                print(f"[{n_done[0]}] {task} attempt {k} {'SETTLED' if fin else 'open'} | elapsed {(time.time()-t0)/60:.1f} min | "
                      + " ".join(f"{r['puzzle_id']}:{r['status']}:{'auto' if r.get('auto_answer') else '-'}" for r in recs), flush=True)
                if not fin and k + 1 < a.max_attempts and (not a.deadline or time.time() < a.deadline):
                    queue.append((task, k + 1))
            if a.deadline and time.time() >= a.deadline and not queue:
                break
    out.close()
    print(f"RULE_A_DONE tasks {len(by_task)} attempts {sum(attempts.values())} settled tasks "
          f"{sum(all(f'{t}_{i}' in settled for i in range(ntest[t])) for t in by_task)} in {(time.time()-t0)/60:.1f} min", flush=True)
    core.shutdown()


if __name__ == "__main__":
    main()
