"""Pair book specs with ARC training summaries and generate mixed specs."""

import argparse
import json
import os
from pathlib import Path
import random
import urllib.request


HERE = Path(__file__).resolve().parent
BOOK_SPECS = HERE.parent / "sourcing" / "inspiration_outputs.jsonl"
TRAINING_SUMMARIES = HERE / "training_summaries.jsonl"
FIELDS = (
    "rules_summary",
    "grid_constraints",
    "input_generation",
    "output_relationship",
    "solution_steps",
    "key_insight",
    "puzzle_concepts",
)


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def prepare(args):
    books = sorted(read_jsonl(BOOK_SPECS), key=lambda row: row["uid"])
    anchors = sorted(read_jsonl(TRAINING_SUMMARIES), key=lambda row: row["task_id"])
    if not 1 <= args.count <= len(books):
        raise ValueError(f"--count must be between 1 and {len(books)}")

    rng = random.Random(args.seed)
    books = rng.sample(books, args.count)
    rng.shuffle(anchors)
    prompt = (HERE / "mix_prompt.txt").read_text(encoding="utf-8").strip()
    for pair_idx, book in enumerate(books):
        anchor = anchors[pair_idx % len(anchors)]
        inputs = {
            "PUZZLE1": anchor["stg1_summary"],
            "PUZZLE2": book["stg1_summary"],
        }
        print(json.dumps({
            "pair_idx": pair_idx,
            "base_puzzle": anchor["task_id"],
            "inspiration_puzzle": book["uid"],
            "inspiration_source_url": book["source_url"],
            "inspiration_scan_page": book["page_start"],
            "style_anchor_split": "arc_agi2_train",
            "eval_anchor_used": False,
            **inputs,
            "prompt": prompt.format(**inputs),
        }, ensure_ascii=False))


def generate(args):
    api_base = os.environ.get("SDG_API_BASE", "").rstrip("/")
    if not api_base.startswith(("https://", "http://")):
        raise ValueError("Set SDG_API_BASE to your model server's API base URL")
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("SDG_API_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    training = {row["task_id"]: row["stg1_summary"] for row in read_jsonl(TRAINING_SUMMARIES)}
    inspirations = {row["uid"]: row for row in read_jsonl(BOOK_SPECS)}
    prompt = (HERE / "mix_prompt.txt").read_text(encoding="utf-8").strip()
    pairs = list(read_jsonl(args.pairs))
    if not pairs:
        raise ValueError("The pair file is empty")
    for pair in pairs:
        base = training.get(pair["base_puzzle"])
        book = inspirations.get(pair["inspiration_puzzle"])
        if (
            not base or base != pair["PUZZLE1"]
            or not book or book["stg1_summary"] != pair["PUZZLE2"]
            or pair["style_anchor_split"] != "arc_agi2_train"
            or pair["eval_anchor_used"] is not False
            or pair["prompt"] != prompt.format(PUZZLE1=base, PUZZLE2=book["stg1_summary"])
        ):
            raise ValueError(f"Pair {pair['pair_idx']} does not match the bundled inputs")

    for pair in pairs:
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": pair["prompt"]}],
        }
        request = urllib.request.Request(
            f"{api_base}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            completion = json.load(response)
        content = completion["choices"][0]["message"]["content"]
        try:
            spec = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            spec = None
        valid = isinstance(spec, dict) and set(spec) == set(FIELDS) and all(
            isinstance(value, str) and value.strip() for value in spec.values()
        )
        source = inspirations[pair["inspiration_puzzle"]]
        result = {
            "pair_idx": pair["pair_idx"],
            "base_puzzle": pair["base_puzzle"],
            "inspiration_puzzle": pair["inspiration_puzzle"],
            "inspiration_source_url": source["source_url"],
            "inspiration_scan_page": source["page_start"],
            "style_anchor_split": pair["style_anchor_split"],
            "eval_anchor_used": pair["eval_anchor_used"],
            "model": args.model,
            "mixed_spec": spec if valid else None,
        }
        if not valid:
            result["raw_response"] = content
        print(json.dumps(result, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="write pair prompts as JSONL")
    prepare_parser.add_argument("--count", type=int, default=12)
    prepare_parser.add_argument("--seed", type=int, default=2026)
    generate_parser = commands.add_parser("generate", help="write mixed specs as JSONL")
    generate_parser.add_argument("pairs", type=Path)
    generate_parser.add_argument("--model", required=True)
    args = parser.parse_args()

    try:
        if args.command == "prepare":
            prepare(args)
        else:
            generate(args)
    except (KeyError, IndexError, ValueError, OSError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
