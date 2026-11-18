"""Validate generated code and keep puzzles with three agreeing output programs."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

from sdg.mixing.mix_puzzles import BOOK_SPECS, FIELDS, TRAINING_SUMMARIES, read_jsonl


PROVENANCE = (
    "pair_idx",
    "base_puzzle",
    "inspiration_puzzle",
    "inspiration_source_url",
    "inspiration_scan_page",
    "style_anchor_split",
    "eval_anchor_used",
)
GRID_SEEDS = range(1000, 1030)
OUTPUT_SEEDS = range(3)
CODE_BLOCK = r"```python[ \t]*\n(.*?)\n```"
WORKER = r"""
import contextlib
import copy
import io
import json
import sys

request = json.load(sys.stdin)
namespace = {}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(request["code"], "<generated>", "exec"), namespace)
    function = namespace[request["function"]]
    inputs = request["inputs"]
    outputs = [function(copy.deepcopy(item)) for item in inputs]
    # Check repeatability without letting a function mutate its first result.
    outputs = copy.deepcopy(outputs)
    repeated = [function(copy.deepcopy(item)) for item in reversed(inputs)]
    if outputs != repeated[::-1]:
        raise ValueError("non-deterministic results")
print(json.dumps(outputs))
"""


def read_by_pair(path):
    indexed = {}
    for row in read_jsonl(path):
        pair_idx = row["pair_idx"]
        if type(pair_idx) is not int or pair_idx < 0 or pair_idx in indexed:
            raise ValueError(f"Invalid or duplicate pair_idx {pair_idx!r} in {path}")
        indexed[pair_idx] = row
    return indexed


def final_answer(text):
    # Text completions may retain the Harmony channel tokens.
    answer = text.rsplit("<|channel|>final", 1)[-1].removeprefix("<|message|>").strip()
    return re.sub(r"(?:<\|(?:return|end)\|>\s*)+$", "", answer).strip()


def grid_key(grid):
    if not isinstance(grid, list) or not 1 <= len(grid) <= 30:
        raise ValueError("Invalid grid height")
    if not isinstance(grid[0], list) or not 1 <= len(grid[0]) <= 30:
        raise ValueError("Invalid grid width")
    for row in grid:
        if not isinstance(row, list) or len(row) != len(grid[0]):
            raise ValueError("Grid must be rectangular")
        if any(type(cell) is not int or not 0 <= cell <= 9 for cell in row):
            raise ValueError("Grid colours must be integers in [0, 9]")
    return tuple(tuple(row) for row in grid)


def run_generated(code, function, inputs):
    # Run this script in the read-only, networkless container in the guide.
    request = json.dumps({"code": code, "function": function, "inputs": inputs})
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-c", WORKER],
            input=request,
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"{function} timed out") from exc
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"{function} failed in the validation process") from exc
    grids = json.loads(result.stdout)
    if not isinstance(grids, list) or len(grids) != len(inputs):
        raise ValueError("Wrong number of grids")
    for grid in grids:
        grid_key(grid)
    return grids


def prepare_stg4(path):
    anchors = {row["task_id"]: row["stg1_summary"] for row in read_jsonl(TRAINING_SUMMARIES)}
    books = {row["uid"]: row for row in read_jsonl(BOOK_SPECS)}
    for pair, row in sorted(read_by_pair(path).items()):
        try:
            anchor = anchors[row["base_puzzle"]]
            book = books[row["inspiration_puzzle"]]
            if (
                row["PUZZLE1"] != anchor
                or row["PUZZLE2"] != book["stg1_summary"]
                or row["inspiration_source_url"] != book["source_url"]
                or row["inspiration_scan_page"] != book["page_start"]
                or row["style_anchor_split"] != "arc_agi2_train"
                or row["eval_anchor_used"] is not False
            ):
                raise ValueError("Source fields do not match bundled inputs")
            final = final_answer(row["generation"])
            match = re.fullmatch(r"<mixed_spec>\s*(.*?)\s*</mixed_spec>\s*" + CODE_BLOCK, final, re.DOTALL)
            if match is None:
                raise ValueError("Expected mixed_spec followed by one Python block")
            puzzle, input_code = match.groups()
            if any(not re.search(rf"<{tag}>\s*\S.*?</{tag}>", puzzle, re.DOTALL) for tag in FIELDS):
                raise ValueError("Incomplete mixed spec")
            grids = run_generated(input_code, "generate_puzzle_input", list(GRID_SEEDS))
            if len({grid_key(grid) for grid in grids}) < 10:
                raise ValueError("Fewer than 10 unique inputs")
        except (KeyError, TypeError, ValueError) as exc:
            print(f"Stage 2/3 pair {pair} rejected: {exc}", file=sys.stderr)
            continue
        print(json.dumps({
            **{key: row[key] for key in PROVENANCE},
            "PUZZLE": puzzle,
            "INPUT_CODE": input_code,
            "validation_grids": grids,
        }, ensure_ascii=False))


def filter_stg4(inputs_path, samples_dir):
    inputs = read_by_pair(inputs_path)
    runs = [read_by_pair(samples_dir / f"output-rs{seed}.jsonl") for seed in OUTPUT_SEEDS]
    for pair, row in sorted(inputs.items()):
        codes = []
        outputs_by_run = []
        try:
            grids = row["validation_grids"]
            if len(grids) != len(GRID_SEEDS) or len({grid_key(grid) for grid in grids}) < 10:
                raise ValueError("Incomplete validation grids")
            for seed, run in zip(OUTPUT_SEEDS, runs):
                candidate = run.get(pair)
                if candidate is None:
                    raise ValueError(f"Missing output sample for seed {seed}")
                if any(candidate[key] != row[key] for key in (*PROVENANCE, "PUZZLE", "INPUT_CODE", "validation_grids")):
                    raise ValueError("Stale Stage 4 result")
                match = re.fullmatch(CODE_BLOCK, final_answer(candidate["generation"]), re.DOTALL)
                if match is None:
                    raise ValueError("Expected one standalone Python block")
                code = match.group(1)
                outputs_by_run.append(run_generated(code, "generate_puzzle_output", grids))
                codes.append(code)
            outputs = outputs_by_run[0]
            if any(other != outputs for other in outputs_by_run[1:]):
                raise ValueError("Output programs disagree")
            if len({grid_key(grid) for grid in outputs}) < 10:
                raise ValueError("Fewer than 10 unique outputs")
            if all(input_grid == output_grid for input_grid, output_grid in zip(grids, outputs)):
                raise ValueError("Identity transformation")
        except (KeyError, TypeError, ValueError) as exc:
            print(f"Stage 4 pair {pair} rejected: {exc}", file=sys.stderr)
            continue

        grid_pairs = [
            {"input": input_grid, "output": output_grid}
            for input_grid, output_grid in zip(grids, outputs)
        ]
        examples = {}
        for grid_pair in grid_pairs:
            examples.setdefault(grid_key(grid_pair["input"]), grid_pair)
        examples = list(examples.values())
        print(json.dumps({
            **{key: row[key] for key in PROVENANCE},
            "spec": row["PUZZLE"],
            "input_code": row["INPUT_CODE"],
            "output_codes": codes,
            "stage4_seeds": list(OUTPUT_SEEDS),
            "consensus_grids": len(grids),
            "grid_pairs": grid_pairs,
            "task": {"train": examples[:3], "test": examples[3:4]},
        }, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-stg4")
    prepare.add_argument("stg23", type=Path)
    filter_stage = commands.add_parser("filter-stg4")
    filter_stage.add_argument("stg4_input", type=Path)
    filter_stage.add_argument("stg4_output_dir", type=Path)
    args = parser.parse_args()
    if args.command == "prepare-stg4":
        prepare_stg4(args.stg23)
    else:
        filter_stg4(args.stg4_input, args.stg4_output_dir)


if __name__ == "__main__":
    main()
