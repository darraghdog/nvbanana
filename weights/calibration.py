"""Rebuild the ARC-16 training prompts used for GPTQ calibration."""

import argparse
from pathlib import Path
import random

from datasets import Dataset
import numpy as np

from training.prepare_model import read_json, sha256, write_json


SEED = 7
COUNT = 512
MIN_LENGTH = 256
MAX_LENGTH = 6144
TRAIN_IDS = Path(__file__).resolve().parents[1] / "sdg/dataset/arc_agi2_train_ids.txt"
TRAIN_INPUTS = {
    "challenges": "779eaba89790ebad9af02514a7efc0aefaf2cf8236f046a31bbf8b9ec48f20f5",
    "solutions": "9f07a38bd25af5e83aa5bf85c5cb1a1fefdb30f6a755256fa65429e697ca97f9",
}


def grid_message(grid, role, palette):
    ids = [14, role]
    for row in grid:
        ids.append(10)
        ids.extend(int(palette[cell]) for cell in row)
    ids.append(15)
    return ids


def calibration_rows(challenges, solutions):
    keys = sorted(challenges)
    rng = random.Random(SEED)
    rng.shuffle(keys)
    rows = []
    for key in keys:
        task = challenges[key]
        augment = np.random.RandomState(rng.randrange(1 << 20))
        palette = augment.permutation(10)
        # The original augmenter drew palettes for all eight orientations
        # of every test query before shuffling the first variant's demos.
        for _ in range(8 * len(task["test"]) - 1):
            augment.permutation(10)
        ids = []
        for index in augment.permutation(len(task["train"])):
            demo = task["train"][index]
            ids.extend(grid_message(demo["input"], 11, palette))
            ids.extend(grid_message(demo["output"], 12, palette))
        ids.extend(grid_message(task["test"][0]["input"], 11, palette))
        ids.extend(grid_message(solutions[key][0], 12, palette))
        ids = ids[:MAX_LENGTH]
        if len(ids) >= MIN_LENGTH:
            rows.append({"input_ids": ids, "attention_mask": [1] * len(ids), "key": key})
        if len(rows) == COUNT:
            return rows
    raise ValueError(f"Found only {len(rows)} qualifying ARC training prompts")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenges", type=Path, required=True)
    parser.add_argument("--solutions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    for name, path in (("challenges", args.challenges), ("solutions", args.solutions)):
        if sha256(path) != TRAIN_INPUTS[name]:
            raise ValueError(f"Expected the original Kaggle ARC-AGI-2 training {name} JSON")
    challenges = read_json(args.challenges)
    solutions = read_json(args.solutions)
    allowed = set(TRAIN_IDS.read_text().splitlines())
    if set(challenges) != allowed or set(solutions) != allowed:
        raise ValueError("ARC task IDs differ from the training allowlist")
    rows = calibration_rows(challenges, solutions)
    Dataset.from_list(rows).save_to_disk(str(args.output))
    lengths = [len(row["input_ids"]) for row in rows]
    write_json(args.output / "CALIBRATION_STATS.json", {
        "n": len(rows),
        "min_tokens": min(lengths),
        "mean_tokens": sum(lengths) / len(lengths),
        "max_tokens": max(lengths),
        "seed": SEED,
        "training_sha256": TRAIN_INPUTS,
    })


if __name__ == "__main__":
    main()
