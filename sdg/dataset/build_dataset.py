"""Build four ARC chat rows per accepted puzzle from its agreed grid pairs."""

import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import random
import re
import tempfile

from sdg.generation.build_puzzles import GRID_SEEDS, OUTPUT_SEEDS, grid_key
from sdg.mixing.mix_puzzles import read_jsonl


TRAIN_IDS = Path(__file__).with_name("arc_agi2_train_ids.txt")
AUGMENTATIONS = 4
DEMONSTRATIONS = 4


def puzzle_id(row):
    pair_idx = row["pair_idx"]
    if type(pair_idx) is not int or pair_idx < 0:
        raise ValueError(f"Invalid pair_idx {pair_idx!r}")
    name = row.get("puzzle_id", f"pair_{pair_idx:06d}")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError(f"Invalid puzzle_id {name!r}")
    return name


def unique_grid_pairs(row, train_ids):
    if (
        row["base_puzzle"] not in train_ids
        or row["style_anchor_split"] != "arc_agi2_train"
        or row["eval_anchor_used"] is not False
    ):
        raise ValueError("Expected an ARC-AGI-2 training anchor and no evaluation lineage")

    pairs = row["grid_pairs"]
    if (
        row["stage4_seeds"] != list(OUTPUT_SEEDS)
        or row["consensus_grids"] != len(GRID_SEEDS)
        or len(pairs) != len(GRID_SEEDS)
    ):
        raise ValueError("Expected three agreeing programs and all 30 Stage 4 grid pairs")

    unique = {}
    for idx, pair in enumerate(pairs):
        input_key = grid_key(pair["input"])
        output_key = grid_key(pair["output"])
        first_idx, first_output = unique.setdefault(input_key, (idx, output_key))
        if first_output != output_key:
            raise ValueError(f"Grid pairs {first_idx} and {idx} assign different outputs to the same input")
    if len(unique) < max(DEMONSTRATIONS + 1, AUGMENTATIONS):
        raise ValueError("Too few distinct inputs for the four augmented rows")
    return [(idx, pairs[idx]) for idx, _ in unique.values()]


def transform_grid(grid, rotations, reflect, palette):
    # Read the original values so a swap never overwrites another source color.
    transformed = [[palette[color] for color in row] for row in grid]
    for _ in range(rotations):
        transformed = [list(column) for column in zip(*transformed[::-1])]
    if reflect:
        transformed = [row[::-1] for row in transformed]
    return transformed


def grid_text(grid):
    return "\n".join("".join(map(str, row)) for row in grid)


def augment_puzzle(name, pairs, seed):
    rng = random.Random(f"{seed}:{name}")
    queries = rng.sample(range(len(pairs)), AUGMENTATIONS)
    symmetries = rng.sample(range(8), AUGMENTATIONS)
    for aug_idx, (query, symmetry) in enumerate(zip(queries, symmetries)):
        demo_pool = [idx for idx in range(len(pairs)) if idx != query]
        demos = rng.sample(demo_pool, DEMONSTRATIONS)
        selected = [pairs[idx] for idx in (*demos, query)]
        rotations, reflect = symmetry % 4, symmetry >= 4
        palette = [0, *rng.sample(range(1, 10), 9)]
        messages = []
        for _, pair in selected:
            for role, key in (("user", "input"), ("assistant", "output")):
                grid = transform_grid(pair[key], rotations, reflect, palette)
                messages.append({"role": role, "content": grid_text(grid)})
        yield (
            {"puzzle_name": f"{name}_aug{aug_idx:02d}", "messages": messages},
            {
                "aug_idx": aug_idx,
                "rotation_clockwise": 90 * rotations,
                "reflect_left_right_after_rotation": reflect,
                "color_map": palette,
                "source_grid_indices": [idx for idx, _ in selected],
            },
        )


def sha256(path):
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            checksum.update(block)
    return checksum.hexdigest()


def build_dataset(args):
    train_ids = set(TRAIN_IDS.read_text(encoding="utf-8").splitlines())
    puzzles = {}
    for row in read_jsonl(args.puzzles):
        name = puzzle_id(row)
        if name in puzzles:
            raise ValueError(f"Duplicate puzzle_id {name}")
        try:
            pairs = unique_grid_pairs(row, train_ids)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{name}: {exc}") from exc
        provenance = {
            key: value for key, value in row.items()
            if key in {"pair_idx", "base_puzzle", "style_anchor_split", "eval_anchor_used"}
            or key.startswith("inspiration_")
        }
        puzzles[name] = (pairs, provenance)

    if not puzzles:
        raise ValueError("No accepted puzzles")
    if not 0 <= args.validation_puzzles < len(puzzles):
        raise ValueError("--validation-puzzles must leave at least one training puzzle")

    names = sorted(puzzles)
    held_out = set(random.Random(args.seed).sample(names, args.validation_puzzles))
    split_names = ("train", "validation") if held_out else ("train",)
    split_puzzles = Counter("validation" if name in held_out else "train" for name in names)

    output = args.output_dir
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as staging_dir:
        staging = Path(staging_dir)
        with ExitStack() as stack:
            folds = {
                split: stack.enter_context((staging / f"{split}.jsonl").open("w", encoding="utf-8"))
                for split in split_names
            }
            audit = stack.enter_context((staging / "provenance.jsonl").open("w", encoding="utf-8"))
            for name in names:
                pairs, source = puzzles[name]
                split = "validation" if name in held_out else "train"
                for record, transform in augment_puzzle(name, pairs, args.seed):
                    folds[split].write(json.dumps(record, ensure_ascii=False) + "\n")
                    audit_row = {
                        "puzzle_name": record["puzzle_name"],
                        "puzzle_id": name,
                        "split": split,
                        **source,
                        **transform,
                    }
                    audit.write(json.dumps(audit_row, ensure_ascii=False) + "\n")
        files = [f"{split}.jsonl" for split in split_names] + ["provenance.jsonl"]
        manifest = {
            "input_file": args.puzzles.name,
            "input_sha256": sha256(args.puzzles),
            "training_ids_sha256": sha256(TRAIN_IDS),
            "seed": args.seed,
            "augmentations_per_puzzle": AUGMENTATIONS,
            "demonstrations_per_row": DEMONSTRATIONS,
            "splits": {
                split: {"base_puzzles": split_puzzles[split], "rows": AUGMENTATIONS * split_puzzles[split]}
                for split in split_names
            },
            "sha256": {name: sha256(staging / name) for name in files},
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staging.rename(output)
    print(json.dumps(manifest["splits"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("puzzles", type=Path, help="Stage 4 accepted puzzle JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--validation-puzzles", type=int, default=0)
    args = parser.parse_args()
    try:
        build_dataset(args)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
