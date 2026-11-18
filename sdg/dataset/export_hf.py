"""Export the release JSONL splits as Hugging Face datasets for NeMo RL."""

import argparse
import json
from pathlib import Path
import tempfile

from datasets import Dataset, Features, Value

from sdg.dataset.build_dataset import sha256
from sdg.mixing.mix_puzzles import read_jsonl


FEATURES = Features({
    "puzzle_name": Value("string"),
    "messages": [{"role": Value("string"), "content": Value("string")}],
})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        parser.exit(1, f"error: Output directory already exists: {args.output_dir}\n")

    manifest = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent) as staging_dir:
        staging = Path(staging_dir)
        for split, counts in manifest["splits"].items():
            if split not in {"train", "validation"}:
                raise ValueError(f"Unknown dataset split: {split}")
            filename = f"{split}.jsonl"
            source = args.dataset / filename
            if sha256(source) != manifest["sha256"][filename]:
                raise ValueError(f"Checksum mismatch for {source}")
            rows = list(read_jsonl(source))
            if len(rows) != counts["rows"]:
                raise ValueError(f"Row count mismatch for {source}")
            Dataset.from_list(rows, features=FEATURES).save_to_disk(staging / split)
        staging.rename(args.output_dir)


if __name__ == "__main__":
    main()
