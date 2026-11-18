"""Apply the step 188222 mixed GPTQ recipe to a BF16 Hugging Face export."""

import argparse
from importlib.metadata import version
from pathlib import Path
import tempfile

from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from datasets import load_from_disk
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.prepare_model import read_json, write_json
from weights.calibration import COUNT, MAX_LENGTH, MIN_LENGTH, SEED, TRAIN_IDS, TRAIN_INPUTS
from weights.package import IGNORE, TARGETS, validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    requirements = Path(__file__).with_name("requirements.txt")
    for line in requirements.read_text().splitlines():
        name, expected = line.split("==")
        if version(name) != expected:
            raise RuntimeError(f"Expected {name}=={expected}, found {version(name)}")
    config = read_json(args.model / "config.json")
    validate_config(config, quantized=False)
    receipt = read_json(args.model / "SOURCE_CHECKPOINT.json")
    step = receipt.get("step", receipt.get("source_optimizer_step"))
    if not isinstance(step, int) or step < 1:
        raise ValueError("The BF16 export needs a saved optimizer step")
    stats = read_json(args.calibration / "CALIBRATION_STATS.json")
    if stats["training_sha256"] != TRAIN_INPUTS or stats["seed"] != SEED:
        raise ValueError("Use the calibration dataset built by weights.calibration")
    dataset = load_from_disk(str(args.calibration))
    keys = set(dataset["key"])
    if len(dataset) != COUNT or len(keys) != COUNT or not keys <= set(TRAIN_IDS.read_text().splitlines()):
        raise ValueError("Expected 512 distinct ARC-AGI-2 training calibration tasks")
    for row in dataset:
        ids = row["input_ids"]
        if not MIN_LENGTH <= len(ids) <= MAX_LENGTH or row["attention_mask"] != [1] * len(ids):
            raise ValueError(f"Invalid calibration row {row['key']}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if len(tokenizer) != 16:
        raise ValueError("Expected an ARC-16 tokenizer")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True, device_map=None,
    )
    groups = {
        name: QuantizationScheme(
            targets=targets,
            weights=QuantizationArgs(
                num_bits=bits, type="int", strategy="group", group_size=128, symmetric=True,
            ),
        )
        for bits, (name, targets) in TARGETS.items()
    }
    dampening = 0.01
    recipe = GPTQModifier(config_groups=groups, ignore=IGNORE, dampening_frac=dampening)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".gptq-", dir=args.output.parent) as temp:
        staging = Path(temp) / "model"
        oneshot(
            model=model, dataset=dataset, recipe=recipe, max_seq_length=MAX_LENGTH,
            num_calibration_samples=COUNT, output_dir=str(staging), pipeline="sequential",
            shuffle_calibration_samples=False,
        )
        tokenizer.save_pretrained(staging)
        config = read_json(staging / "config.json")
        config.pop("mtp_num_hidden_layers", None)
        validate_config(config)
        write_json(staging / "config.json", config)
        write_json(staging / "GPTQ_RECIPE.json", {
            "variant": "mixed",
            "source_optimizer_step": step,
            "num_samples": COUNT,
            "max_seq_length": MAX_LENGTH,
            "pipeline": "sequential",
            "dampening_frac": dampening,
            "ignore": IGNORE,
        })
        staging.rename(args.output)


if __name__ == "__main__":
    main()
