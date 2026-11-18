"""Export a NeMo RL Megatron checkpoint as BF16 ARC-16 Hugging Face weights."""

import argparse
from pathlib import Path
import shutil
import tempfile

from nemo_rl.models.megatron.community_import import export_model_from_megatron
from safetensors import safe_open
import yaml

from training.prepare_model import ASSETS, INDEX, read_json, sha256, write_json
from training.run_sft import verify_model, verify_runtime
from weights.package import validate_config


def checkpoint_inputs(step_dir, source, step):
    weights = step_dir / "policy/weights"
    checkpoint = weights / "iter_0000000"
    required = (
        step_dir / "config.yaml",
        step_dir / "training_info.json",
        weights / "latest_train_state.pt",
        checkpoint / ".metadata",
        checkpoint / "common.pt",
        source / INDEX,
        source / "conversion_manifest.json",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if step_dir.name != f"step_{step}" or int(read_json(step_dir / "training_info.json")["total_steps"]) != step:
        raise ValueError(f"Expected a step_{step} checkpoint and matching training_info.json")
    policy = yaml.safe_load((step_dir / "config.yaml").read_text())["policy"]
    if policy["model_name"] != policy["tokenizer"]["name"]:
        raise ValueError("Expected the ARC-16 base and tokenizer to come from the same model")
    verify_model(source)
    spec = read_json(ASSETS / "model_spec.json")
    if read_json(source / "conversion_manifest.json")["token_id_mapping"] != spec["token_id_mapping"]:
        raise ValueError("The base model has a different compact source ID mapping")
    return checkpoint, policy.get("hf_overrides") or {}


def verify_export(staging):
    validate_config(read_json(staging / "config.json"), quantized=False)
    verify_model(staging)
    weight_map = read_json(staging / INDEX)["weight_map"]
    if any(name.startswith(("mtp.", "model.visual.", "model.language_model.")) for name in weight_map):
        raise ValueError("Unexpected vision or MTP weights in the text export")
    compact = {"model.embed_tokens.weight", "lm_head.weight"}
    if not compact <= weight_map.keys():
        raise ValueError("Missing ARC-16 input embedding or output head")
    for shard in sorted(set(weight_map.values())):
        expected = {name for name, filename in weight_map.items() if filename == shard}
        with safe_open(staging / shard, framework="pt", device="cpu") as weights:
            if set(weights.keys()) != expected:
                raise ValueError(f"Export index disagrees with {shard}")
            for name in expected:
                tensor = weights.get_slice(name)
                if tensor.get_dtype() != "BF16":
                    raise ValueError(f"Expected BF16 output for {name}")
                if name in compact and tensor.get_shape() != [16, 5120]:
                    raise ValueError(f"Expected a 16 × 5120 tensor for {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step-dir", type=Path, required=True)
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, default=188222)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    verify_runtime()
    checkpoint, overrides = checkpoint_inputs(args.step_dir, args.source_artifact, args.expected_step)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".hf-export-", dir=args.output.parent) as temp:
        staging = Path(temp) / "model"
        source = str(args.source_artifact.resolve())
        export_model_from_megatron(
            hf_model_name=source, input_path=str(checkpoint.resolve()), output_path=str(staging),
            hf_tokenizer_path=source, overwrite=False, hf_overrides=overrides,
        )
        for name in read_json(ASSETS / "model_spec.json")["tokenizer_sha256"]:
            shutil.copyfile(args.source_artifact / name, staging / name)
        write_json(staging / "generation_config.json", {
            "_from_model_config": True, "eos_token_id": 15, "pad_token_id": 13,
        })
        verify_export(staging)
        write_json(staging / "SOURCE_CHECKPOINT.json", {
            "step": args.expected_step,
            "config_sha256": sha256(staging / "config.json"),
            "weight_index_sha256": sha256(staging / INDEX),
        })
        staging.rename(args.output)
    print(f"Exported step {args.expected_step} to {args.output}")


if __name__ == "__main__":
    main()
