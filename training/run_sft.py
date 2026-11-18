"""Train the ARC-16 Qwen3.8 text model with NeMo RL and Megatron."""

import argparse
from importlib.util import find_spec
from pathlib import Path
import re
import subprocess

from datasets import load_from_disk
from nemo_rl.algorithms.sft import MasterConfig, setup, sft_train
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.llm_message_utils import get_formatted_message_log
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
from nemo_rl.utils.logger import get_next_experiment_dir
from omegaconf import OmegaConf
import transformers

from training.compact_role_tokens import require_compact_role_tokens, restore_compact_role_tokens
from training.prepare_model import ASSETS, read_json, sha256


HERE = Path(__file__).resolve().parent


def verify_runtime():
    pins = read_json(HERE / "runtime.json")
    for module, name in (("nemo_rl", "nemo_rl"), ("megatron.bridge", "megatron_bridge")):
        package = find_spec(module)
        if package is None or package.origin is None:
            raise ValueError(f"Missing pinned source package: {module}")
        path = Path(package.origin).resolve().parent
        revision = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        if revision != pins[name]:
            raise ValueError(f"{name} expected {pins[name]}, got {revision}")
    if transformers.__version__ != pins["transformers"]:
        raise ValueError(f"Expected Transformers {pins['transformers']}, got {transformers.__version__}")


def verify_model(path):
    spec = read_json(ASSETS / "model_spec.json")
    config = read_json(path / "config.json")
    if any(config.get(key) != value for key, value in spec["config_contract"].items()):
        raise ValueError("Model config does not match the Qwen3.8 ARC-16 contract")
    for name, digest in spec["tokenizer_sha256"].items():
        if sha256(path / name) != digest:
            raise ValueError(f"Model tokenizer asset differs from the ARC-16 release: {name}")


def sft_preprocessor(sample, task_data_spec, tokenizer, max_seq_length, idx):
    messages = sample["messages"]
    expected_roles = ["user", "assistant"] * (len(messages) // 2)
    if not messages or [message["role"] for message in messages] != expected_roles:
        raise ValueError(f"Sample {idx} must alternate user and assistant grid messages")

    expected_ids = []
    for message in messages:
        grid = message["content"]
        if not re.fullmatch(r"[0-9]+(?:\n[0-9]+)*", grid):
            raise ValueError(f"Sample {idx} contains non-grid text")
        role_id = 11 if message["role"] == "user" else 12
        expected_ids.extend((14, role_id, 10))
        expected_ids.extend(10 if cell == "\n" else int(cell) for cell in grid)
        expected_ids.append(15)

    formatted = get_formatted_message_log(
        messages, tokenizer, task_data_spec,
        add_bos_token=False, add_eos_token=False, add_generation_prompt=True,
    )
    restore_compact_role_tokens(formatted)
    require_compact_role_tokens(formatted)
    actual_ids = [int(token) for chunk in formatted for token in chunk["token_ids"]]
    if actual_ids != expected_ids:
        raise ValueError(f"Sample {idx} token stream differs from the ARC-16 chat template")
    length = len(actual_ids)
    if length > max_seq_length:
        raise ValueError(f"Sample {idx}: {length} tokens exceeds {max_seq_length}")
    return {
        "message_log": formatted,
        "length": length,
        "extra_env_info": None,
        "loss_multiplier": 1.0,
        "idx": idx,
    }


def load_split(path, tokenizer, task_spec, config, audit_all):
    dataset = load_from_disk(str(path))
    if dataset.column_names != ["puzzle_name", "messages"] or not len(dataset):
        raise ValueError(f"{path} must be a nonempty dataset from the SDG Hugging Face export")
    sequence_length = config.data["max_input_seq_length"]
    for idx in range(len(dataset) if audit_all else 1):
        prepared = sft_preprocessor(dataset[idx], task_spec, tokenizer, sequence_length, idx)
        print(f"ARC-16 formatter check: {dataset[idx]['puzzle_name']} ({prepared['length']} tokens)", flush=True)
    return AllTaskProcessedDataset(
        dataset, tokenizer, task_spec, sft_preprocessor, max_seq_length=sequence_length,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="Hugging Face export containing train and optional validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=HERE / "sft_smoke.yaml")
    parser.add_argument("--preflight", action="store_true", help="verify every row with the pinned NeMo formatter, without initializing Ray")
    args = parser.parse_args()

    verify_runtime()
    verify_model(args.model)
    register_omegaconf_resolvers()
    config = load_config(str(args.config))
    config.policy.model_name = str(args.model.resolve())
    config.checkpointing.checkpoint_dir = str((args.output / "checkpoints").resolve())
    config.logger.log_dir = str((args.output / "logs").resolve())
    config = MasterConfig(**OmegaConf.to_container(config, resolve=True))

    tokenizer = get_tokenizer(config.policy["tokenizer"])
    if len(tokenizer) != 16:
        raise ValueError(f"Expected 16 tokenizer IDs, got {len(tokenizer)}")
    task_spec = TaskDataSpec(task_name="AGI")
    train = load_split(args.data / "train", tokenizer, task_spec, config, args.preflight)
    if len(train) < config.policy["train_global_batch_size"]:
        raise ValueError("Training split is smaller than the global batch size")
    validation = None
    if (args.data / "validation").exists():
        validation = load_split(args.data / "validation", tokenizer, task_spec, config, args.preflight)
    elif config.sft.val_at_start or config.sft.val_at_end or config.sft.val_period > 0:
        raise ValueError("Validation is enabled but the dataset has no validation split")

    if args.preflight:
        return

    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    init_ray()
    policy, _, train_loader, val_loader, loss_fn, logger, checkpointer, state, config = setup(
        config, tokenizer, train, validation,
    )
    sft_train(policy, train_loader, val_loader, tokenizer, loss_fn, config, logger, checkpointer, state)
    print("ARC-16 SFT finished", flush=True)


if __name__ == "__main__":
    main()
