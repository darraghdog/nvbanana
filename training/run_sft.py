"""Train the ARC-16 Qwen3.8 text model with NeMo RL and Megatron."""

import argparse
from importlib.util import find_spec
import os
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
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
import transformers

from training.compile_cache import COMPLETE_NAME, CacheSettings, cache_settings, publish, seed
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


def throughput_runtime_environment() -> dict[str, str]:
    root = Path(find_spec("nemo_rl").origin).parent
    patches = {
        "algorithms/sft.py": "ARC2_SFT_DATALOADER_OPTIONS_V2",
        "models/megatron/train.py": "ARC2_PRE_FINALIZE_EMPTY_CACHE_V2",
    }
    for relative_path, marker in patches.items():
        if marker not in (root / relative_path).read_text(encoding="utf-8"):
            raise RuntimeError("Apply training/throughput_optimizations.patch to the pinned NeMo RL checkout")
    return {
        "ARC2_NEMORL_DATALOADER_PIN_MEMORY": "1",
        "ARC2_NEMORL_DATALOADER_SNAPSHOT_EVERY_N_STEPS": "1",
        "ARC2_NEMORL_EMPTY_CACHE_BEFORE_FINALIZE_GRADS": "1",
    }


def shuffle_training_dataset(dataset, seed):
    cache_dir = os.environ.get("ARC2_DATASET_SHUFFLE_CACHE_DIR")
    if cache_dir is None:
        return dataset.shuffle(seed=seed, keep_in_memory=True)
    directory = Path(cache_dir)
    if not directory.is_absolute():
        raise ValueError("ARC2_DATASET_SHUFFLE_CACHE_DIR must be absolute")
    fingerprint = dataset._fingerprint
    if not re.fullmatch(r"[0-9a-f]+", fingerprint):
        raise ValueError(f"Unexpected dataset fingerprint {fingerprint!r}")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"shuffle-{fingerprint}-seed{seed}.arrow"
    return dataset.shuffle(seed=seed, indices_cache_file_name=str(path))


def load_split(path, tokenizer, task_spec, config, audit_all, *, cached_shuffle=False):
    dataset = load_from_disk(str(path))
    if dataset.column_names != ["puzzle_name", "messages"] or not len(dataset):
        raise ValueError(f"{path} must be a nonempty dataset from the SDG Hugging Face export")
    sequence_length = config.data["max_input_seq_length"]
    for idx in range(len(dataset) if audit_all else 1):
        prepared = sft_preprocessor(dataset[idx], task_spec, tokenizer, sequence_length, idx)
        print(f"ARC-16 formatter check: {dataset[idx]['puzzle_name']} ({prepared['length']} tokens)", flush=True)
    if cached_shuffle:
        dataset = shuffle_training_dataset(dataset, config.sft.seed)
    return AllTaskProcessedDataset(
        dataset, tokenizer, task_spec, sft_preprocessor, max_seq_length=sequence_length,
    )


def run_cache_on_nodes(action, cache, nodes):
    task = ray.remote(num_cpus=0)(action)
    ray.get([
        task.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node, soft=False),
        ).remote(cache.local_root, cache.snapshot_dir, cache.key)
        for node in nodes
    ])


def seed_compile_cache(cache: CacheSettings, expected_nodes: int) -> list[str]:
    nodes = sorted(
        (node for node in ray.nodes() if node["Alive"] and node["Resources"].get("GPU", 0) > 0),
        key=lambda node: (node["NodeManagerHostname"], node["NodeID"]),
    )
    if len(nodes) != expected_nodes:
        raise RuntimeError(
            f"Compiler cache expects {expected_nodes} live GPU Ray nodes, found {len(nodes)}"
        )
    node_ids = [node["NodeID"] for node in nodes]
    run_cache_on_nodes(seed, cache, node_ids)
    return node_ids


def arm_compile_cache_publication(policy, cache, node):
    train = policy.train
    published = False

    def train_and_publish(*args, **kwargs):
        nonlocal published
        result = train(*args, **kwargs)
        # NeMo RL uses policy.train(eval_mode=True) for validation too.
        if not published and not kwargs.get("eval_mode", False):
            run_cache_on_nodes(publish, cache, [node])
            published = True
        return result

    policy.train = train_and_publish


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
    config_dict = OmegaConf.to_container(config, resolve=True)
    cache = cache_settings(config_dict.pop("compile_cache", None))
    throughput_optimizations = config_dict.pop("throughput_optimizations", False)
    if type(throughput_optimizations) is not bool:
        raise ValueError("throughput_optimizations must be a boolean")
    runtime_env = throughput_runtime_environment() if throughput_optimizations else {}
    config = MasterConfig(**config_dict)

    tokenizer = get_tokenizer(config.policy["tokenizer"])
    if len(tokenizer) != 16:
        raise ValueError(f"Expected 16 tokenizer IDs, got {len(tokenizer)}")
    task_spec = TaskDataSpec(task_name="AGI")
    train = load_split(
        args.data / "train", tokenizer, task_spec, config, args.preflight,
        cached_shuffle=throughput_optimizations and not args.preflight,
    )
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
    if cache is not None:
        runtime_env.update(cache.environment())
    if runtime_env:
        os.environ.update(runtime_env)
        megatron = config.policy["megatron_cfg"]
        megatron["env_vars"] = {**(megatron.get("env_vars") or {}), **runtime_env}

    init_ray()
    cache_nodes = seed_compile_cache(cache, config.cluster["num_nodes"]) if cache else []
    policy, _, train_loader, val_loader, loss_fn, logger, checkpointer, state, config = setup(
        config, tokenizer, train, validation,
    )
    if cache is not None and not (cache.snapshot_dir / COMPLETE_NAME).is_file():
        arm_compile_cache_publication(policy, cache, cache_nodes[0])
    sft_train(policy, train_loader, val_loader, tokenizer, loss_fn, config, logger, checkpointer, state)
    print("ARC-16 SFT finished", flush=True)


if __name__ == "__main__":
    main()
