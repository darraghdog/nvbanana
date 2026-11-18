"""Stage a mixed GPTQ model and its split-TTFT metadata for Kaggle."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import tempfile

from training.prepare_model import ASSETS, BLOCK_SIZE, INDEX, read_json, write_json


REFERENCE = Path(__file__).with_name("published.json")
LIMIT = 4 << 30
SPLIT = 32
LAYER = re.compile(r"^model\.layers\.(\d+)\.")
MODEL_FILE = "model.safetensors"
ASSET_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json")
RECIPE_FILES = ("GPTQ_RECIPE.json", "recipe.yaml")
IGNORE = ["lm_head", "re:.*embed_tokens$", "re:.*in_proj_a$", "re:.*in_proj_b$", "re:.*conv1d$"]
TARGETS = {
    4: ("mlp_w4", [r"re:.*mlp\.(gate_proj|up_proj|down_proj)$"]),
    8: ("attn_gdn_w8", [
        r"re:.*self_attn\.(q_proj|k_proj|v_proj|o_proj)$",
        r"re:.*linear_attn\.(in_proj_qkvz|in_proj_qkv|in_proj_z|out_proj)$",
    ]),
}
FP8 = {
    "quant_method": "fp8",
    "modules_to_not_convert": ["lm_head", "embed_tokens", "in_proj_a", "in_proj_b", "conv1d"],
    "activation_scheme": "dynamic",
    "weight_block_size": [128, 128],
    "dequantize": False,
}


def validate_config(config, quantized=True):
    contract = read_json(ASSETS / "model_spec.json")["config_contract"]
    if quantized:
        contract.pop("mtp_num_hidden_layers")
    if any(config.get(key) != value for key, value in contract.items()):
        raise ValueError("Expected the 64-layer Qwen3.8 ARC-16 model config")
    layer_types = config["layer_types"]
    if len(layer_types) != 2 * SPLIT or layer_types[:SPLIT] != layer_types[SPLIT:]:
        raise ValueError("Expected matching layer types in both TTFT halves")
    quantization = config.get("quantization_config")
    if not quantized:
        if quantization or config.get("dtype") != "bfloat16":
            raise ValueError("Quantization requires the BF16 checkpoint export")
        return
    if config.get("mtp_num_hidden_layers", 0) or not quantization:
        raise ValueError("Expected quantized weights without an MTP head")
    groups = list(quantization["config_groups"].values())
    if quantization["format"] != "pack-quantized" or sorted(
        group["weights"]["num_bits"] for group in groups
    ) != sorted(TARGETS):
        raise ValueError("Expected the mixed pack-quantized GPTQ recipe")
    for group in groups:
        weights = group["weights"]
        fields = {"group_size": 128, "symmetric": True, "strategy": "group", "type": "int", "actorder": "static"}
        if (group["targets"] != TARGETS[weights["num_bits"]][1]
                or group.get("input_activations") is not None
                or group.get("output_activations") is not None
                or any(weights.get(key) != value for key, value in fields.items())):
            raise ValueError("GPTQ targets or quantization parameters differ from the release recipe")


def half_metadata(config, weight_map, offset):
    end = offset + SPLIT
    half_config = dict(config, num_hidden_layers=SPLIT, layer_types=config["layer_types"][offset:end])
    half_config["quantization_config"] = FP8
    half_map = {}
    for name, shard in weight_map.items():
        layer = LAYER.match(name)
        if layer is None or offset <= int(layer[1]) < end:
            half_map[name] = shard
    source = {
        "source": "..",
        "layer_offset": offset,
        "layer_prefix": "model",
        "quantization_config": config["quantization_config"],
    }
    return half_config, {"metadata": {}, "weight_map": half_map}, source


def read_header(handle):
    length = struct.unpack("<Q", handle.read(8))[0]
    if not 0 < length <= 100_000_000:
        raise ValueError("Invalid Safetensors header size")
    header = json.loads(handle.read(length))
    return header, handle.tell()


def tensor_groups(header, limit):
    group = []
    size = 0
    for name in sorted(header):
        start, end = header[name]["data_offsets"]
        length = end - start
        if not 0 <= start <= end or length > limit:
            raise ValueError(f"Tensor {name} cannot fit in a shard")
        if group and size + length > limit:
            yield group
            group = []
            size = 0
        group.append(name)
        size += length
    if group:
        yield group


def reshard(source, output, limit):
    weight_map = {}
    total_size = 0
    with source.open("rb") as handle:
        header, payload = read_header(handle)
        metadata = header.pop("__metadata__", {"format": "pt"})
        groups = list(tensor_groups(header, limit))
        for index, group in enumerate(groups, 1):
            filename = f"model-{index:05d}-of-{len(groups):05d}.safetensors"
            shard_header = {"__metadata__": metadata}
            offset = 0
            for name in group:
                start, end = header[name]["data_offsets"]
                length = end - start
                shard_header[name] = dict(header[name], data_offsets=[offset, offset + length])
                offset += length
            encoded = json.dumps(shard_header, separators=(",", ":")).encode()
            encoded += b" " * (-len(encoded) % 8)
            with (output / filename).open("xb") as shard:
                shard.write(struct.pack("<Q", len(encoded)))
                shard.write(encoded)
                for name in group:
                    start, end = header[name]["data_offsets"]
                    handle.seek(payload + start)
                    remaining = end - start
                    while remaining:
                        block = handle.read(min(remaining, BLOCK_SIZE))
                        if not block:
                            raise OSError(f"Truncated tensor {name} in {source}")
                        shard.write(block)
                        remaining -= len(block)
                    weight_map[name] = filename
            total_size += offset
    return {"metadata": {"total_size": total_size}, "weight_map": weight_map}


def build(source, output):
    if output.exists():
        raise FileExistsError(output)
    config = read_json(source / "config.json")
    validate_config(config)
    recipe = read_json(source / "GPTQ_RECIPE.json")
    step = recipe["source_optimizer_step"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".kaggle-weights-", dir=output.parent) as temp:
        stage = Path(temp) / "package"
        stage.mkdir()
        for name in (*ASSET_FILES, *RECIPE_FILES, "config.json"):
            shutil.copyfile(source / name, stage / name)
        index = reshard(source / MODEL_FILE, stage, LIMIT)
        write_json(stage / INDEX, index)
        for dirname, offset in (("bottom", 0), ("top", SPLIT)):
            half = stage / dirname
            half.mkdir()
            for name in (*ASSET_FILES, *RECIPE_FILES):
                shutil.copyfile(stage / name, half / name)
            half_config, half_index, marker = half_metadata(config, index["weight_map"], offset)
            write_json(half / "config.json", half_config)
            write_json(half / INDEX, half_index)
            write_json(half / "gptq_source.json", marker)
        write_json(stage / "PROVENANCE.json", {
            "step": step,
            "quantization": "GPTQ mixed W4/W8, group size 128",
            "halves": "FP8 at load from the shared GPTQ shards",
        })
        verify(stage, metadata_only=False, published=False)
        stage.rename(output)


def canonical_sha(value):
    data = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(data).hexdigest()


def verify(package, metadata_only, published):
    config = read_json(package / "config.json")
    validate_config(config)
    index = read_json(package / INDEX)
    weight_map = index["weight_map"]
    layers = {int(match[1]) for name in weight_map if (match := LAYER.match(name))}
    if layers != set(range(2 * SPLIT)) or not {"lm_head.weight", "model.embed_tokens.weight"} <= weight_map.keys():
        raise ValueError("Unexpected packed tensor inventory")
    if any(name.startswith(("mtp.", "model.visual.", "model.language_model.")) for name in weight_map):
        raise ValueError("Expected a text-only model without MTP")
    step = read_json(package / "PROVENANCE.json")["step"]
    for dirname, offset in (("bottom", 0), ("top", SPLIT)):
        half = package / dirname
        expected_config, expected_index, expected_source = half_metadata(config, weight_map, offset)
        actual_source = read_json(half / "gptq_source.json")
        actual_source.pop("source", None)
        expected_source.pop("source")
        if (read_json(half / "config.json") != expected_config
                or read_json(half / INDEX) != expected_index
                or actual_source != expected_source):
            raise ValueError(f"{dirname} TTFT metadata differs from the root GPTQ model")
        if tuple(half.glob("*.safetensors")):
            raise ValueError(f"{dirname} must refer to the root shards")
    if published:
        reference = read_json(REFERENCE)
        if step != reference["step"]:
            raise ValueError(f"Expected published step {reference['step']}")
        for name, digest in reference["metadata_sha256"].items():
            if canonical_sha(read_json(package / name)) != digest:
                raise ValueError(f"{name} differs from Kaggle v1 metadata")
    if not metadata_only:
        for name in ASSET_FILES:
            for directory in (package, package / "bottom", package / "top"):
                if not (directory / name).is_file():
                    raise FileNotFoundError(directory / name)
        total_size = 0
        for name in sorted(set(weight_map.values())):
            if not re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", name):
                raise ValueError(f"Invalid shard name {name}")
            path = package / name
            with path.open("rb") as handle:
                header, payload = read_header(handle)
            header.pop("__metadata__", None)
            size = path.stat().st_size - payload
            expected_names = {tensor for tensor, shard in weight_map.items() if shard == name}
            end = max(tensor["data_offsets"][1] for tensor in header.values())
            if header.keys() != expected_names or end != size or size > LIMIT:
                raise ValueError(f"Shard does not match the index: {name}")
            total_size += size
        if total_size != index["metadata"]["total_size"]:
            raise ValueError("Shard payload sizes do not match the index")
    return len(weight_map), len(set(weight_map.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("build")
    make.add_argument("--source", type=Path, required=True)
    make.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("verify")
    check.add_argument("package", type=Path)
    check.add_argument("--metadata-only", action="store_true")
    check.add_argument("--published", action="store_true")
    args = parser.parse_args()
    if args.command == "build":
        build(args.source, args.output)
        print(f"Prepared {args.output}")
    else:
        tensors, shards = verify(args.package, args.metadata_only, args.published)
        print(f"Verified {tensors} packed tensors across {shards} shards")


if __name__ == "__main__":
    main()
