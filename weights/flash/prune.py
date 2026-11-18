"""Build the po352 NVFP4 checkpoint using the submitted expert IDs."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from safetensors import safe_open
from safetensors.torch import save_file

from training.prepare_model import INDEX, read_json, sha256


HERE = Path(__file__).parent
KEEP = HERE / "keep_ours_352.json"
PUBLISHED = HERE / "published.json"
EXPERT = re.compile(r"^(model\.language_model\.layers\.(\d+)\.mlp\.experts\.)(\d+)(\..*)?$")
ROUTER = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.gate\.weight$")
LAYERS = 48
SOURCE_EXPERTS = 448
KEEP_EXPERTS = 352


def write_json(path, data):
    # Preserve the original serializer and field order of the model metadata.
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_keep():
    keep = read_json(KEEP)
    layers = keep["layers"]
    if keep["num_experts"] != KEEP_EXPERTS or set(layers) != {str(i) for i in range(LAYERS)}:
        raise ValueError("Invalid po352 layer manifest")
    for layer, ids in layers.items():
        if (len(ids) != KEEP_EXPERTS or ids != sorted(set(ids))
                or not 0 <= ids[0] <= ids[-1] < SOURCE_EXPERTS):
            raise ValueError(f"Invalid po352 expert IDs for layer {layer}")
    return layers


def write_shard(args):
    source, destination, keys, layers, remap = args
    tensors = {}
    with safe_open(source, framework="pt", device="cpu") as weights:
        metadata = weights.metadata() or {"format": "pt"}
        for name in keys:
            if match := EXPERT.fullmatch(name):
                prefix, layer, id_text, suffix = match.groups()
                expert = int(id_text)
                if expert in remap[layer]:
                    tensors[f"{prefix}{remap[layer][expert]}{suffix}"] = weights.get_tensor(name)
            elif match := ROUTER.fullmatch(name):
                router = weights.get_tensor(name)
                if router.shape[0] != SOURCE_EXPERTS:
                    raise ValueError(f"Unexpected router shape for {name}")
                tensors[name] = router[layers[match[1]]].contiguous()
            else:
                tensors[name] = weights.get_tensor(name)
    save_file(tensors, destination, metadata=metadata)
    return destination.name, sorted(tensors), sum(t.numel() * t.element_size() for t in tensors.values())


def remap_modules(names, remap):
    result = []
    for name in names:
        if match := EXPERT.fullmatch(name):
            prefix, layer, id_text, suffix = match.groups()
            expert = int(id_text)
            if layer in remap:
                if expert in remap[layer]:
                    result.append(f"{prefix}{remap[layer][expert]}{suffix or ''}")
                continue
        result.append(name)
    return result


def validate_index(weight_map, expected_count):
    experts = {str(i): {} for i in range(LAYERS)}
    routers = set()
    for name in weight_map:
        if match := EXPERT.fullmatch(name):
            _, layer, id_text, suffix = match.groups()
            experts[layer].setdefault(int(id_text), set()).add(suffix)
        elif match := ROUTER.fullmatch(name):
            routers.add(match[1])
    for layer, ids in experts.items():
        if (set(ids) != set(range(expected_count))
                or len({frozenset(suffixes) for suffixes in ids.values()}) != 1):
            raise ValueError(f"Incomplete expert bank in layer {layer}")
    if routers != experts.keys():
        raise ValueError("Expected a router for each routed expert layer")


def build(source, output, workers):
    if output.exists():
        raise FileExistsError(output)
    release = read_json(PUBLISHED)
    for name, digest in release["source"]["sha256"].items():
        if sha256(source / name) != digest:
            raise ValueError(f"{name} differs from the pinned REAP448 checkpoint")
    layers = load_keep()
    remap = {layer: {old: new for new, old in enumerate(ids)} for layer, ids in layers.items()}
    indexed = read_json(source / INDEX)
    weight_map = indexed["weight_map"]
    validate_index(weight_map, SOURCE_EXPERTS)
    groups = {}
    for name, filename in weight_map.items():
        groups.setdefault(filename, []).append(name)
    generated = {"README.md", "prune_keepset.json", "config.json", "hf_quant_config.json", INDEX}
    source_files = [
        name for name in release["parts"]["part1"]["files"]
        if not name.endswith(".safetensors") and name not in generated
    ]
    for name in (*groups, *source_files, "hf_quant_config.json"):
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".po352-", dir=output.parent) as temp:
        stage = Path(temp) / "checkpoint"
        stage.mkdir()
        new_map = {}
        total_size = 0
        jobs = []
        for filename, keys in sorted(groups.items()):
            path = source / filename
            if any(EXPERT.fullmatch(key) or ROUTER.fullmatch(key) for key in keys):
                jobs.append((path, stage / filename, keys, layers, remap))
            else:
                # Reuse untouched shards, including the original PLE and MTP bank.
                try:
                    os.link(path.resolve(), stage / filename)
                except OSError:
                    (stage / filename).symlink_to(path.resolve())
                new_map.update(dict.fromkeys(keys, filename))
                total_size += path.stat().st_size
        with ProcessPoolExecutor(workers) as pool:
            for filename, keys, size in pool.map(write_shard, jobs):
                new_map.update(dict.fromkeys(keys, filename))
                total_size += size
        validate_index(new_map, KEEP_EXPERTS)
        write_json(stage / INDEX, {
            "metadata": {**indexed.get("metadata", {}), "total_size": total_size},
            "weight_map": dict(sorted(new_map.items())),
        })

        for name in source_files:
            shutil.copyfile(source / name, stage / name)

        config = read_json(source / "config.json")
        text = config.get("text_config", config)
        if text["num_experts"] != SOURCE_EXPERTS or text["num_experts_per_tok"] != 10:
            raise ValueError("Expected 448 routed experts with top-10 routing")
        text["num_experts"] = KEEP_EXPERTS
        quantization = config["quantization_config"]
        for field in ("ignore", "modules_to_not_convert", "modules_to_convert"):
            if field in quantization:
                quantization[field] = remap_modules(quantization[field], remap)
        write_json(stage / "config.json", config)

        hf_quant = read_json(source / "hf_quant_config.json")
        hf_quant["quantization"]["exclude_modules"] = remap_modules(
            hf_quant["quantization"]["exclude_modules"], remap,
        )
        write_json(stage / "hf_quant_config.json", hf_quant)
        shutil.copyfile(KEEP, stage / "prune_keepset.json")
        (stage / "README.md").write_text(
            "# Qwen3.8-Flash-Next NVFP4 po352\n\n"
            f"Source: {release['source']['model']} at {release['source']['revision']}.\n"
            "The keep-list covers 352 routed experts per layer. Top-10 routing and the MTP bank are preserved.\n",
            encoding="utf-8",
        )
        stage.rename(output)
    print(f"Built {output} with {KEEP_EXPERTS} routed experts per layer")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    build(args.source, args.output, args.workers)


if __name__ == "__main__":
    main()
