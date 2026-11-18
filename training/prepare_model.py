"""Build the pinned Qwen3.8-27B text-only ARC-16 training base."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import tempfile


ASSETS = Path(__file__).with_name("arc16")
INDEX = "model.safetensors.index.json"
BLOCK_SIZE = 16 << 20
PREFIX = "model.language_model."
SLICED = {"model.embed_tokens.weight", "lm_head.weight"}
TOKENS = tuple("0123456789") + ("Ċ", "user", "assistant", "<|endoftext|>", "<|im_start|>", "<|im_end|>")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(BLOCK_SIZE), b""):
            result.update(block)
    return result.hexdigest()


def make_plan(source, spec):
    for name in ("config.json", INDEX):
        if sha256(source / name) != spec["source_sha256"][name]:
            raise ValueError(f"{name} does not match the pinned source revision")
    tokenizer = read_json(source / "tokenizer.json")
    ids = tokenizer["model"]["vocab"] | {entry["content"]: entry["id"] for entry in tokenizer["added_tokens"]}
    mapping = spec["token_id_mapping"]
    if [ids.get(token) for token in TOKENS] != mapping:
        raise ValueError("The source tokenizer does not match the ARC-16 row mapping")

    for name, expected in spec["tokenizer_sha256"].items():
        if sha256(ASSETS / name) != expected:
            raise ValueError(f"Bundled tokenizer asset checksum mismatch: {name}")

    weight_map = read_json(source / INDEX)["weight_map"]
    by_shard = {}
    for name, shard in weight_map.items():
        by_shard.setdefault(shard, set()).add(name)
    expected = spec["tensor_stats"]
    if set(by_shard) != set(expected["source_shards"]) or len(weight_map) != expected["source_tensor_count"]:
        raise ValueError("Unexpected source tensor inventory")

    plan = {}
    target_names = set()
    tensor_bytes = 0
    for shard, names in sorted(by_shard.items()):
        path = source / shard
        with path.open("rb") as handle:
            header_size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_size))
        header.pop("__metadata__", None)
        if set(header) != names:
            raise ValueError(f"{shard} tensor inventory does not match its index")
        rows = {}
        for name, info in sorted(header.items(), key=lambda item: item[1]["data_offsets"][0]):
            if name.startswith(tuple(spec["skipped_prefixes"])):
                continue
            target = "model." + name[len(PREFIX):] if name.startswith(PREFIX) else name
            if target in target_names:
                raise ValueError(f"Duplicate output tensor {target}")
            shape = info["shape"]
            start, end = info["data_offsets"]
            size = end - start
            start += 8 + header_size
            if size < 0 or start + size > path.stat().st_size:
                raise ValueError(f"Invalid tensor data range for {name}")
            ranges = [(start, size)]
            if target in SLICED:
                width = spec["config_contract"]["hidden_size"]
                if (
                    len(shape) != 2 or shape[1] != width or shape[0] <= max(mapping)
                    or info["dtype"] != "BF16" or size != shape[0] * width * 2
                ):
                    raise ValueError(f"Unexpected vocabulary matrix shape or dtype for {name}")
                ranges = [(start + index * width * 2, width * 2) for index in mapping]
                shape = [len(mapping), width]
            rows[target] = (info["dtype"], shape, ranges)
            target_names.add(target)
            tensor_bytes += sum(size for _, size in ranges)
        if rows:
            plan[shard] = rows

    if (
        not SLICED <= target_names
        or len(target_names) != expected["output_tensor_count"]
        or tensor_bytes != expected["output_tensor_bytes"]
    ):
        raise ValueError("Output tensor count or size differs from the verified model")
    return plan, tensor_bytes


def write_shard(source, destination, rows):
    index = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, (dtype, shape, ranges) in rows.items():
        length = sum(size for _, size in ranges)
        index[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + length]}
        offset += length
    header = json.dumps(index, separators=(",", ":")).encode()
    header += b" " * (-len(header) % 8)

    with source.open("rb") as src, destination.open("xb") as out:
        out.write(struct.pack("<Q", len(header)))
        out.write(header)
        for _, _, ranges in rows.values():
            for start, size in ranges:
                src.seek(start)
                while size:
                    block = src.read(min(size, BLOCK_SIZE))
                    if not block:
                        raise OSError(f"Unexpected EOF reading {source}")
                    out.write(block)
                    size -= len(block)


def prepare(source, output, dry_run):
    spec = read_json(ASSETS / "model_spec.json")
    plan, tensor_bytes = make_plan(source, spec)
    print(f"ARC-16 plan: {len(plan)} output shards, {tensor_bytes:,} tensor bytes", flush=True)
    if dry_run:
        return
    if output is None:
        raise ValueError("--output is required for a full build")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        staged = Path(temporary)
        target_index = {}
        for shard, expected in sorted(spec["source_sha256"].items()):
            if not shard.endswith(".safetensors"):
                continue
            if sha256(source / shard) != expected:
                raise ValueError(f"{shard} does not match the pinned source revision")
            if shard in plan:
                write_shard(source / shard, staged / shard, plan[shard])
                target_index.update({name: shard for name in plan[shard]})
                print(f"Wrote {shard}", flush=True)

        text_config = read_json(source / "config.json")["text_config"]
        text_config.update(spec["config_contract"])
        write_json(staged / "config.json", text_config)
        write_json(staged / INDEX, {"metadata": {"total_size": tensor_bytes}, "weight_map": target_index})
        for name in spec["tokenizer_sha256"]:
            shutil.copyfile(ASSETS / name, staged / name)
        write_json(staged / "conversion_manifest.json", {
            "source_model_id": spec["source_model_id"],
            "source_revision": spec["source_revision"],
            "source_sha256": spec["source_sha256"],
            "token_id_mapping": spec["token_id_mapping"],
            "config_contract": spec["config_contract"],
            "output_sha256": {path.name: sha256(path) for path in sorted(staged.iterdir())},
        })
        staged.rename(output)
    print(f"Prepared {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="verify pinned metadata and plan without writing weights")
    args = parser.parse_args()
    try:
        prepare(args.source, args.output, args.dry_run)
    except (OSError, ValueError, KeyError, struct.error) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
