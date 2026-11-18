"""Re-shard po352 at tensor boundaries and verify the two Kaggle parts."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import struct
import tempfile
from typing import NamedTuple

from training.prepare_model import BLOCK_SIZE, INDEX, read_json, sha256
from weights.package import read_header


PUBLISHED = Path(__file__).with_name("published.json")
PAYLOAD_LIMIT = 4_000_000_000


class Tensor(NamedTuple):
    name: str
    path: Path
    dtype: str
    shape: list[int]
    offset: int
    length: int


def read_tensors(index, paths):
    """Follow the original file name and data offset order."""
    tensors = []
    metadata = None
    found = {}
    for name in sorted(set(index["weight_map"].values())):
        path = paths[name]
        with path.open("rb") as handle:
            header, base = read_header(handle)
        shard_metadata = header.pop("__metadata__", None)
        metadata = metadata or shard_metadata
        file_size = path.stat().st_size
        offset = 0
        for key, entry in sorted(header.items(), key=lambda item: item[1]["data_offsets"][0]):
            start, end = entry["data_offsets"]
            if key in found or start != offset or end < start or base + end > file_size:
                raise ValueError(f"Invalid tensor offset or duplicate tensor in {name}: {key}")
            tensors.append(Tensor(key, path, entry["dtype"], entry["shape"], base + start, end - start))
            found[key] = name
            offset = end
        if base + offset != file_size:
            raise ValueError(f"Invalid file length for {name}")
    if found != index["weight_map"]:
        raise ValueError("Safetensors headers disagree with the weight index")
    return tensors, metadata or {"format": "pt"}


def groups(tensors, payload_limit):
    group = []
    size = 0
    for tensor in tensors:
        if tensor.length > payload_limit:
            raise ValueError(f"{tensor.name} exceeds the shard payload limit")
        if group and size + tensor.length > payload_limit:
            yield group
            group = []
            size = 0
        group.append(tensor)
        size += tensor.length
    if group:
        yield group


def read_payload(handle, tensor):
    handle.seek(tensor.offset)
    remaining = tensor.length
    while remaining:
        block = handle.read(min(BLOCK_SIZE, remaining))
        if not block:
            raise OSError(f"Truncated tensor {tensor.name} in {tensor.path}")
        yield block
        remaining -= len(block)


def write_shard(task):
    path, metadata, tensors = task
    header = {"__metadata__": metadata}
    offset = 0
    for tensor in tensors:
        end = offset + tensor.length
        header[tensor.name] = {
            "dtype": tensor.dtype, "shape": tensor.shape, "data_offsets": [offset, end],
        }
        offset = end
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    with path.open("xb") as output, ExitStack() as files:
        output.write(struct.pack("<Q", len(encoded)) + encoded)
        handles = {}
        for tensor in tensors:
            if tensor.path not in handles:
                handles[tensor.path] = files.enter_context(tensor.path.open("rb"))
            for block in read_payload(handles[tensor.path], tensor):
                output.write(block)
    return path.name


def inventory(part1, part2, release):
    paths = {}
    for name, directory in (("part1", part1), ("part2", part2)):
        files = release["parts"][name]["files"]
        actual = {path.name for path in directory.iterdir() if path.is_file()}
        if actual != files.keys():
            missing = sorted(files.keys() - actual)
            extra = sorted(actual - files.keys())
            raise ValueError(f"{name} file inventory differs. Missing {missing}, extra {extra}")
        paths.update({name: directory / name for name in files})
    return paths


def build(source, output, workers):
    if output.exists():
        raise FileExistsError(output)
    release = read_json(PUBLISHED)
    indexed = read_json(source / INDEX)
    paths = {name: source / name for name in set(indexed["weight_map"].values())}
    tensors, metadata = read_tensors(indexed, paths)
    groups_to_write = list(groups(tensors, PAYLOAD_LIMIT))
    shards = {f"model-{i:05d}-of-{len(groups_to_write):05d}.safetensors": group
              for i, group in enumerate(groups_to_write, 1)}
    expected = {name for part in release["parts"].values() for name in part["files"]
                if name.endswith(".safetensors")}
    if shards.keys() != expected:
        raise ValueError("Expected the 30-shard po352 layout")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".po352-parts-", dir=output.parent) as temp:
        stage = Path(temp) / "model"
        jobs = []
        for part, info in release["parts"].items():
            destination = stage / part
            destination.mkdir(parents=True)
            for name in info["files"]:
                if name in shards:
                    jobs.append((destination / name, metadata, shards[name]))
                elif name != INDEX:
                    shutil.copyfile(source / name, destination / name)

        with ProcessPoolExecutor(workers) as pool:
            for filename in pool.map(write_shard, jobs):
                print(f"Wrote {filename}", flush=True)
        indexed["weight_map"] = {
            tensor.name: filename for filename, group in shards.items() for tensor in group
        }
        (stage / "part1" / INDEX).write_text(json.dumps(indexed, indent=2), encoding="utf-8")
        validate_structure(stage / "part1", stage / "part2", release)
        stage.rename(output)
    print(f"Prepared two model parts in {output}")


def validate_structure(part1, part2, release):
    paths = inventory(part1, part2, release)
    index = read_json(paths[INDEX])
    tensors, _ = read_tensors(index, paths)
    actual = {tensor.path.name for tensor in tensors}
    expected = {name for name in paths if name.endswith(".safetensors")}
    if actual != expected or len(tensors) != release["tensor_count"]:
        raise ValueError("Unexpected po352 tensor or shard count")
    return paths, index, tensors


def hash_tensor_pair(pair):
    original, rebuilt = pair
    if (original.dtype, original.shape, original.length) != (rebuilt.dtype, rebuilt.shape, rebuilt.length):
        return original.name
    digests = []
    for tensor in pair:
        with tensor.path.open("rb") as handle:
            digest = hashlib.sha256()
            for block in read_payload(handle, tensor):
                digest.update(block)
        digests.append(digest.digest())
    return None if digests[0] == digests[1] else original.name


def verify(part1, part2, source, workers):
    release = read_json(PUBLISHED)
    paths, index, tensors = validate_structure(part1, part2, release)
    if source is None:
        for part in release["parts"].values():
            for filename, entry in part["files"].items():
                path = paths[filename]
                if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
                    raise ValueError(f"{filename} differs from the Kaggle v1 reference")
        print(f"Verified {len(paths)} files against the Kaggle v1 checksums")
        return

    original_index = read_json(source / INDEX)
    original_paths = {name: source / name for name in set(original_index["weight_map"].values())}
    original_tensors, _ = read_tensors(original_index, original_paths)
    original = {tensor.name: tensor for tensor in original_tensors}
    if original.keys() != index["weight_map"].keys() or original_index["metadata"] != index["metadata"]:
        raise ValueError("The re-shard changed tensor names or index metadata")
    for filename, path in paths.items():
        if not filename.endswith(".safetensors") and filename != INDEX:
            if sha256(path) != sha256(source / filename):
                raise ValueError(f"The re-shard changed {filename}")
    pairs = ((original[tensor.name], tensor) for tensor in tensors)
    with ProcessPoolExecutor(workers) as pool:
        for mismatch in pool.map(hash_tensor_pair, pairs, chunksize=32):
            if mismatch is not None:
                raise ValueError(f"The re-shard changed tensor bytes for {mismatch}")
    weights_size = sum(tensor.length for tensor in tensors)
    print(f"Verified {len(tensors)} identical tensors and {weights_size:,} payload bytes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("build")
    make.add_argument("--source", type=Path, required=True)
    make.add_argument("--output", type=Path, required=True)
    make.add_argument("--workers", type=int, default=4)
    check = commands.add_parser("verify")
    check.add_argument("--part1", type=Path, required=True)
    check.add_argument("--part2", type=Path, required=True)
    check.add_argument("--source", type=Path, help="Compare with the locally pruned checkpoint")
    check.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "build":
        build(args.source, args.output, args.workers)
    else:
        verify(args.part1, args.part2, args.source, args.workers)


if __name__ == "__main__":
    main()
