#!/usr/bin/env python3
"""Seed and publish a node-local torch compiler cache shared across chained jobs."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import socket
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from training.prepare_model import sha256


CACHE_COMPONENTS = ("torchinductor", "triton", "xdg", "cuda")
ARCHIVE_NAME = "cache.tar"
CHECKSUM_NAME = "cache.tar.sha256"
MANIFEST_NAME = "manifest.json"
COMPLETE_NAME = "COMPLETE"


@dataclass(frozen=True)
class CacheSettings:
    local_root: Path
    snapshot_dir: Path
    key: str
    compile_threads: int

    def environment(self) -> dict[str, str]:
        return {
            "TORCHINDUCTOR_CACHE_DIR": str(self.local_root / "torchinductor"),
            "TRITON_CACHE_DIR": str(self.local_root / "triton"),
            "XDG_CACHE_HOME": str(self.local_root / "xdg"),
            "CUDA_CACHE_PATH": str(self.local_root / "cuda"),
            "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
            "TORCHINDUCTOR_AUTOGRAD_CACHE": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": str(self.compile_threads),
        }


def cache_settings(config: dict | None) -> CacheSettings | None:
    if config is None:
        return None
    if not isinstance(config, dict):
        raise ValueError("compile_cache must be a mapping")
    if type(config.get("enabled", False)) is not bool:
        raise ValueError("compile_cache.enabled must be a boolean")
    if not config.get("enabled"):
        return None
    unknown = config.keys() - {"enabled", "key", "local_base", "snapshot_base", "compile_threads"}
    if unknown:
        raise ValueError(f"Unknown compile_cache options: {sorted(unknown)}")
    key = config.get("key")
    if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", key):
        raise ValueError("compile_cache.key must be a single path component starting with a letter or digit")
    folders = []
    for name in ("local_base", "snapshot_base"):
        value = config.get(name)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError(f"compile_cache.{name} must be an absolute path")
        folders.append(Path(value).resolve() / key)
    local, snapshot = folders
    if local == snapshot or local.is_relative_to(snapshot) or snapshot.is_relative_to(local):
        raise ValueError("compile_cache local and shared directories must be separate")
    threads = config.get("compile_threads", 4)
    if type(threads) is not int or threads < 1:
        raise ValueError("compile_cache.compile_threads must be a positive integer")
    return CacheSettings(local, snapshot, key, threads)


def ensure_local_components(local_root: Path) -> None:
    local_root.mkdir(parents=True, exist_ok=True)
    for component in CACHE_COMPONENTS:
        (local_root / component).mkdir(parents=True, exist_ok=True)


def read_snapshot(snapshot_dir: Path, cache_key: str) -> tuple[Path, str]:
    complete_path = snapshot_dir / COMPLETE_NAME
    archive_path = snapshot_dir / ARCHIVE_NAME
    checksum_path = snapshot_dir / CHECKSUM_NAME
    manifest_path = snapshot_dir / MANIFEST_NAME
    missing = [
        path.name
        for path in (complete_path, archive_path, checksum_path, manifest_path)
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(
            f"Incomplete compiler-cache snapshot {snapshot_dir}: missing={missing}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("cache_key") != cache_key:
        raise RuntimeError(
            "Compiler-cache snapshot key mismatch: "
            f"expected={cache_key!r}, actual={manifest.get('cache_key')!r}"
        )
    expected_checksum = checksum_path.read_text(encoding="utf-8").partition(" ")[0].strip()
    actual_checksum = sha256(archive_path)
    if actual_checksum != expected_checksum:
        raise RuntimeError(
            "Compiler-cache snapshot checksum mismatch: "
            f"expected={expected_checksum}, actual={actual_checksum}"
        )
    if (manifest.get("archive_sha256") != actual_checksum
            or complete_path.read_text(encoding="utf-8").strip() != actual_checksum):
        raise RuntimeError("Compiler-cache manifest archive checksum mismatch")
    return archive_path, actual_checksum


def seed(local_root: Path, snapshot_dir: Path, cache_key: str) -> None:
    ensure_local_components(local_root)
    lock_path = local_root.parent / f".{local_root.name}.seed.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if not snapshot_dir.exists():
            print(
                f">>> compile_cache_seed=cold node={socket.gethostname()} "
                f"local={local_root} snapshot={snapshot_dir}",
                flush=True,
            )
            return

        archive_path, archive_checksum = read_snapshot(snapshot_dir, cache_key)
        local_marker = local_root / ".arc2_snapshot_sha256"
        if (
            local_marker.is_file()
            and local_marker.read_text(encoding="utf-8").strip()
            == archive_checksum
        ):
            print(
                f">>> compile_cache_seed=already-current "
                f"node={socket.gethostname()} local={local_root}",
                flush=True,
            )
            return

        with tarfile.open(archive_path, "r") as archive:
            archive.extractall(local_root, filter="data")
        marker_tmp = local_root / f".arc2_snapshot_sha256.{os.getpid()}.tmp"
        marker_tmp.write_text(archive_checksum + "\n", encoding="utf-8")
        marker_tmp.replace(local_marker)
        print(
            f">>> compile_cache_seed=restored node={socket.gethostname()} "
            f"local={local_root} sha256={archive_checksum}",
            flush=True,
        )


def publish(local_root: Path, snapshot_dir: Path, cache_key: str) -> None:
    ensure_local_components(local_root)
    if (snapshot_dir / COMPLETE_NAME).is_file():
        read_snapshot(snapshot_dir, cache_key)
        print(
            f">>> compile_cache_publish=already-complete "
            f"snapshot={snapshot_dir}",
            flush=True,
        )
        return
    if snapshot_dir.exists():
        raise RuntimeError(
            "Refusing to replace incomplete compiler-cache snapshot: "
            f"{snapshot_dir}"
        )

    snapshot_dir.parent.mkdir(parents=True, exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID", "nojob")
    with tempfile.TemporaryDirectory(
        dir=snapshot_dir.parent, prefix=f".{snapshot_dir.name}.incoming-",
    ) as temporary:
        staging_dir = Path(temporary) / "snapshot"
        staging_dir.mkdir()
        archive_path = staging_dir / ARCHIVE_NAME
        with tarfile.open(archive_path, "w") as archive:
            for component in CACHE_COMPONENTS:
                component_path = local_root / component
                if component_path.is_dir():
                    archive.add(component_path, arcname=component, recursive=True)

        archive_bytes = archive_path.stat().st_size
        archive_checksum = sha256(archive_path)
        (staging_dir / CHECKSUM_NAME).write_text(
            f"{archive_checksum}  {ARCHIVE_NAME}\n", encoding="utf-8"
        )
        manifest = {
            "format": "arc2-torch-compile-cache-v1",
            "cache_key": cache_key,
            "archive": ARCHIVE_NAME,
            "archive_bytes": archive_bytes,
            "archive_sha256": archive_checksum,
            "components": list(CACHE_COMPONENTS),
            "created_at": datetime.now(UTC).isoformat(),
            "source_host": socket.gethostname(),
            "slurm_job_id": job_id,
        }
        (staging_dir / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (staging_dir / COMPLETE_NAME).write_text(
            archive_checksum + "\n", encoding="utf-8"
        )
        try:
            staging_dir.rename(snapshot_dir)
        except OSError:
            if (snapshot_dir / COMPLETE_NAME).is_file():
                read_snapshot(snapshot_dir, cache_key)
                print(
                    f">>> compile_cache_publish=lost-race "
                    f"snapshot={snapshot_dir}",
                    flush=True,
                )
                return
            raise
    print(
        f">>> compile_cache_publish=complete node={socket.gethostname()} "
        f"snapshot={snapshot_dir} bytes={archive_bytes} "
        f"sha256={archive_checksum}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("seed", "publish"))
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--cache-key", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.local_root.is_absolute():
        raise ValueError("--local-root must be absolute")
    if not args.snapshot_dir.is_absolute():
        raise ValueError("--snapshot-dir must be absolute")
    if not args.cache_key:
        raise ValueError("--cache-key must be nonempty")
    if args.mode == "seed":
        seed(args.local_root, args.snapshot_dir, args.cache_key)
    else:
        publish(args.local_root, args.snapshot_dir, args.cache_key)


if __name__ == "__main__":
    main()
