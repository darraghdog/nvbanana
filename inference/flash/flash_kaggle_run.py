#!/usr/bin/env python3
"""Run Flash-Next po352 on a set of ARC tasks inside a Kaggle 4xL4 notebook, next to the notebook's own environment.

Standard library only, so the notebook's own Python (the Kaggle image) can run it. Everything heavy runs in subprocesses with the
portable runtime from the dataset darraghdog/flash-vllm-runtime-r1026b (CPython 3.12 + vLLM nightly 29468dde + overlay +
PR #50551 with the keep fix, torch 2.13 cu130), so nothing in the notebook environment changes.

Steps: unpack the runtime to /kaggle/tmp/flash; write harness rows for the chosen tasks (expected_answer "" when there are no
solutions, which the harness scores as 0); start the vLLM server with the L4 serving recipe (TP2xPP2 + EP, NVFP4 via
Marlin, FP8 KV, bf16 Mamba state, align prefix caching, PLE + embed_tokens on CPU, FP8 indexer KV, native CPU KV offload,
decode checkpoints); run the harness at --seats; stop at --deadline; write answers.json (per task, per test input,
per sample: answer grid text and whether it was an auto answer, i.e. check() passed) next to the raw records.

The blend notebook runs it with --pools (rule A, see flash_rule_a.py) and --setting ov_l4_fnt07cs72_a, for example:

    python flash_kaggle_run.py --runtime /kaggle/input/.../flash-runtime-r1026.part-00 \
        --harness /kaggle/input/.../flash-harness-fnt07cs72-r1027 --model /kaggle/tmp/po352_nvfp4 \
        --challenges .../arc-agi_test_challenges.json --setting ov_l4_fnt07cs72_a \
        --keys <out>/keys.json --pools <out>/pools.json --out <out> --deadline <unix time>
"""

from __future__ import annotations

import argparse
import io
import json
import os
import signal
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

PORT = 8000


def log(msg: str) -> None:
    print(f"[flash {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class _Parts(io.RawIOBase):
    """The runtime tarball read back from its parts (flash-runtime-*.part-00, -01, ...), in order, as one stream. Kaggle
    unpacks an uploaded .tar.gz (and failed to create the dataset for the 4.3 GB one), so the runtime is stored as parts."""

    def __init__(self, paths):
        self.paths, self.f = list(paths), None

    def readable(self):
        return True

    def readinto(self, b):
        while True:
            if self.f is None:
                if not self.paths:
                    return 0
                self.f = open(self.paths.pop(0), "rb", buffering=0)
            n = self.f.readinto(b)
            if n:
                return n
            self.f.close()
            self.f = None


def unpack(runtime: Path, dest: Path) -> Path:
    """runtime: the tarball, or a directory (or any file in it) holding flash-runtime-*.part-NN pieces."""
    root = dest / "flash"
    if (root / "python" / "bin" / "python3.12").exists():
        log(f"runtime already unpacked at {root}")
        return root
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    folder = runtime if runtime.is_dir() else runtime.parent
    parts = sorted(folder.glob("flash-runtime-*.part-*"))
    if parts:
        with tarfile.open(fileobj=io.BufferedReader(_Parts(parts), 16 << 20), mode="r|gz") as tf:
            tf.extractall(dest)
    else:
        with tarfile.open(runtime) as tf:
            tf.extractall(dest)
    log(f"runtime unpacked in {time.time() - t0:.0f} s ({len(parts)} parts)")
    return root


def harness_dir(harness: Path, dest: Path) -> Path:
    """The harness tree (scripts/, tokenizer/): the dataset dir itself, its harness/ folder (Kaggle unpacks an uploaded
    harness.tar.gz into harness/), or harness.tar.gz unpacked under dest."""
    for cand in (harness, harness / "harness"):
        if (cand / "scripts").is_dir():
            return cand
    out = dest / "flash_harness"
    if not (out / "scripts").is_dir():
        with tarfile.open(harness / "harness.tar.gz") as tf:
            tf.extractall(out)
        log(f"harness unpacked to {out}")
    return out


def grid_text(grid) -> str:
    return "\n".join("".join(str(int(c)) for c in row) for row in grid)


def write_rows(challenges: Path, solutions: Path | None, task_ids: list[str] | None, out: Path) -> int:
    tasks = json.loads(challenges.read_text())
    sols = json.loads(solutions.read_text()) if solutions else {}
    keep = sorted(task_ids if task_ids is not None else tasks)
    n = 0
    with out.open("w") as f:
        for tid in keep:
            task = tasks[tid]
            train = [{"input": p["input"], "output": p["output"]} for p in task["train"]]
            for idx, test in enumerate(task["test"]):
                exp = grid_text(sols[tid][idx]) if tid in sols else ""
                f.write(json.dumps({"task_id": tid, "test_idx": idx, "n_tests": len(task["test"]), "train": train,
                                    "test_input": test["input"], "expected_answer": exp}) + "\n")
                n += 1
    return n


def server_cmd(py: Path, model: str, seats: int, maxlen: int, offload_gib: int, mem: float, load: str, load_threads: int,
               p2p: bool = False) -> list[str]:
    # load "prefetch": read the checkpoint into the page cache with load_threads parallel readers before the workers load it
    # (vLLM auto-enables this only on NFS/Lustre; /kaggle/input is FUSE, where one reader per file is slow).
    loadargs = ["--safetensors-load-strategy", load, "--safetensors-prefetch-num-threads", str(load_threads)] if load != "default" else []
    return [str(py), "-m", "vllm.entrypoints.openai.api_server", "--model", model, "--served-model-name", "qwen",
            "--host", "127.0.0.1", "--port", str(PORT), "--quantization", "modelopt_fp4", "--dtype", "bfloat16",
            "--engram-config", '{"cpu_offload": true}', "--tensor-parallel-size", "2", "--pipeline-parallel-size", "2",
            "--enable-expert-parallel", "--max-model-len", str(maxlen), "--gpu-memory-utilization", str(mem),
            "--max-num-seqs", str(seats), "--max-num-batched-tokens", "1024", "--kv-cache-dtype", "fp8_e4m3",
            "--mamba-ssm-cache-dtype", "bfloat16", "--enable-prefix-caching", "--mamba-cache-mode", "align",
            "--enable-chunked-prefill", *([] if p2p else ["--disable-custom-all-reduce"]),
            "--compilation-config", '{"cudagraph_mode": "FULL_DECODE_ONLY"}', "--generation-config", "vllm",
            "--mm-processor-cache-gb", "0", "--limit-mm-per-prompt", '{"image": 32}',
            "--mm-processor-kwargs", '{"size": {"shortest_edge": 1024, "longest_edge": 1048576}}',
            "--enable-scale-out", "--enable-prompt-tokens-details",
            *(["--kv-offloading-size", str(offload_gib), "--kv-offloading-backend", "native"] if offload_gib > 0 else []),
            "--enable-mamba-decode-checkpoint", "--prefix-cache-retention-interval", "0",
            "--cpu-offload-params", "embed_tokens", "--attention-config", '{"indexer_kv_dtype":"fp8"}'] + loadargs


def runtime_env(rt: Path, p2p: bool = False) -> dict:
    py_dir = rt / "python"
    cache = rt / "cache"
    for d in ("triton", "hf", "xdg", "flashinfer", "tmp"):
        (cache / d).mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "VIRTUAL_ENV", "CONDA"))}
    cuda_home = py_dir / "lib" / "python3.12" / "site-packages" / "nvidia" / "cu13"
    env.update(
        TMPDIR=str(cache / "tmp"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLLM_NO_USAGE_STATS="1",
        PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", HF_HOME=str(cache / "hf"), XDG_CACHE_HOME=str(cache / "xdg"),
        TRITON_CACHE_DIR=str(cache / "triton"), CUDA_HOME=str(cuda_home),
        PATH=f"{cuda_home / 'bin'}:{py_dir / 'bin'}:{env.get('PATH', '/usr/bin:/bin')}",
        VLLM_USE_FLASHINFER_SAMPLER="0", FLASHINFER_WORKSPACE_BASE=str(cache / "flashinfer"),
        VLLM_SPARSE_INDEXER_MAX_LOGITS_MB="128", PYTORCH_CUDA_ALLOC_CONF="", PYTORCH_ALLOC_CONF="",
        VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="0", VLLM_WORKER_MULTIPROC_METHOD="spawn", NCCL_CUMEM_ENABLE="0",
        NCCL_P2P_DISABLE="0" if p2p else "1", OMP_NUM_THREADS="1")
    return env


def wait_server(proc: subprocess.Popen, log_path: Path, limit_s: int) -> bool:
    t0 = time.time()
    while time.time() - t0 < limit_s:
        if proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5)
            log(f"server up after {time.time() - t0:.0f} s")
            for line in log_path.read_text(errors="ignore").splitlines():
                if "GPU KV cache size" in line or "Available KV cache memory" in line:
                    log("  " + line.split("]")[-1].strip()[:160])
            return True
        except Exception:
            time.sleep(15)
    return False


def host_line() -> str:
    """cgroup memory use / limit, /dev/shm use and major page faults so far (memory pressure shows as rising majfaults)."""
    def read(p):
        try:
            return Path(p).read_text().strip()
        except OSError:
            return ""
    cur, lim = read("/sys/fs/cgroup/memory.current"), read("/sys/fs/cgroup/memory.max")
    maj = next((l.split()[1] for l in read("/proc/vmstat").splitlines() if l.startswith("pgmajfault ")), "?")
    try:
        st = os.statvfs("/dev/shm"); shm = (st.f_blocks - st.f_bfree) * st.f_frsize / 2**30
    except OSError:
        shm = -1
    gib = lambda v: f"{int(v) / 2**30:.0f}" if v.isdigit() else v or "?"
    return f"mem {gib(cur)}/{gib(lim)} GiB, shm {shm:.0f} GiB, majfault {maj}"


def progress_line(out: Path, t0: float) -> str:
    """One console line for the Kaggle live log: harness progress ([n/N] conversations), records written, and the server's
    latest throughput / running / KV-use / prefix-hit status line."""
    conv = recs = ""
    try:
        conv = next((l.split("]")[0] + "]" for l in reversed((out / "client.log").read_text(errors="ignore").splitlines())
                     if l.startswith("[")), "")
    except OSError:
        pass
    try:
        recs = str(sum(1 for _ in (out / "shard0.jsonl").open()))
    except OSError:
        recs = "0"
    srv = ""
    try:
        with (out / "vllm-server.log").open("rb") as f:
            f.seek(max(0, f.seek(0, 2) - 65536))
            tail = f.read().decode(errors="ignore").splitlines()
        line = next((l for l in reversed(tail) if "Avg generation throughput" in l), "")
        keep = ("Avg generation throughput", "Running", "Waiting", "GPU KV cache usage", "Prefix cache hit rate", "External prefix")
        srv = ", ".join(p.strip() for p in line.split("]")[-1].split(",") if p.strip().startswith(keep))
    except OSError:
        pass
    return f"progress {(time.time() - t0) / 60:.0f} min: conversations {conv or '?'}, records {recs} | {srv} | {host_line()}"


def stop(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass


def collect(records: Path, out: Path) -> dict:
    """answers.json: {task_id: {test_idx: [{sample, answer, auto, correct}, ...]}} from the harness's per-pair records."""
    res: dict = {}
    if not records.exists():
        out.write_text("{}")
        return res
    for line in records.read_text().splitlines():
        r = json.loads(line)
        tid, idx = r["puzzle_id"].split("#")
        res.setdefault(tid, {}).setdefault(idx, []).append(
            {"sample": r["sample"], "answer": r.get("answer"), "auto": bool(r.get("auto_answer")), "correct": r.get("correct"),
             "status": r.get("status")})
    out.write_text(json.dumps(res))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runtime", type=Path, required=True, help="flash-runtime-*.tar.gz from the runtime dataset")
    ap.add_argument("--harness", type=Path, required=True, help="harness dataset dir (scripts/, tokenizer/, settings)")
    ap.add_argument("--model", required=True, help="REAP448 model dir")
    ap.add_argument("--challenges", type=Path, required=True)
    ap.add_argument("--solutions", type=Path, default=None, help="eval only: scores the run; omit in the rerun")
    ap.add_argument("--keys", type=Path, default=None, help="JSON task list (list or {'keys': [...]}); default all tasks")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work", type=Path, default=Path("/kaggle/tmp"))
    ap.add_argument("--setting", default="ov_l4_fnt07cs48_a")
    ap.add_argument("--seats", type=int, default=22)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--maxlen", type=int, default=81920)
    ap.add_argument("--offload-gib", type=int, default=48)
    ap.add_argument("--mem", type=float, default=0.96)
    ap.add_argument("--deadline", type=float, default=0.0, help="unix time: stop the client here and keep what finished")
    ap.add_argument("--boot-limit", type=int, default=3600)
    ap.add_argument("--progress-every", type=int, default=300, help="seconds between progress lines in the console log")
    ap.add_argument("--p2p", action="store_true", help="NCCL P2P + custom all-reduce on (Kaggle 4xL4 supports P2P)")
    ap.add_argument("--client-minutes", type=float, default=0, help="stop the client this many minutes after the server is up (debug A/B)")
    ap.add_argument("--load", default="prefetch", choices=["prefetch", "eager", "lazy", "default"], help="safetensors load strategy")
    ap.add_argument("--load-threads", type=int, default=16, help="parallel readers for --load prefetch")
    ap.add_argument("--pools", type=Path, default=None, help="rule A: base-model candidate pools {task_i: [[grid, votes], ...]}; "
                    "runs flash_rule_a.py (per-task requeue until settled) instead of a fixed --samples harness run")
    ap.add_argument("--max-attempts", type=int, default=16, help="rule A: attempts per task at most")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rt = unpack(args.runtime, args.work)
    args.harness = harness_dir(args.harness, args.work)
    py, sbpy = rt / "python" / "bin" / "python3.12", rt / "sbpy"
    env = runtime_env(rt, args.p2p)

    keys = None
    if args.keys:
        k = json.loads(args.keys.read_text())
        keys = k["keys"] if isinstance(k, dict) else k
    rows = args.out / "rows.jsonl"
    log(f"{write_rows(args.challenges, args.solutions, keys, rows)} test inputs from {len(keys) if keys else 'all'} tasks")

    server = client = None
    try:
        slog = args.out / "vllm-server.log"
        server = subprocess.Popen(server_cmd(py, args.model, args.seats, args.maxlen, args.offload_gib, args.mem, args.load, args.load_threads, args.p2p),
                                  env=env, stdout=slog.open("w"), stderr=subprocess.STDOUT, start_new_session=True)
        if not wait_server(server, slog, args.boot_limit):
            log("SERVER_FAILED; last lines:\n" + "\n".join(slog.read_text(errors="ignore").splitlines()[-15:]))
            sys.exit(1)
        cenv = dict(env, PYTHONPATH=str(args.harness))
        cmd = [str(py), "-m", "scripts.grpo_arc2_code.image_multiturn_eval", "--base-url", f"http://127.0.0.1:{PORT}/v1",
               "--tokenizer", str(args.harness / "tokenizer"), "--data", str(rows), "--out", str(args.out / "shard0.jsonl"),
               "--setting", args.setting, "--shard", "0", "--nshards", "1", "--samples", str(args.samples),
               "--concurrency", str(args.seats), "--sandbox-workers", "24", "--python", str(sbpy), "--probe",
               "--server-tokenizer", args.model]
        if args.pools:   # rule A: same server, harness and records; per-task requeue instead of a fixed sample count
            cmd = [str(py), str(Path(__file__).resolve().parent / "flash_rule_a.py"), "--base-url", f"http://127.0.0.1:{PORT}/v1",
                   "--tokenizer", str(args.harness / "tokenizer"), "--data", str(rows), "--out", str(args.out / "shard0.jsonl"),
                   "--setting", args.setting, "--seats", str(args.seats), "--sandbox-workers", "24", "--python", str(sbpy),
                   "--server-tokenizer", args.model, "--pools", str(args.pools), "--keys", str(args.keys),
                   "--deadline", str(args.deadline), "--max-attempts", str(args.max_attempts)]
        t0 = time.time()
        client = subprocess.Popen(cmd, cwd=str(args.harness), env=cenv, stdout=(args.out / "client.log").open("w"),
                                  stderr=subprocess.STDOUT, start_new_session=True)
        last_progress = 0.0
        while client.poll() is None:
            if args.client_minutes and time.time() - t0 > args.client_minutes * 60:
                log(f"client time limit ({args.client_minutes} min) reached: stopping the client")
                stop(client)
                break
            if args.deadline and time.time() > args.deadline:
                log("deadline reached: stopping the client; finished attempts are kept")
                stop(client)
                break
            if time.time() - last_progress >= args.progress_every:
                last_progress = time.time()
                log(progress_line(args.out, t0))
            time.sleep(30)
        log(f"client done after {(time.time() - t0) / 60:.1f} min, rc {client.returncode}")
    finally:
        stop(client)
        stop(server)
    res = collect(args.out / "shard0.jsonl", args.out / "answers.json")
    n = sum(len(v) for t in res.values() for v in t.values())
    log(f"answers.json: {len(res)} tasks, {n} attempts")
    if args.solutions and res:
        right = sum(any(a.get("correct") for a in v) for t in res.values() for v in t.values())
        log(f"eval: {right} of {sum(len(t) for t in res.values())} test inputs right (any sample)")


if __name__ == "__main__":
    main()
