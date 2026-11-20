# ARC-16 Supervised Fine-Tuning

The reduced recipe prepares the original Qwen3.8-27B weights as a text
model, converts the Hoffmann dataset to Hugging Face format, and runs one
full-parameter BF16 update with NeMo RL and Megatron. The sample contains
four augmented rows from one accepted puzzle. The
[production checkpoint record](production.md) gives the original data
mixtures and resume chain.

## Prepare Qwen3.8-27B ARC-16

Download the [pinned Qwen source](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0)
as a local Hugging Face snapshot. Run from the repository root.

```bash
hf download Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --local-dir models/Qwen3.8-27B

python3 -m training.prepare_model --source models/Qwen3.8-27B --dry-run
python3 -m training.prepare_model --source models/Qwen3.8-27B \
  --output models/Qwen3.8-27B-ARC16
```

The builder checks the source revision against its config, index, and
weight-shard checksums. It drops the vision encoder and multi-token
prediction weights, then selects the same 16 source rows in both the input
embedding and the untied output head. Both output matrices have shape
`[16, 5120]` in BF16. The build streams the shards on CPU, writes a
Safetensors index, and records output checksums in a conversion manifest.
Allocate about 105 GB of disk space for the source and prepared model
together.

The [model spec](arc16/model_spec.json) records the fixed source ID mapping.
The compact IDs represent digits 0–9, newline, user, assistant, padding,
message start, and message end, in that order. The bundled tokenizer and
chat template are the files used by the production base, verified against
its conversion manifest. The 16-token vocabulary follows the
[NVARC and ARChitects tokenizer](https://github.com/1ytic/NVARC/tree/846d0198efa752534594e321fc3289fc0a06c657/ARChitects).

## Build the smoke dataset

```bash
python3 -m sdg.dataset.build_dataset sdg/dataset/examples/hoffmann-rd1-20854.jsonl \
  --output-dir outputs/hoffmann-sft
python3 -m sdg.dataset.export_hf outputs/hoffmann-sft \
  --output-dir outputs/hoffmann-sft-hf
```

The [dataset builder](../sdg/dataset/README.md) accepts Stage 4 output.
Use its `--validation-puzzles` option on a larger corpus to assign whole
generated puzzles to a validation split.

## Run NeMo RL SFT

Use a NeMo RL 0.7.0 environment with CUDA, Megatron Core, Megatron Bridge,
Transformer Engine, and the matching Transformers release. Install the
Megatron Bridge worker dependencies, including NVIDIA ModelOpt, in the
training environment. The
[runtime pins](runtime.json) identify the NeMo RL and Megatron Bridge
source checkouts and Transformers version. Make both pinned Git
checkouts importable from the training environment. The driver checks
the revisions of the imported source before it loads data. The Megatron
Bridge pin is the submodule recorded by this NeMo RL revision.

```bash
git clone --branch r0.7.0 https://github.com/NVIDIA-NeMo/RL.git /path/to/NeMo-RL
git -C /path/to/NeMo-RL checkout 87f1dfb1f6e050ec1d8f9dc3a93dbe98ca763280
bridge_path=3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
git -C /path/to/NeMo-RL submodule update --init --recursive "$bridge_path"
bridge_root="/path/to/NeMo-RL/$bridge_path"
export PYTHONPATH="/path/to/NeMo-RL:$bridge_root/src:$bridge_root/3rdparty/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}"
```

[NeMo RL installation](https://docs.nvidia.com/nemo/rl/latest/about/installation.html) ·
[NeMo RL SFT guide](https://docs.nvidia.com/nemo/rl/latest/guides/sft.html)

The [smoke config](sft_smoke.yaml) uses one node with four GPUs, TP4,
BF16, and a global batch of four. Run inside a compatible allocation. Set
`PYTHON` to the runtime's interpreter if needed.

```bash
export NRL_MEGATRON_CHECKPOINT_DIR="$PWD/outputs/arc16-megatron-import"

training/run_sft.sh --model models/Qwen3.8-27B-ARC16 \
  --data outputs/hoffmann-sft-hf --output outputs/hoffmann-training \
  --preflight

training/run_sft.sh --model models/Qwen3.8-27B-ARC16 \
  --data outputs/hoffmann-sft-hf --output outputs/hoffmann-training
```

Preflight checks the model metadata and every sample with NeMo RL's
message formatter. The driver restores the two compact role IDs when the
incremental formatter omits them and compares the complete token stream
to the digit-grid chat format. The smoke config disables checkpoint
writes, validation, and external logging. For a longer run, pass a
separate `--config` with the training schedule, hardware topology,
validation settings, and checkpoint interval for your data.

## Reuse the compiler cache across jobs

For chained Slurm runs, add this optional top-level block to the SFT YAML
passed with `--config`. The [cache helper](compile_cache.py) seeds and
publishes the snapshots.

```yaml
compile_cache:
  enabled: true
  key: qwen38-27b-sm103-tp4-pp1-bf16-pack4k-v1
  local_base: /raid/scratch/my-user/torch-compile
  snapshot_base: /shared/my-user/torch-compile-snapshots
  compile_threads: 4
```

Mount `local_base` on fast node-local storage at the same path on every
GPU node, and `snapshot_base` on a shared filesystem writable by the
training user. Use a trusted snapshot directory. Choose a new key when
the model, runtime, GPU architecture, parallelism, precision, or token
shapes change.

The runner configures TorchInductor, Triton, XDG, and CUDA cache paths
before starting Ray. It checks for a completed snapshot and restores it
on each GPU node before building the model. On a cold run, one node
publishes a checksum-verified snapshot after the first training step.
Kernels absent from that node's snapshot compile on demand. The default
smoke config runs with the cache disabled.

Look for `compile_cache_seed=cold` followed by
`compile_cache_publish=complete` on the first run. A later job using the
same key reports `compile_cache_seed=restored` or
`compile_cache_seed=already-current`.

## 16K throughput profile

The [throughput profile](sft_throughput_16k.yaml) takes the same reduced
dataset through the settings of the initial production SFT phase
(steps 0–72,333, see the [production record](production.md)). It uses
BF16, TP4/PP2/CP1 with sequence parallelism, fused RoPE, full activation
recomputation, and overlapping gradient reduction and parameter gather.
It packs into 16,384-token bins with a 12,288-token per-sequence cap.
The production run used 32K bins and switched to 16K at step 3,400.
We checked these settings against the saved production checkpoint config.

The [NeMo RL patch](throughput_optimizations.patch) carries the two runtime changes
enabled by that run. The training dataloader pins host memory, uses four
workers, and snapshots its state every step. The Megatron worker releases
unused CUDA allocator blocks before gradient finalization. The production
driver also cached the seeded Hugging Face shuffle index and reused a
topology-matched HF-to-Megatron conversion.

Prepare a separate runtime worktree from the NeMo RL checkout described
above. Keep the pinned Megatron Bridge paths on `PYTHONPATH`.

```bash
patched_rl=/path/to/NeMo-RL-throughput
git -C /path/to/NeMo-RL worktree add --detach "$patched_rl" 87f1dfb1f6e050ec1d8f9dc3a93dbe98ca763280
git -C "$patched_rl" apply "$PWD/training/throughput_optimizations.patch"
export PYTHONPATH="$patched_rl${PYTHONPATH:+:$PYTHONPATH}"

export NVBANANA_COMPILE_CACHE_KEY=qwen38-27b-r070-sm90-tp4-pp2-bf16-pack16k-seq12k-v1
export NVBANANA_COMPILE_CACHE_LOCAL=/raid/scratch/my-user/torch-compile
export NVBANANA_COMPILE_CACHE_SHARED=/shared/my-user/torch-compile-snapshots
export ARC2_DATASET_SHUFFLE_CACHE_DIR=/shared/my-user/arc-shuffle
export NRL_MEGATRON_CHECKPOINT_DIR=/shared/my-user/qwen38-27b-tp4-pp2-import

training/run_sft.sh --config training/sft_throughput_16k.yaml \
  --model models/Qwen3.8-27B-ARC16 --data outputs/hoffmann-sft-hf \
  --output outputs/hoffmann-throughput
```

The example uses one eight-GPU node and one optimizer update. Larger runs
can set the node count, batch, scheduler, and checkpoint cadence in a
config that inherits this profile. Production used NVIDIA Resiliency
Extension 0.6.0 at `6c5f2a13c7688d92a7ac7ee6e464721eb8b7345d` for
checkpointing. The [production record](production.md) describes the later
MXFP8 continuation and the optimizer-preserving decay fork.

## Verification

A full build from the pinned Qwen snapshot produced the same tensor
inventory and byte-identical ARC-16 input and output matrices as the
production base. The pinned NeMo RL formatter preserved all grid cells
and role markers in the four Hoffmann rows. The smoke config completed
one BF16 SFT update on four GB300 GPUs with loss 1.2720.

Use a short node-local `RAY_TMPDIR` if overriding Ray's temporary
directory. Ray uses this path for its Unix sockets.

The throughput profile completed one update cold and one after restoring the
checksum-verified snapshot into a fresh local cache directory. Both
reported loss 1.2759. On eight H100s, the logged step time was 148.92 s
for the cold run including publication, and 28.31 s for the warm run.
The worker logs confirmed the pre-finalization CUDA cache release.
