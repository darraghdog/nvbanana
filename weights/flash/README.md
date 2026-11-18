# Qwen3.8-Flash-Next po352 Weight Preparation

The [27B + Flash-Next blend](../../inference/README.md) also uses a
Qwen3.8-Flash-Next checkpoint with 352 routed experts in each of its 48
MoE layers. The archived notebook's name includes `po358`. Its attached
weights are `po352`.

Start from the public
[REAP448 snapshot](https://huggingface.co/lee-chang-93/Qwen3.8-Flash-Next-NVFP4-REAP-k448/tree/8d565c90add22b39cf3632943f859d7a9a411b48).
Qwen trained Flash-Next, [NVIDIA quantized the routed experts to
NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4), and Lee
Chang applied REAP to retain 448 of the original 512 routed experts per
layer. See those model cards for the NVIDIA Open Model License and Qwen
Community License terms.

## Select and prune experts

We ranked experts within each layer by the sum of their normalized top-10
router weights across all routed tokens in ARC-AGI-2 training and
synthetic INSP16B traces. Expert selection frequency broke ties. The
[exact keep-list](keep_ours_352.json) records the 352 source expert IDs per
layer in the REAP448 numbering. Start with that list to reproduce the
submitted model.

The pruner remaps retained IDs to 0–351, slices the router rows, and
updates the model and quantization configs. Kept weights retain their
original NVFP4 bytes. Routing still selects ten experts per token, and the
separate MTP expert bank and PLE embedding are preserved.

Run from the repository root in a Python 3.12 environment. The pruner
uses CPU tensors. The original build used PyTorch 2.13.0+cu130 and
Safetensors 0.8.0. Allow disk space for the source, the rewritten
intermediate shards, and roughly 112 GB of packaged shards. The pruner
reuses untouched source shards on disk.

```bash
python -m pip install 'torch==2.13.0' 'safetensors==0.8.0' numpy huggingface_hub

hf download lee-chang-93/Qwen3.8-Flash-Next-NVFP4-REAP-k448 \
  --revision 8d565c90add22b39cf3632943f859d7a9a411b48 \
  --local-dir models/flash-reap448

python -m weights.flash.prune \
  --source models/flash-reap448 --output models/flash-po352 --workers 2
```

## Re-shard and verify the two parts

```bash
python -m weights.flash.package build \
  --source models/flash-po352 --output outputs/flash-po352-kaggle --workers 4

python -m weights.flash.package verify \
  --part1 outputs/flash-po352-kaggle/part1 \
  --part2 outputs/flash-po352-kaggle/part2 \
  --source models/flash-po352 --workers 4
```

The packager streams tensors in source file and offset order with a
4 billion byte payload limit per shard. Safetensors headers add a few MB
to the largest files. The release has 30 shards and about 111.37 GB of
tensor data. Part 1 contains shards 1–15, the full index, and model
metadata. Part 2 contains shards 16–30. The original packaging job
reopened every shard and matched all 207,385 tensor payloads by SHA-256
against the pruned model. The local command above checks the same
names, dtypes, shapes, and bytes.

The [release manifest](published.json) records both Kaggle v1 model
parts. The file names and byte lengths come from Kaggle. The SHA-256
digests come from the matching archived upload payloads. To check a
downloaded copy of both parts, pass their directories to
`weights.flash.package verify`. Use `--source` for a rebuild, which
writes a portable model README and keep-list provenance.

- [Kaggle po352 part 1, version 1](https://www.kaggle.com/models/darraghdog/qwen3-8-flash-next-nvfp4-reap-po352/transformers/nvfp4-po352-kaggle-4gb-part1/1)
- [Kaggle po352 part 2, version 1](https://www.kaggle.com/models/darraghdog/qwen3-8-flash-next-nvfp4-reap-po352/transformers/nvfp4-po352-kaggle-4gb-part2/1)

## Load for inference

The blend notebook symlinks both parts into one model directory. vLLM
reads the shards in place. The [inference manifest](../../manifests/inference.json)
and [versioned launch entry point](../../inference/flash/flash_kaggle_run.py)
record the vLLM runtime (nightly revision `29468dde` with the prefix
cache fix), TP2 × PP2 with expert parallelism, ModelOpt NVFP4, FP8 KV
cache, and CPU offload settings used on four L4 GPUs.

In the four-L4 replica, po352 raised KV-cache capacity from about 0.83M
to 1.63M tokens compared with REAP448 under the same serving recipe. The
re-sharded and standard-layout po352 models reported the same capacity.
