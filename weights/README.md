# Qwen3.8-27B Weight Preparation

Both [submission notebooks](../inference/README.md) use the [mixed GPTQ
weights for step 188222](https://www.kaggle.com/datasets/darraghdog/qwen38-27b-src11plus16abcd-step188222-gptq-mixed/versions/1).
The dataset contains four weight shards, the ARC-16 tokenizer, and two
32-layer views for test-time fine-tuning. Start with the Kaggle weights
to reproduce inference. Rebuilding the dataset requires the Megatron
checkpoint identified in the [production training record](../training/production.md).

## Export the checkpoint

Run from the repository root in the [pinned NeMo RL / Megatron
environment](../training/README.md#run-nemo-rl-sft). Use the ARC-16 base
built by [training.prepare_model](../training/README.md#prepare-qwen38-27b-arc-16)
and the saved Megatron step directory. The exporter checks the step,
tokenizer, compact embedding and output head, and BF16 tensor inventory.

```bash
python -m weights.export_hf \
  --step-dir checkpoints/step_188222 \
  --source-artifact models/Qwen3.8-27B-ARC16 \
  --output models/step188222-bf16
```

## Build calibration prompts and quantize

Use a separate CUDA 13 environment with PyTorch `2.14.1+cu130`. Install
the [GPTQ runtime versions](requirements.txt) there. The script checks
these versions before loading the checkpoint.

```bash
python -m pip install -r weights/requirements.txt

python -m weights.calibration \
  --challenges data/arc-agi_training_challenges.json \
  --solutions data/arc-agi_training_solutions.json \
  --output outputs/gptq-calibration

python -m weights.quantize \
  --model models/step188222-bf16 \
  --calibration outputs/gptq-calibration \
  --output models/step188222-gptq
```

The calibration builder selects 512 tasks from the competition's ARC-AGI-2
**training** split using seed 7. It applies a color permutation and
shuffles the demonstrations, then formats each prompt with the first
query and its answer in the ARC-16 vocabulary. It verifies the training
input checksums and task IDs and truncates each row to 6,144 tokens.

The GPTQ recipe uses 4-bit weights for the MLP projections and 8-bit
weights for the attention and gated delta network projections. Both use
symmetric groups of 128, static activation ordering, and BF16 activations.
The embeddings, output head, and smaller gated delta network inputs and
convolution keep their BF16 weights. Calibration runs layer by layer with
dampening 0.01. The compressed-tensors export is the sampler's input.

## Stage for Kaggle inference

```bash
package=outputs/qwen38-27b-src11plus16abcd-step188222-gptq-mixed

python -m weights.package build \
  --source models/step188222-gptq \
  --output "$package"

python -m weights.package verify "$package" --published
```

The packager streams the single compressed Safetensors export into four
shards with at most 4 GiB of tensor data per shard. It writes the root
weight index and `PROVENANCE.json`, which the notebooks use to locate the
dataset. Keep the dataset name above when attaching a rebuild to either
archived notebook.

The `bottom/` and `top/` directories each contain a 32-layer config, a
filtered index, and a GPTQ source marker. The notebooks link both halves
to the same four shards and convert their weights to block-128 FP8 at
load time for TTFT.

Use `verify --published --metadata-only` to check the configs and indexes
against [the version 1 reference](published.json) when only the dataset
metadata is available. Full verification also opens each shard header
and checks its tensor inventory and payload size. The reference checks
the package layout and config.
