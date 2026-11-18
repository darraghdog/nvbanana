# NVBANANA solution to ARC-AGI-2 2026

This repository is being assembled to document and reproduce the NVBANANA submissions to the [ARC Prize 2026 competition on Kaggle](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2).

The [solution writeup](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/writeups/arc-agi-2-public-3rd-place-solution) describes four parts.

- Book-inspired synthetic puzzle generation
- Supervised fine-tuning of Qwen3.8-27B
- Checkpoint conversion for inference on four L4 GPUs
- Per-puzzle test-time fine-tuning and a Flash-Next blend

## Synthetic Data Generation

The [SDG guide](sdg/README.md) shows how book ideas became verified ARC puzzles, with a worked Hoffmann example and two Kaggle demos. The [book sourcing guide](sdg/sourcing/README.md) contains the example bibliography and prompts. The [dataset builder](sdg/dataset/README.md) turns accepted puzzles into four augmented SFT rows each.

![Book rules and ARC training summaries feed Stage 2/3 input generation, Stage 4 output verification, and the SFT dataset builder.](sdg/assets/sdg-stage-flow.png)

![Hoffmann’s counters puzzle becomes an ARC specification, tested generator code, and example input and output grids.](sdg/assets/hoffmann-book-to-arc.png)

*From a counters puzzle in Hoffmann’s Puzzles Old and New (1893) to a generated ARC-AGI-2 task.*

## Model Training

The text model is Qwen3.8-27B fine-tuned with NeMo RL and Megatron. The [training guide](training/README.md) builds the ARC-16 base and runs the reduced SFT recipe. The [production record](training/production.md) traces the dataset mixtures and checkpoints through step 188222.

## Weight Preparation

The [weight preparation guide](weights/README.md) exports step 188222, rebuilds the ARC training calibration prompts, and packages the mixed GPTQ weights and split TTFT configs used on Kaggle. The blend also uses Qwen3.8-Flash-Next pruned to 352 experts per layer.

## Kaggle Inference

The baseline uses per-puzzle LoRA test-time fine-tuning followed by SGLang sampling and voting. The blend also runs Flash-Next on the least-confident outputs.

- [Qwen3.8-27B baseline notebook](https://www.kaggle.com/code/darraghdog/fast-v18-insp4-to-16-ep1-6-ttft32-s48-symlink?scriptVersionId=361158513)
- [Qwen3.8-27B and Flash-Next blend notebook](https://www.kaggle.com/code/darraghdog/blend-v18-27b-text-po358-fnt07cs72)

The [inference instructions](inference/README.md) list the notebook snapshots and versioned dependencies.
