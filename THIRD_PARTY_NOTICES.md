# Third-Party Notices

The repository's own work is licensed under [CC BY 4.0](LICENSE). The
components below keep their own terms. Dependencies installed separately
(Megatron Bridge, Megatron-LM, Transformers, SGLang wheels, the vLLM
runtime and others) are not redistributed in this repository and are
governed by their own licenses.

## Models

| Component | Used for | License |
| --- | --- | --- |
| [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | Base of the fine-tuned model. The ARC-16 tokenizer files in `training/arc16/` and the released GPTQ weights are derived from it. | Apache-2.0 |
| [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) | Upstream of the Flash-Next checkpoint used by the blend. | NVIDIA Open Model License, with the Qwen Community License 1.0 for the base model |
| [lee-chang-93/Qwen3.8-Flash-Next-NVFP4-REAP-k448](https://huggingface.co/lee-chang-93/Qwen3.8-Flash-Next-NVFP4-REAP-k448) at `8d565c90add22b39cf3632943f859d7a9a411b48` | Source of the po352 pruned checkpoint (`weights/flash/`). The po352 weights are a derivative and carry the same terms. | NVIDIA Open Model License and Qwen Community License 1.0 |
| GPT-OSS-120B | Generated Stage 2/3 and Stage 4 code and grids, including the Hoffmann example. | Apache-2.0 |
| Google Gemini 3 Flash Preview and Gemini 3.1 Pro Preview | Generated the page screening results, the 240 book rule specs in `sdg/sourcing/inspiration_outputs.jsonl` and the 24 summaries in `sdg/mixing/training_summaries.jsonl`. | Model output, provided subject to the [Gemini API terms](https://ai.google.dev/gemini-api/terms) |

## Code

| Component | Used for | License |
| --- | --- | --- |
| [NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL) at `87f1dfb1f6e050ec1d8f9dc3a93dbe98ca763280` | Training framework. `training/throughput_optimizations.patch` modifies two of its files; the patch header lists the changes. | Apache-2.0 |
| [The ARChitects, ARC Prize 2024](https://github.com/da-fr/arc-prize-2024) (Daniel Franzen and Jan Disselhoff) | The puzzle loader, augmentation, formatter and decoder code embedded in the inference notebooks (`arc_loader.py`, `arc_decoder.py`) is adapted from this code. Modified by NVARC and by the NVBANANA team: Qwen ARC-16 formatting, test-challenge handling, and the selection scores used by the notebooks. Each embedded file starts with an attribution header. | Apache-2.0. [License](licenses/ARChitects/LICENSE.txt) and [NOTICE](licenses/ARChitects/NOTICE) |
| [NVARC](https://github.com/1ytic/NVARC/tree/846d0198efa752534594e321fc3289fc0a06c657) (ARC Prize 2025 solution) | NVARC-specific material: the `getter_kgmon`/`score_kgmon` selection score in `arc_decoder.py`, the 16-token tokenizer approach, and the structure of the summary and mixing prompts, which are rewritten rather than copied. | No license is published upstream; used with attribution to the NVARC authors |
| [SGLang](https://github.com/sgl-project/sglang) | The notebooks patch SGLang's Qwen3.5 LoRA support at runtime. | Apache-2.0 |
| [vLLM](https://github.com/vllm-project/vllm) | Serves Flash-Next in the blend. The runtime is distributed as a Kaggle dataset, not in this repository. | Apache-2.0 |

## Data

| Component | Used for | License |
| --- | --- | --- |
| [ARC-AGI-2](https://github.com/arcprize/ARC-AGI-2/tree/f3283f727488ad98fe575ea6a5ac981e4a188e49) training set | Training task IDs (`sdg/dataset/arc_agi2_train_ids.txt`), the training-task summaries, GPTQ calibration prompts, and the anchors of generated puzzles. The evaluation set was not used for training, generation or calibration. | Apache-2.0 |
| [MINI-ARC](https://github.com/KSB21ST/MINI-ARC) | Training data (not redistributed). | Apache-2.0 |
| [ConceptARC](https://github.com/victorvikram/ConceptARC) | Training data (not redistributed). | MIT |
| [RE-ARC](https://github.com/michaelhodel/re-arc) | Training data (not redistributed). | MIT |

## Books and images

| Component | Used for | License |
| --- | --- | --- |
| Hoffmann, *Puzzles Old and New* (1893); Dudeney, *Amusements in Mathematics* (1917) and *The Canterbury Puzzles*; and the other editions listed in [`sdg/sourcing/README.md`](sdg/sourcing/README.md) | Source ideas for the book specs. `sdg/assets/hoffmann-book-to-arc.png` includes a crop of a scanned page from the [Internet Archive copy](https://archive.org/details/puzzlesoldnew00hoff). | Public domain |
| Author portraits in `sdg/assets/book-inspiration-collage.jpg` | Illustration. | Public domain, Public Domain Mark or CC0, per [portrait credits](sdg/assets/portrait-credits.md) |
