# Kaggle Inference

The 27B notebooks use split LoRA test-time fine-tuning followed by SGLang sampling and voting. The blend writes the base submission first, then runs Flash-Next on the least-confident outputs and updates the submission with verified answers.

## Notebook snapshots

| Kaggle submission ID | Notebook snapshot | Script version |
| --- | --- | --- |
| `57535591` | [Qwen3.8-27B baseline](notebooks/baseline/fast-v18-insp4-to-16-ep1-6-ttft32-s48-symlink.ipynb) | `361158513` in the original run record. Live and archived cell sources matched. |
| `57744045` | [Qwen3.8-27B + Flash-Next](notebooks/blend/blend-v18-27b-text-po358-fnt07cs72.ipynb) | Historical Kaggle pull. Exact submitted script version still to confirm. |

The files preserve the notebook code and metadata, including historical attachments. Use the [dependency manifest](../manifests/inference.json) for notebook source evidence, asset versions, image digests, and access checks.

Verify the archived files from the repo root.

```bash
shasum -a 256 -c manifests/inference.sha256
```

For this release, internal names and run tags were removed from comments and docstrings in the two notebooks and the two Flash entry points, and third-party attribution headers were added to the embedded `arc_loader.py` and `arc_decoder.py`. Their code is unchanged. [`inference-submitted.sha256`](../manifests/inference-submitted.sha256) keeps the checksums of the files as submitted and as stored in the Kaggle harness dataset.

## Inputs

Use four NVIDIA L4 GPUs and disable internet for inference. Attach the ARC Prize 2026 competition data through Kaggle. The notebooks write `/kaggle/working/submission.json`.

Both notebooks use the [TTFT runtime v3](https://www.kaggle.com/datasets/darraghdog/arc2-ttft27b-runtime/versions/3) and the [step-188222 GPTQ dataset v1](https://www.kaggle.com/datasets/darraghdog/qwen38-27b-src11plus16abcd-step188222-gptq-mixed/versions/1). The GPTQ dataset contains the checkpoint, tokenizer and split-half configs. See the [weight preparation guide](../weights/README.md) for the export, calibration and packaging workflow. Check that the named weight shards are present after download. The bulk GPTQ download returned an empty ZIP during the access check.

The blend adds the [Flash harness v2](https://www.kaggle.com/datasets/darraghdog/flash-harness-fnt07cs72-r1027/versions/2), the [vLLM runtime v1](https://www.kaggle.com/datasets/darraghdog/flash-vllm-runtime-r1026b/versions/1), and both [po352 model parts at version 1](../weights/flash/README.md#re-shard-and-verify-the-two-parts). The [Flash launch and rule-A entry points](flash/) are copied from the versioned harness. The notebook still needs the accompanying harness tree and tokenizer from that dataset. The [upstream runtime checksums](../manifests/flash-runtime-r1026.sha256) cover the five parts and the reconstructed archive.

The notebook metadata records dataset slugs. The manifest gives the observed dataset versions whose timestamps are consistent with the submitted runs. Where an attachment is a kernel output, verify the exact output version and download access before packaging it. The release account could list files from all five attached kernel outputs. Host access and the exact output versions still need verification.

## Reproduction check

Use a reproduction copy with the required inputs and recorded image. The original metadata includes a private tokenizer attachment. The 27B code loads the tokenizer from its GPTQ checkpoint. Detach the private attachment in the reproduction copy and run the notebook's interactive smoke. Run the blend smoke as well and check the Flash log and final output so a fallback to the baseline cannot pass as a successful blend.

Open checks are the exact blend version, the original wheel and cache output versions, host access to the inputs, and a new offline GPU run. The files in this commit were checked locally for consistency and syntax.
