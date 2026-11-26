# Production Qwen3.8-27B Checkpoint

The submitted 27B checkpoint descends from Qwen3.8-27B revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Its text-only ARC-16 base
initialized the first phase. Later phases restored trained weights and
Adam state from checkpoints. The [model spec](arc16/model_spec.json)
records the exact base conversion contract and the source weight
checksums.

## Training and resume chain

1. **Initial SFT through step 72,333.** The run used the four core datasets
   and ten synthetic `_075` subsets. It trained in BF16 with TP4, PP2, CP1
   and global batch size 1,024. The learning rate rose from 1e-6 to 3e-5
   over the first 200 updates, then stayed constant. After step 3,400,
   the run switched from 32K to 16K packed bins and preserved Adam state,
   scheduler progress, and data loader position. The maximum individual
   sequence length stayed at 12,288 tokens.
2. **Continuation through step 154,800.** The run retained Adam state
   from step 72,333. It reset the data loader and added the
   `_025` complements and both splits of INSP16 and INSP16A. A one-update
   BF16 bridge saved step 72,334 in a reshardable format. The main run
   used TP1, PP4, MXFP8, global batch size 1,024, and a constant 3e-5
   schedule.
3. **Final decay and export at step 188,222.** The fork retained the
   optimizer from step 154,800, reset the data loader for the final mix,
   and kept the TP1 / PP4 MXFP8 layout with a global batch of 1,024.
   Linear decay began immediately at 3e-5 and targeted 1e-7 after
   33,991 updates (step 188,791). Training stopped at step 188,222 after
   33,422 decay updates to avoid queuing another job for the remaining 569.
   The schedule gives a learning rate of about 6.0e-7 at that checkpoint.
   The checkpoint was exported in Hugging Face BF16 format. For Kaggle,
   GPTQ quantized the MLP projections to 4 bits and the attention and
   selected gated delta network (GDN) projections to 8 bits. The embeddings,
   output head, smaller GDN input projections, and GDN convolution stayed
   in BF16. The baseline and blend notebook records
   identify the distributed inference artifacts.
   [Inference manifest](../manifests/inference.json)

## Compiler cache

During initial SFT, each node compiled kernels locally and reused shared
`torch.compile` cache snapshots across jobs. The 32K and 16K packing
configurations used separate cache keys. The bridge, continuation, and
final decay used node-local compiler caches with shared snapshots disabled.

## Data mixtures

The four core datasets were ARC-AGI-2 training, MINI-ARC, ConceptARC,
and RE-ARC. The initial ten synthetic subsets were the combined
TIR/RD1/INSP4/INSP5 dataset, INSP5A, INSP9, INSP11, INSP12, INSP13, and
INSP14A–D, for 14 dataset paths in total. The continuation used both
splits of those ten subsets, both splits of INSP16 and INSP16A, and the
four core datasets, for 28 paths.

The final fork used 15 paths. It kept the four core datasets and the
`_025` subsets of the combined TIR/RD1/INSP4/INSP5 dataset, INSP5A,
INSP9, INSP11, INSP12, INSP13, INSP14A, INSP16, and INSP16A. It added
the `_075` subsets of INSP16B and INSP16C.

The production configs identify ARC evaluation tasks as a source for some
of the synthetic data in the initial raw `_075` mix and in INSP16 and INSP16A.

The final fork used NeMo RL 0.7.0 with the [source pins](runtime.json),
NVIDIA Resiliency Extension 0.6.0, Transformer Engine 2.16, and FlashAttention 4.
The reduced recipe uses its own four-row dataset and one-update schedule.
