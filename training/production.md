# Production Qwen3.8-27B Checkpoint

The submitted 27B checkpoint descends from Qwen3.8-27B revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Its text-only ARC-16 base
was used for each training phase. The [model spec](arc16/model_spec.json)
records the exact base conversion contract and the source weight
checksums.

## Training and resume chain

1. **Initial SFT through step 72,333.** The run used 74,069,541 rows from
   the four core datasets and ten synthetic `_075` subsets. It trained in
   BF16 with TP4, PP2, CP1 and global batch size 1,024. The learning rate
   was 3e-5 after a 200-step warmup. Packed bins started at 32K and
   changed to 16K at step 3,400. Maximum individual sequence length was
   12,288.
2. **Continuation through step 154,800.** The run retained Adam state
   from step 72,333. It reset the data loader to 125,249,244 rows across
   28 paths, adding the `_025` complements and both splits of INSP16 and
   INSP16A. A one-update BF16 bridge saved step 72,334 in a reshardable
   format. The main run used TP1, PP4, MXFP8, global batch size 1,024,
   and a constant 3e-5 schedule. It later moved from 64 to 128 GB300 GPUs.
3. **Final decay and export at step 188,222.** The fork retained the
   optimizer from step 154,800, reset the data loader to 34,807,560 rows,
   and kept the TP1 / PP4 MXFP8 layout with a global batch of 1,024.
   The planned 33,991-update linear decay ran from 3e-5 toward 1e-7 with
   zero warmup or constant steps. Step 188,222 was exported to Hugging
   Face BF16 and quantized for Kaggle, with W4 MLP weights and W8
   attention and gated-delta-network weights. The baseline and blend
   notebook records identify the distributed inference artifacts.
   [Inference manifest](../manifests/inference.json)

## Data mixtures

The four core datasets were ARC-AGI-2 training (155,904 rows), MINI-ARC
(37,632), ConceptARC (40,960), and RE-ARC (102,392). The initial ten
synthetic subsets were the combined TIR/RD1/INSP4/INSP5 dataset, INSP5A,
INSP9, INSP11, INSP12, INSP13, and INSP14A–D. The continuation used both
splits of those ten subsets, both splits of INSP16 and INSP16A, and the
four core datasets.

The final fork used the following rows.

| Dataset | Training rows |
| --- | ---: |
| Four core datasets | 336,888 |
| TIR/RD1/INSP4/INSP5 `_025` | 2,053,313 |
| INSP5A `_025` | 850,692 |
| INSP9 `_025` | 1,464,944 |
| INSP11 `_025` | 1,250,939 |
| INSP12 `_025` | 2,941,512 |
| INSP13 `_025` | 1,327,262 |
| INSP14A `_025` | 3,741,685 |
| INSP16 `_025` | 3,342,228 |
| INSP16A `_025` | 3,308,310 |
| INSP16B `_075` | 7,940,842 |
| INSP16C `_075` | 6,248,945 |
| **Total** | **34,807,560** |

The final fork used NeMo RL 0.7.0 with the [source pins](runtime.json),
NVIDIA Resiliency Extension 0.6.0, Transformer Engine 2.16, and FlashAttention 4.
The reduced recipe uses its own four-row dataset and one-update schedule.
