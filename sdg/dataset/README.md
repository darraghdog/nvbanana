# Build the SFT Dataset

The release recipe builds **four augmented chat rows per accepted base
puzzle**. It starts from the [Stage 4 output](../generation/README.md#stage-4--generate-output-programs-and-check-agreement),
which retains all 30 agreed input/output pairs. The builder uses the bundled
[ARC-AGI-2 training task IDs](arc_agi2_train_ids.txt) to check the training
anchor and requires explicit evaluation-free lineage. The ID list matches
the 1,000 files in the upstream training directory at the pinned revision.
[ARC-AGI-2 training source](https://github.com/arcprize/ARC-AGI-2/tree/f3283f727488ad98fe575ea6a5ac981e4a188e49/data/training)

## Create four rows per puzzle

Run from the repository root with Python 3.10 or newer.

```bash
python3 -m sdg.dataset.build_dataset outputs/sdg-demo/puzzles.jsonl \
  --output-dir outputs/sdg-demo-sft --seed 2026
```

Each row draws four demonstrations and a fifth, distinct query from that
puzzle's agreed grid pairs. The query input and its target output are the
last two messages. Each row applies the same rotation, reflection, and
one-to-one color map to every input and output. Color 0 stays fixed. The
four rows use different source queries and spatial transforms.

The default puts all puzzles in the training split. With a larger sample,
use `--validation-puzzles N` to reserve `N` whole generated puzzles
before augmentation. The four rows for each puzzle stay in the same split.

- `train.jsonl` and optional `validation.jsonl` have the NeMo RL chat
  columns `puzzle_name` and `messages`. Each grid is digit text with one
  line per row. User and assistant messages alternate.
- `provenance.jsonl` records the source IDs, chosen pair indices, and
  transformations for every row.
- `manifest.json` records the seed, input checksum, row counts by split,
  and output checksums.

## Rebuild the Hoffmann example

The bundled [Hoffmann record](examples/hoffmann-rd1-20854.jsonl) contains
the 30 original grid pairs from RD1 pair 20854. All three independent
output programs agreed on all 30. The ARC training anchor is `fcb5c309`.
The archived Stage 4 record has SHA-256
`71875dcc5977c18b0a99d4f8a7eee51e734e17e9724d4d10c78e9f0899c0d383`.
[Source and worked puzzle](../sourcing/examples.md#hoffmann--counters-in-intersecting-lines)

```bash
python3 -m sdg.dataset.build_dataset sdg/dataset/examples/hoffmann-rd1-20854.jsonl \
  --output-dir outputs/hoffmann-sft --seed 2026
cmp outputs/hoffmann-sft/train.jsonl sdg/dataset/examples/hoffmann-train.jsonl
```

## Export for NeMo RL

With Hugging Face `datasets` installed in the training environment, export
the JSONL splits to `save_to_disk` format.

```bash
python3 -m sdg.dataset.export_hf outputs/hoffmann-sft \
  --output-dir outputs/hoffmann-sft-hf
```

The training path is `outputs/hoffmann-sft-hf/train`. A build with a
validation split also creates a `validation` dataset alongside it. This
small sample documents the workflow. The production checkpoint used a
larger training mixture and its recorded settings.
