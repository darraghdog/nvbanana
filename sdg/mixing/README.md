# Mixing Puzzle Specs

Start with two puzzle summaries. The first describes an ARC-AGI-2 training
task in the style of [NVARC's summary step](https://github.com/1ytic/NVARC/blob/846d0198efa752534594e321fc3289fc0a06c657/SDG/prompts/summary_v1.md).
The second is a [book-derived inspiration](../sourcing/inspiration_outputs.jsonl).
The [mix prompt](mix_prompt.txt) asks a model to make one rule control where
or how the other applies. This follows [NVARC's two-summary mixing step](https://github.com/1ytic/NVARC/blob/846d0198efa752534594e321fc3289fc0a06c657/SDG/prompts/mix_v1.md)
and the two-input layout used in INSP11. The compact prompt emits a spec,
which production INSP11 generated together with executable input code.

## Prepare pairs

Run from the repository root with Python 3.10 or newer.

```bash
mkdir -p outputs
python3 sdg/mixing/mix_puzzles.py prepare --count 12 --seed 2026 > outputs/pairs.jsonl
```

This runs offline. Each pair contains the two input specs, the rendered
prompt, the ARC training task ID, and the book mechanism ID and scan page.
Use `--count 240` to pair every bundled book spec once. The seed controls
which book specs are selected and the order of the training partners.

The [bundled training summaries](training_summaries.jsonl) contain 24
Gemini 3.1 Pro summaries from our Stage 1 training-task pool. Their task IDs
were checked against `arc-agi_training_challenges.json` from ARC-AGI-2
(SHA-256 `779eaba89790ebad9af02514a7efc0aefaf2cf8236f046a31bbf8b9ec48f20f5`).
[ARC-AGI-2 training data](https://github.com/arcprize/ARC-AGI-2/tree/f3283f727488ad98fe575ea6a5ac981e4a188e49/data/training) ·
[Apache 2.0 license](https://github.com/arcprize/ARC-AGI-2/blob/f3283f727488ad98fe575ea6a5ac981e4a188e49/LICENSE)

## Continue to GPT-OSS generation

The [Stage 2/3 and Stage 4 guide](../generation/README.md) takes `pairs.jsonl`
directly into the combined GPT-OSS-120B pass with NeMo Skills.

## Preview mixed specs

Point `SDG_API_BASE` at an OpenAI-compatible Chat Completions server's API
base (typically ending in `/v1`). Set `SDG_API_TOKEN` in the environment
if the server requires it. Use a model identifier from that server.

```bash
python3 sdg/mixing/mix_puzzles.py generate outputs/pairs.jsonl \
  --model "$SDG_MODEL" > outputs/mixed_specs.jsonl
```

Each output includes the source IDs and a `mixed_spec` with seven fields.
If the model returns malformed JSON or misses a field, the row has a null
`mixed_spec` and retains the response for inspection. A successful row is a
candidate rule for inspecting the mix before running the combined generation
workflow.
[Code generation and agreement demo](https://www.kaggle.com/code/darraghdog/book-spec-to-arc-puzzles-with-flash)
