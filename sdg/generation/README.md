# Generate and Verify Puzzles

This small recipe follows the two-input INSP11 workflow. GPT-OSS-120B
runs both generation stages through NeMo Skills and vLLM. Stage 2/3
produces the mixed rule and tests its input generator with the stateful
Python tool. Stage 4 writes three independent output programs for each rule.
The filter keeps a puzzle when all three programs agree on 30 fresh grids.

## Setup

Install [NeMo Skills](https://github.com/NVIDIA-NeMo/Skills/tree/022904023ad7a83a87662a313cf72e7df5891d55)
on the launch machine. Use a configured local or Slurm cluster with enough
GPU memory to serve GPT-OSS-120B and a sandbox image for the Python tool.
Mount this checkout at `/workspace/nvbanana` in the generation client so
it can read the prompts and write to `outputs/`. For a remote cluster mount
the remote copy of the checkout. Keep the same run path on both machines and
sync `pairs.jsonl` and `stg4-input.jsonl` to the cluster before generation.
Sync each stage's completed outputs back before running its CPU checks.
Set `SDG_CLUSTER` to your NeMo Skills cluster name. The default below uses eight GPUs. Set `SDG_GPUS` to use a
different tensor parallel size supported by your hardware.
[NeMo Skills cluster configuration](https://nvidia-nemo.github.io/Skills/basics/cluster-configs/) ·
[GPT-OSS stateful code execution](https://nvidia-nemo.github.io/Skills/tutorials/2025/08/29/inference-with-gpt-oss-120b-using-stateful-python-code-execution/)

Run the following commands in Bash from the repository root. Choose a fresh
run directory name for each new generation.

```bash
python3 -m pip install 'git+https://github.com/NVIDIA-NeMo/Skills.git@022904023ad7a83a87662a313cf72e7df5891d55'
```

```bash
run=outputs/sdg-demo
mkdir -p "$run"
python3 sdg/mixing/mix_puzzles.py prepare --count 12 --seed 2026 > "$run/pairs.jsonl"
mounted=/workspace/nvbanana

generation_args=(
  --cluster "${SDG_CLUSTER:?Set SDG_CLUSTER to your NeMo Skills config name}"
  --model openai/gpt-oss-120b
  --server-type vllm
  --server-gpus "${SDG_GPUS:-8}"
  --with-sandbox
  --sandbox-env-overrides NEMO_SKILLS_SANDBOX_BLOCK_NETWORK=1
)

python_tool_args=(
  '++inference.endpoint_type=text'
  '++inference.temperature=1.0'
  '++code_execution=True'
  '++code_tags=gpt-oss'
  '++chat_template_kwargs.builtin_tools=[python]'
  '++server.code_execution.max_code_executions=64'
)
```

The CPU commands below execute generated Python. Run them in a read-only,
networkless container with only the SDG code and this run's results mounted.
The NeMo Skills jobs start their own stateful Python sandbox.

```bash
check_generated() {
  docker run --rm --network none --read-only --cap-drop ALL \
    --security-opt no-new-privileges --memory 1g --pids-limit 64 \
    --user 65534:65534 --tmpfs /tmp:rw,nosuid,nodev,size=64m \
    --mount "type=bind,source=$PWD/sdg,target=/recipe/sdg,readonly" \
    --mount "type=bind,source=$PWD/$run,target=/run-data,readonly" \
    --workdir /recipe python:3.12-slim \
    python -B -m sdg.generation.build_puzzles "$@"
}
```

## Stage 2/3 — Mix the rule and write input code

`PUZZLE1` is an ARC-AGI-2 training summary in NVARC format. `PUZZLE2`
is an extracted book rule. The [Stage 2/3 prompt](prompts/stg23.yaml)
asks the model to make the two rules interact and verify a seeded
`generate_puzzle_input(seed)` implementation.

```bash
ns generate "${generation_args[@]}" \
  --input-file "$mounted/$run/pairs.jsonl" \
  --output-dir "$mounted/$run/stg23" \
  "${python_tool_args[@]}" \
  "++prompt_config=$mounted/sdg/generation/prompts/stg23.yaml" \
  '++chat_template_kwargs.reasoning_effort=high' \
  '++inference.tokens_to_generate=32768'
```

After NeMo Skills writes `stg23/output.jsonl`, extract the spec and
input code from its `generation` field. The preparation step rejects
malformed specs and runs the input code at seeds 1000 through 1029, away
from the model's development seeds 0 through 19. Stage 4 receives the spec
and input code. The 30 generated grids are used by the filter.

```bash
check_generated prepare-stg4 /run-data/stg23/output.jsonl > "$run/stg4-input.jsonl"
```

## Stage 4 — Generate output programs and check agreement

NeMo Skills samples one output program per seed for seeds 0, 1, and 2. The
[Stage 4 prompt](prompts/stg4.yaml) gives the model the input generator
to initialize in the Python tool. Each resulting
`generate_puzzle_output(input_grid)` must work independently.

```bash
ns generate "${generation_args[@]}" \
  --input-file "$mounted/$run/stg4-input.jsonl" \
  --output-dir "$mounted/$run/stg4" \
  --num-random-seeds 3 \
  "${python_tool_args[@]}" \
  "++prompt_config=$mounted/sdg/generation/prompts/stg4.yaml" \
  '++chat_template_kwargs.reasoning_effort=medium' \
  '++inference.tokens_to_generate=16384'
```

After `stg4/output-rs0.jsonl` through `output-rs2.jsonl` are ready,
run all three candidate programs independently on the same 30 grids.
Keep only exact 3/3 agreement across the entire set, with valid grids,
varied outputs, and at least one visible change. The resulting JSONL
keeps the source IDs, mixed spec, generator code, all three agreeing
programs, all 30 agreed `grid_pairs`, and one ARC task with three demos
and one test pair. The [SFT dataset builder](../dataset/README.md) samples
from the full pool of agreed pairs.

```bash
check_generated filter-stg4 /run-data/stg4-input.jsonl /run-data/stg4 \
  > "$run/puzzles.jsonl"
```

The compact release uses standard-library Python generators and puts the
Stage 4 input code in the prompt. Production INSP11 supplied that code
through NeMo Skills' per-row sandbox preload and used a second consensus
pass to recover more puzzles. The release runs one strict pass documented
above. [Kaggle walkthrough with Flash-Next](https://www.kaggle.com/code/darraghdog/book-spec-to-arc-puzzles-with-flash)
