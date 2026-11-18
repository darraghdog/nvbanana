#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1
exec "${PYTHON:-python3}" -m training.run_sft "$@"
