#!/usr/bin/env bash
set -euo pipefail
example_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$example_dir/../.." && pwd)"
cd "$repo_dir"
uv run --no-project --python "${FOUNDRY_PYTHON:-.venv/bin/python}" rf3 fold \
    "inputs=$example_dir/${1:-distance.json}" \
    "out_dir=$example_dir/outputs/${2:-distance}" \
    diffusion_batch_size=2 n_recycles=2 num_steps=50 \
    early_stopping_plddt_threshold=null seed=42 skip_existing=false
