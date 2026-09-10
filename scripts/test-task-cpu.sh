#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD:$PWD/external/ardy-src"
export PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=2
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
exec .conda-policy/bin/python -m pytest -q tests/test_task_cpu.py "$@"
