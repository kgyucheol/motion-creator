#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD:$PWD/external/ProtoMotions"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
.conda-protomotions/bin/python -m pytest -q tests/test_protomotions.py
