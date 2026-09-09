#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ ! -x .conda/bin/python ]]; then
  echo '먼저 ./setup.sh 를 실행하세요.' >&2
  exit 1
fi
if [[ ! -f frontend/dist/index.html ]]; then
  npm --prefix frontend run build
fi
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD"
exec .conda/bin/python -m motioncreator.server "$@"
