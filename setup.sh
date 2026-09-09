#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
conda_executable="${CONDA_EXE:-$HOME/miniconda3/bin/conda}"
if [[ ! -x "$conda_executable" ]]; then
  echo 'Miniconda를 설치하거나 CONDA_EXE를 지정하세요.' >&2
  exit 1
fi
if [[ ! -x .conda/bin/python ]]; then
  CONDA_PKGS_DIRS="$PWD/.conda-pkgs" "$conda_executable" create --prefix "$PWD/.conda" python=3.11 pip -y
fi
.conda/bin/python -m pip install -r requirements.lock.txt
npm --prefix frontend ci
npm --prefix frontend run build
echo '설치 완료. ./run.sh 실행 후 http://127.0.0.1:8765 에 접속하세요.'
