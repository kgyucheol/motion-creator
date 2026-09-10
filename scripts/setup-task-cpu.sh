#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
conda_executable="${CONDA_EXE:-$HOME/miniconda3/bin/conda}"
revision=693f74d13b3d04a0a22ce127ee79c929dd89756b
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD"
export CONDA_PKGS_DIRS="$PWD/.conda-pkgs"
if [[ ! -x .conda-policy/bin/python ]]; then
  "$conda_executable" create --prefix "$PWD/.conda-policy" python=3.11 pip -y
fi
.conda-policy/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r integrations/task-cpu-requirements.lock.txt
if [[ ! -f external/ardy-src/ardy/model/load_model.py ]]; then
  mkdir -p external
  git clone --no-checkout --filter=blob:none https://github.com/nv-tlabs/ardy.git external/ardy-src
  git -C external/ardy-src checkout --detach "$revision"
fi
.conda-policy/bin/python scripts/download-task-models.py
PYTHONPATH="$PWD:$PWD/external/ardy-src" .conda-policy/bin/python -c 'import onnxruntime, ardy, torch; print("CPU inference ready; torch", torch.__version__)'
echo 'CPU 환경 설치 완료. 실행 중인 편집기를 재시작하세요.'
