#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
conda_executable="${CONDA_EXE:-$HOME/miniconda3/bin/conda}"
revision=607ca7a0bb92e261120bcab8d9f97f28b3130ffc
if [[ ! -x .conda-protomotions/bin/python ]]; then
  CONDA_PKGS_DIRS="$PWD/.conda-pkgs" "$conda_executable" create --prefix "$PWD/.conda-protomotions" python=3.11 pip -y
fi
export PYTHONPATH="$PWD"
if [[ ! -d external/ProtoMotions ]]; then
  mkdir -p external
  git clone --no-checkout --filter=blob:none https://github.com/NVlabs/ProtoMotions.git external/ProtoMotions
  git -C external/ProtoMotions checkout --detach "$revision"
elif [[ "$(git -C external/ProtoMotions rev-parse HEAD)" != "$revision" ]]; then
  echo '기존 ProtoMotions 소스가 검증한 버전과 다릅니다. 기존 코드를 자동 변경하지 않습니다.' >&2
  exit 1
fi
.conda-protomotions/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r integrations/protomotions-requirements.lock.txt
echo 'ProtoMotions CPU 변환 환경 준비 완료. 편집기의 저장 옵션 또는 CLI로 내보낼 수 있습니다.'
