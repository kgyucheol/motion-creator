#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source_root="${1:-../GR00T-WholeBodyControl}"
asset_root="$source_root/decoupled_wbc/sim2mujoco/resources/robots/g1"
target_root="external/task-models/decoupled-wbc"

if [[ ! -f "$asset_root/g1_gear_wbc.xml" ]]; then
  echo "GR00T-WholeBodyControl 자산을 찾을 수 없습니다: $asset_root" >&2
  echo "사용법: scripts/setup-decoupled-wbc.sh /path/to/GR00T-WholeBodyControl" >&2
  exit 1
fi

mkdir -p "$target_root"
cp "$asset_root/g1_gear_wbc.xml" "$target_root/g1_gear_wbc.xml"
cp "$asset_root/policy/GR00T-WholeBodyControl-Balance.onnx" "$target_root/GR00T-WholeBodyControl-Balance.onnx"
cp "$asset_root/policy/GR00T-WholeBodyControl-Walk.onnx" "$target_root/GR00T-WholeBodyControl-Walk.onnx"
cp "$source_root/LICENSE" "$target_root/LICENSE"
rm -rf "$target_root/meshes"
cp -a "$asset_root/meshes" "$target_root/meshes"

python_executable=.conda-policy/bin/python
if [[ ! -x "$python_executable" ]]; then
  echo '.conda-policy 환경이 없습니다. 먼저 scripts/setup-task-cpu.sh를 실행하세요.' >&2
  exit 1
fi
PYTHONPATH="$PWD" "$python_executable" -c 'from motioncreator.decoupled_wbc import verify_assets; print(verify_assets())'
echo 'Decoupled WBC 시뮬레이션 자산 설치 완료'
