#!/usr/bin/env bash
# Build the sim2sim environment (decoupled_wbc + ROS 2 humble + Unitree SDK + MuJoCo/EGL) on Ubuntu 22.04.
# Idempotent. Verified on the GPU PC (RTX 5070 Laptop). Needs internet, git-lfs, ROS humble, and a built
# CycloneDDS C library (CYCLONEDDS_HOME, needed by the cyclonedds python binding).
set -euo pipefail

GR00T_DIR="${GR00T_DIR:-$HOME/GR00T-WholeBodyControl}"
GR00T_REV="${GR00T_REV:-087f9ac}"                     # NVlabs/GR00T-WholeBodyControl revision used so far
VENV="${VENV:-$HOME/envs/ramen}"
CYCLONEDDS_HOME="${CYCLONEDDS_HOME:-$HOME/cyclonedds/install}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

[ -f /opt/ros/humble/setup.bash ] || { echo "ROS 2 humble not found" >&2; exit 1; }
[ -f "$CYCLONEDDS_HOME/lib/libddsc.so" ] || { echo "CycloneDDS not built at $CYCLONEDDS_HOME" >&2; exit 1; }

# 1. decoupled_wbc lives in the GR00T-WholeBodyControl repo; only decoupled_wbc/ needs its LFS files.
if [ ! -d "$GR00T_DIR/.git" ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone -q https://github.com/NVlabs/GR00T-WholeBodyControl.git "$GR00T_DIR"
fi
git -C "$GR00T_DIR" checkout -q "$GR00T_REV"
git -C "$GR00T_DIR" lfs pull --include="decoupled_wbc/**,external_dependencies/**"

# 2. venv from the system python so rclpy (built for it) imports after sourcing ROS.
[ -d "$VENV" ] || python3 -m venv "$VENV"
. "$VENV/bin/activate"
pip install -q --upgrade pip wheel setuptools

# 3. Pinned to what decoupled_wbc expects: numpy 1.26, scipy 1.15, pinocchio 2.7 (4.x needs numpy 2).
pip install -q "numpy==1.26.4" "scipy==1.15.3"
pip install -q torch --index-url https://download.pytorch.org/whl/cpu     # lower-body policy runs on ONNX/CPU
pip install -q mujoco "pin==2.7.0" onnxruntime gymnasium loguru termcolor pygame glfw msgpack msgpack-numpy \
  pyzmq tyro pyyaml sshkeyboard lark "opencv-python-headless==4.11.0.86" "rerun-sdk==0.21.0" pytest "numpy==1.26.4"

# 4. Unitree SDK: use the copy vendored by GR00T (it adds OdoState_, absent from upstream unitree_sdk2_python).
CYCLONEDDS_HOME="$CYCLONEDDS_HOME" pip install -q "cyclonedds==0.10.2"
pip install -q --no-deps -e "$GR00T_DIR/external_dependencies/unitree_sdk2_python"

# 5. decoupled_wbc: its pyproject reads ../README.md, which current setuptools refuses, so add a .pth instead.
echo "$GR00T_DIR" > "$(python -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')/decoupled_wbc_path.pth"

# 6. This project.
pip install -q --no-deps -e "$PROJECT_DIR"

set +u   # ROS setup.bash reads unset variables
. /opt/ros/humble/setup.bash
set -u
CYCLONEDDS_HOME="$CYCLONEDDS_HOME" python - <<'PY'
import importlib
for m in ("decoupled_wbc.control.main.teleop.run_sim_loop", "decoupled_wbc.control.main.teleop.run_g1_control_loop",
          "rclpy", "cv_bridge", "pinocchio", "onnxruntime", "unitree_sdk2py.idl.unitree_hg.msg.dds_", "ramen_controller.fsm"):
    importlib.import_module(m)
print("environment OK")
PY
