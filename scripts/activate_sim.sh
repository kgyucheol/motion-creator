# source this file: venv + ROS 2 humble + headless GPU rendering for MuJoCo
VENV="${VENV:-$HOME/envs/ramen}"
. "$VENV/bin/activate"
. /opt/ros/humble/setup.bash   # also exposes ROS pytest plugins; they need `lark` (installed by setup_sim_env.sh)
export CYCLONEDDS_HOME="${CYCLONEDDS_HOME:-$HOME/cyclonedds/install}"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYOPENGL_PLATFORM=egl   # device 0 = the NVIDIA GPU on the GPU PC
