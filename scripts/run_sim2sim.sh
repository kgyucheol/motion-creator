#!/usr/bin/env bash
# One-shot sim2sim: build the scene, start the sim stack, run the FSM, stop the stack.
#   scripts/run_sim2sim.sh <bundle-dir> [run_sim2sim.py options...]
set -uo pipefail
[ -z "$(pgrep -f "[r]amen_controller.sim_stack")" ] || { echo "a sim stack is already running; stop it first" >&2; exit 1; }
cd "$(dirname "$0")/.."
BUNDLE="${1:?usage: run_sim2sim.sh <bundle-dir> [options]}"; shift
set +u   # ROS setup.bash reads unset variables
. scripts/activate_sim.sh
set -u
python scripts/build_scene.py "$BUNDLE" build/scene --target "${TARGET:-Mupama_Stack_L03_R04}" >/dev/null || exit 1
mkdir -p logs
setsid python -m ramen_controller.sim_stack --scene-dir build/scene > logs/stack.log 2>&1 &
STACK=$!
trap 'kill -TERM -$STACK 2>/dev/null; sleep 2; kill -KILL -$STACK 2>/dev/null' EXIT
sleep 20
python scripts/run_sim2sim.py "$@"
