#!/usr/bin/env bash
# Run on the robot as root: sudo bash ~/robot_cloud/install_robot_objects_service.sh
# Installs robot-objects.service (needs ~/robot_cloud/venv, models/ramen_container.onnx and robot_object_detector.py).
# Roll back: sudo systemctl disable --now robot-objects.service
set -euo pipefail
directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $EUID -ne 0 ]]; then
  echo 'sudo 로 실행하세요.' >&2
  exit 1
fi
pkill -f '^/home/unitree/robot_cloud/venv/bin/python robot_object_detector.py' || true
sleep 1
install -m 644 "$directory/robot_objects.service" /etc/systemd/system/robot-objects.service
systemctl daemon-reload
systemctl enable --now robot-objects.service
sleep 10
systemctl --no-pager --lines=5 status robot-objects.service
