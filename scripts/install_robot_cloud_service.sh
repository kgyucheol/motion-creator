#!/usr/bin/env bash
# Run on the robot as root: sudo bash ~/robot_cloud/install_robot_cloud_service.sh
# Stops teleimager (it owns the camera), installs robot-cloud.service and starts it.
# Roll back: sudo systemctl disable --now robot-cloud.service && sudo systemctl enable --now teleimager.service
set -euo pipefail
directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $EUID -ne 0 ]]; then
  echo 'sudo 로 실행하세요.' >&2
  exit 1
fi
systemctl disable --now teleimager.service
pkill -f '^/home/unitree/miniconda3/envs/teleimager/bin/python /tmp/robot_cloud_publisher.py' || true
sleep 1
install -m 644 "$directory/robot_cloud.service" /etc/systemd/system/robot-cloud.service
systemctl daemon-reload
systemctl enable --now robot-cloud.service
sleep 8
systemctl --no-pager --lines=5 status robot-cloud.service
