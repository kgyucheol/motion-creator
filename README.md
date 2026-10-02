# NSE-ramen-controller

라면 스택 집기·이송·내려놓기 작업을 하나의 FSM으로 실행하는 제어기. 같은 FSM을 sim2sim과 실기에서
돌리고, 로봇 입출력 어댑터만 교체하는 것이 목표다.

구성: 전신제어기(`decoupled_wbc`) + 비전 모듈(`FoundationPose-Shap`) + 모션 모듈(MotionCreator 번들).

## FSM

`INITIALIZE → MANUAL_POSITION → OBSERVE → PLAN_GRASP → EXECUTE_GRASP → TRANSFER → PLACE → RETURN → OBSERVE …`
어느 단계에서든 실패하면 `RECOVERY`. 모든 전환은 `TaskEvent`로 사유와 함께 기록된다.

- `OBSERVE`: 1초 창의 검출을 이상치 제거 후 평균(회전은 Euler가 아닌 회전 공간에서 평균).
  실패 시 5초 후 `RECOVERY`, 관측·계획 실패만 자동 재시도(최대 3회).
- `EXECUTE_GRASP`/`TRANSFER`/`PLACE`: 모션이 끝났다는 사실이 아니라 `GraspMonitor.held()`로 다음 단계 진행을 판정.
- 낙상, 파지 미확인, 이송 중 물체 이탈은 자동 재시도하지 않고 작업자 reset을 기다린다.
- 각 모션 시작 전 직전 명령에서 첫 프레임까지 `blend_s` 동안 부드럽게 연결한다.

## 구조

| 모듈 | 역할 |
|---|---|
| `contracts.py` | `RobotState`, `ObjectPose`, `MotionSegment`, `WbcCommand`, `TaskEvent`, 허리→몸통 RPY 변환 |
| `motion_bundle.py` | MotionCreator 번들(`metadata.json` + `motion.csv`)을 `MotionSegment`로 읽기 |
| `pose_average.py` | 반복 검출의 강건한 평균 |
| `fsm.py` | `TaskFsm`과 포트(`VisionPort`, `MotionPort`, `OperatorPort`, `GraspMonitor`) |

FSM은 하드웨어를 직접 다루지 않는다. `step(now, robot_state)`가 `WbcCommand`를 반환하고,
호출하는 쪽(sim/실기 어댑터)이 전신제어기에 보낸다. 관절 소유권: 모션은 팔 목표와 몸통 자세·높이·이동 속도를
*요청*하고, 허리·다리의 최종 관절 명령은 전신제어기가 정한다.

## 개발

FSM 단위 테스트는 어디서나 돈다: `pip install -e ".[dev]" && pytest`

sim2sim 환경(decoupled_wbc + ROS 2 humble + Unitree SDK + MuJoCo/EGL)은 GPU PC(Ubuntu 22.04)에서 구축한다:

```bash
bash scripts/setup_sim_env.sh      # 멱등. 인터넷, git-lfs, ROS humble, 빌드된 CycloneDDS 필요
. scripts/activate_sim.sh          # venv + ROS + 헤드리스 EGL(MUJOCO_EGL_DEVICE_ID=0 = NVIDIA GPU)
python -m decoupled_wbc.control.main.teleop.run_sim_loop --no-enable-onscreen --enable-offscreen --interface lo
```

주의: pinocchio는 2.7.0(4.x는 numpy 2 필요), Unitree SDK는 GR00T가 포함한 사본(`OdoState_` 포함)을 쓴다.

## sim2sim

```bash
. scripts/activate_sim.sh
python scripts/export_robot_assets.py ~/motioncreator            # 한 번: 툴 장착 G1 MJCF를 데이터로 내보냄 (assets/, git 제외)
scripts/run_sim2sim.sh data/bundle_a --cycles 3 --gui-port 8765  # scene 생성 → sim 스택 기동 → FSM 실행 → 정리
```

구성 (실기와 같은 분리): `sim_stack`(MuJoCo sim + WBC 정책 + ROS 노드, 한 프로세스) ⇄ ROS 2 ⇄ FSM 프로세스, 카메라 RGB-D와
sim 정답은 ZMQ(`sim_feed`)로 전달. 비전은 `MockPoseEstimator`(정답 + 노이즈·누락·이상치)이며 `PoseEstimator`를 구현한
백엔드로 교체한다(깊이가 필요하면 `needs_depth=True`로 선언). 디버그 GUI: `http://127.0.0.1:<gui-port>/`(SSH 터널로 접속).

- `--proxy-grasp`: 파지 모션이 끝나면 잡힌 것으로 **가정**한다(파지 물리 미검증). 나머지 단계 검증용.
- 기본(`SimFeed.held`)은 양쪽 툴과 타깃의 접촉력(≥1 N)으로 판정한다.
- sim 시작 시 로봇은 박스 1 m 뒤에 서고, `AutoPositioner`가 오도메트리로 지정 위치까지 걸어간다(`--keyboard`로 수동 가능).
- 업스트림은 정책을 `]`로 켜고 `9`로 탄성 밴드를 풀어야 한다. `bringup_sim`이 이 키를 보낸다.

## 상태

- 확인됨(GPU PC, RTX 5070): sim 스택 + 가상 카메라(RGB-D, 정답) + FSM이 **프록시 파지로 3사이클(작업 시간 73 s) 무낙상 완료**.
  비전 관측(1 s 평균) → 파지 계획 → 모션 → 이송 → 배치 → 복귀가 반복된다.
- **파지 물리는 실패한다**(프록시 없이): 보행 후 로봇이 ~10 cm 밀리고 서는 높이가 0.747 m(저작 기준 0.79 m)라
  관절 공간으로 저작된 고정 파지가 ~8 cm 빗나가 툴이 스택에 닿지 않는다(접촉력 0 N). 포즈에 적응하는 파지 플래너가 필요하다.
  `FixedGraspPlanner`(프로토타입)는 저작 위치에서 3 cm 넘게 벗어난 포즈를 거부한다.
- 비전: 목업만. 실제 백엔드(FoundationPose-Shap 등, RGB-D 필요)는 `PoseEstimator`로 붙인다. 이 PC에서 GPU 환경은 미구축.
- 실기 어댑터와 실제 파지 판정(grip 센서)은 미구현. 실제 로봇과 연결해 본 적 없음.
