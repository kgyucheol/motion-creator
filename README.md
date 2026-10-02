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

## 상태

- 구현·테스트됨(가짜 포트 기준): 계약, 번들 로더, 포즈 평균, FSM 전체 사이클·실패 경로.
- 확인됨(GPU PC): decoupled_wbc sim 루프가 헤드리스로 ~200 Hz `rt/lowstate`를 발행, EGL이 RTX 5070으로 RGB-D 렌더링.
- **미구현:** 가상 카메라(RGB-D) 소스, 비전 모듈 어댑터, `decoupled_wbc` sim 어댑터, 디버그 GUI, 실기 어댑터.
- 실제 로봇·시뮬레이터·비전과 연결해 본 적 없음.
