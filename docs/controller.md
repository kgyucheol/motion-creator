# Ramen controller (런타임 패키지)

라면 스택 집기·이송·내려놓기 작업을 하나의 FSM으로 실행하는 제어기. 같은 FSM을 sim2sim과 실기에서 돌리고, 로봇 입출력
어댑터만 교체하는 것이 목표다. 구성: 전신제어기(`decoupled_wbc`) + 비전 모듈(교체 가능) + 모션 모듈(MotionCreator 번들).

## 저장소 구조: 패키지 두 개

```
motioncreator/            저작 패키지 (서버, 프런트엔드 연동, Kimodo, policy preview …)
  robot.py tool_model.py scene_geometry.py grasp.py ramen_sequence.py motion.py interaction.py …   "코어"
  sim_scene.py            sim2sim MJCF 생성: 코어의 툴 모델·충돌 프록시 + 헤드 카메라 + 번들 환경
ramen_controller/         런타임 패키지 (numpy/scipy만으로 import 가능)
  contracts.py fsm.py runner.py vision.py pose_average.py motion_bundle.py motion_library.py
  operator.py auto_positioner.py debug_gui.py
  adapters/decoupled_wbc.py   ROS 2 전신제어기 연결
  sim/                        stack.py hook.py feed.py proxy_grasp.py  (MuJoCo + decoupled_wbc 필요)
tests/controller/         런타임 테스트 + 패키지 경계 테스트
configs/sim_motions.json  FSM 단계 → 번들 구간 매핑
scripts/                  setup_sim_env.sh activate_sim.sh run_sim2sim.sh/.py build_sim_scene.py probe_stack.py
```

경계 규칙 (`tests/controller/test_package_boundary.py`가 강제한다):
1. `ramen_controller` → `motioncreator`는 **코어 모듈만** import 할 수 있다. 서버 등 나머지는 금지.
2. `motioncreator`는 `ramen_controller`를 import 하지 않는다.
3. 런타임 코어는 fastapi/torch/kimodo/onnxruntime 없이 import 된다.
4. 둘 사이의 데이터 계약은 파일이다: 번들(`metadata.json`, `motion.csv`, `environment.json`)과 scene 사이드카
   (`scene_objects.json`: 카메라, 툴 바디, 물체 이름/geom).

설치: `pip install -e .[controller]` (런타임), `pip install -e .[controller,sim]` (+ sim2sim). 저작 도구는 기존 `requirements.txt` 흐름.

## FSM

`INITIALIZE → MANUAL_POSITION → OBSERVE → PLAN_GRASP → EXECUTE_GRASP → TRANSFER → PLACE → RETURN → OBSERVE …`
어느 단계에서든 실패하면 `RECOVERY`. 모든 전환은 `TaskEvent`로 사유와 함께 기록된다.

- `OBSERVE`: 1초 창의 검출을 이상치 제거 후 평균(회전은 회전 공간에서). 실패 시 5초 후 `RECOVERY`, 관측·계획 실패만 자동 재시도(3회).
- `EXECUTE_GRASP`/`TRANSFER`/`PLACE`: 모션 종료가 아니라 `GraspMonitor.held()`로 다음 단계 진행을 판정.
- 낙상, 파지 미확인, 이송 중 물체 이탈은 자동 재시도 없이 작업자 reset을 기다린다.
- 모션 단계 사이 전환 시간은 자세 차이에 비례한다(`blend_speed_rad_s`, 최대 `blend_s`).
- 관절 소유권: 모션은 팔 목표·허리 관절·높이·이동 속도를 *요청*하고, 허리·다리의 최종 명령은 전신제어기가 정한다.

## 비전은 교체 가능하다

`vision.PoseEstimator`(프레임 → 6D pose)를 구현하면 된다. 깊이가 필요하면 `needs_depth = True`로 선언한다
(RGB 전용 카메라에 깊이 백엔드를 붙이면 명확한 오류로 멈춘다). 백엔드 예외는 "검출 없음"으로 처리된다.
`MockPoseEstimator`는 sim 정답에 노이즈·누락·이상치를 얹는다. FoundationPose 계열은 RGB-D가 필요하다.

## sim2sim

한 번: `scripts/setup_sim_env.sh` (decoupled_wbc + ROS 2 humble + Unitree SDK + MuJoCo/EGL venv). 번들을 `data/bundle_a`에 복사한다.

```bash
. scripts/activate_sim.sh
scripts/run_sim2sim.sh data/bundle_a --cycles 3 --gui-port 8765   # scene 생성 → sim 스택 기동 → FSM 실행 → 정리
```

- scene의 로봇 툴은 번들 `metadata.json`의 `mounted_tools`를 따른다(`--tools`로 덮어쓰기). 에디터에서 툴을 바꾸면 sim 로봇도 바뀐다.
- 구성(실기와 같은 분리): `sim.stack`(MuJoCo sim + WBC 정책 + ROS 노드, 한 프로세스) ⇄ ROS 2 ⇄ FSM 프로세스. 카메라 RGB-D와 sim 정답은 ZMQ(`sim.feed`).
- 로봇은 박스 1 m 뒤에서 시작하고 `AutoPositioner`가 오도메트리로 지정 위치까지 걷는다(`--keyboard`로 수동 가능).
- 업스트림은 하체 정책을 `]`로 켠 뒤 `9`로 탄성 밴드를 풀어야 서 있다. `bringup_sim`이 이 키를 보낸다.
- `--proxy-grasp`: 파지 모션이 끝나면 잡힌 것으로 **가정**한다(파지 물리 미검증). 기본은 양쪽 툴과 타깃의 접촉력(≥ 1 N)으로 판정.
- 디버그 GUI: `http://127.0.0.1:<gui-port>/` (실행 중에만 존재; 다른 PC에서는 SSH 터널).

## 상태

- 확인됨(GPU PC, RTX 5070): sim 스택 + 가상 카메라(RGB-D, 정답) + FSM이 프록시 파지로 3사이클 무낙상 완료.
- **파지 물리는 실패한다**(프록시 없이): 고정 파지(`FixedGraspPlanner`)는 저작된 한 자세만 재생하는데 보행 후 로봇이 밀리고 서는 높이가
  0.747 m(저작 0.79 m)라 ~8 cm 빗나간다. 포즈에 적응하는 파지 플래너가 필요하다. 사이클을 반복하면 위치 오차가 누적된다.
- 합치기 전 sim 모델은 기본 툴 쌍(주걱+받침)으로 고정돼 있어 번들의 `mounted_tools`(주걱+주걱)와 달랐다. 합친 뒤에는 번들을 따른다.
- 비전: 목업만. 실제 백엔드, 실기 어댑터, 실제 grip 센서 판정은 미구현. 실제 로봇과 연결해 본 적 없음.
