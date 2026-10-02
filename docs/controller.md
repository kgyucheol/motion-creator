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

## MotionCreator 화면에서 실행하기 (`/controller`)

편집기 우상단 **Task Controller** 버튼 → `/controller` 페이지. 편집기의 인프로세스 MuJoCo 시뮬(하체 정책 + 팔 PD)을 같은 FSM이 외부에서
구동한다(`DecoupledSimulation`의 `external` 모드: 팔 14·허리 3·이동 속도·높이를 매 제어 틱 입력). ROS/DDS는 필요 없다.

1. `./run.sh --model g1-tools`로 편집기를 띄운다(프로젝트에 기록된 툴 쌍으로 로봇과 sim 모델이 자동 선택된다).
2. 모션을 번들로 저장한다(`motions/<이름>_<id>/`에 `project.json`, `metadata.json`, `motion.csv`가 있어야 한다).
3. `/controller`에서 **저장 프로젝트**를 고르고, **집을 물체**와 단계별 구간(준비·파지·이송·배치·복귀, 번들 시간 s)을 확인한다.
   기본값은 `configs/sim_motions.json`이고, 번들 폴더에 `task_stages.json`(같은 형식)을 두면 그것을 읽는다.
4. **작업 준비**(세션 생성, 약 20 s) → **Start**. 로봇은 박스 1 m 뒤(`robot_start_xy_yaw`)에서 시작해 팔을 올리고 저작 위치까지 걸어간다.
5. 화면: 3D 장면(로봇·물체·헤드캠), 작업 상태(FSM 단계, 사이클, 인식 pose, 파지 접촉력), 이벤트(전환 사유), HUD.
   **파지 가정(proxy)**: 파지 모션이 끝나면 잡힌 것으로 가정한다. **목표 가림**: 비전이 목표를 못 보는 상황을 시험한다.
   **작업자 확인**: 정지(`RECOVERY`) 원인을 확인한 뒤 준비 자세부터 다시 시작한다(낙상 후에는 Reset이 먼저).
6. 세션은 최대 2개까지 유지되고 더 만들면 가장 오래된 것이 닫힌다. 페이지를 닫으면 세션도 정리된다.

구성: `controller_session.py`(FSM ↔ sim 연결, 비전 목업, 접촉 기반 파지 판정) + `pose_adaptive_grasp.py`(포즈 적응 파지 플래너) + `controller_api.py`(`/api/controller/*`) + `frontend/app/controller-page.tsx`.
실기와의 차이: 이 sim은 편집기의 인프로세스 시뮬이므로 실기와 같은 ROS/DDS 제어 경로가 아니다. 그 경로(실기 어댑터 포함)는 아래 sim2sim 스택이 검증한다.

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

## 위치 잡기 (MANUAL_POSITION)와 건너뛰기

FSM은 `MANUAL_POSITION`에서 매 틱 OperatorPort의 이동 명령(vx, vy, yaw rate)을 WBC의 `navigate_cmd`로 넘기고, 작업자 확인(`position_confirmed`)이
서고 이동 명령이 0일 때 `OBSERVE`로 넘어간다. 오퍼레이터 구현은 두 가지다. `KeyboardOperator`는 키 입력(실로봇/수동), `AutoPositioner`(sim2sim)는
시뮬 월드 좌표를 읽어 저작 모션의 루트 위치(`authored_root_at_s`)를 목표로 P 제어(최소/최대 속도 제한, 위치 3 cm·요 0.05 rad 허용, 1 s 유지 시
확인, 서 있는 로봇의 흔들림 때문에 허용 오차의 1.7배까지는 유지로 간주)한다. 월드 좌표를 쓰므로 sim 전용이다.

로코모션은 선택 사항이라 건너뛸 수 있다: 작업 상태의 **위치 잡기** 칩(또는 테스트 옵션 ‘위치 잡기 건너뜀’)을 클릭하면 `skip_positioning`이 켜지고 로봇이
작업 위치(저작 원점)에서 시작하며 FSM은 `INITIALIZE → OBSERVE`로 간다. 설정을 바꾸면 시작 위치가 바뀌므로 세션이 초기화된다(Start를 다시 누른다).
서 있는 로봇은 시작 직후 약 6~9 cm 뒤로 자리잡는다(WBC 서기에서 골반이 발 뒤에 놓임). 고정 파지(허용 8 cm)는 이 때문에 거부될 수 있어 건너뛸 때는
포즈 적응 파지를 쓴다. **다음 상태까지** 버튼(`step-state`)은 FSM이 다음 상태로 넘어가는 순간까지 실행한 뒤 일시정지한다.

## TCP 오버레이 (계획 vs 실제 툴 끝)

페이지의 **TCP 오버레이**(기본 켜짐)는 재생 중인 모션 구간의 *계획* 툴 경로(흐린 선: 구간 시작 시점에 측정한 골반·다리 위에서 팔·허리 각도의
순기구학, 하체가 그대로라는 가정)와 *실제* 툴 끝 궤적(밝은 선), 현재 위치의 차이(빨간 선)를 3D에 그린다. 사이드바 표는 양손별 계획 대비 오차(mm),
타깃까지의 최소 거리(툴 충돌 형상 기준, 0 이하 = 접촉), 가장 가까운 이웃 물체와의 거리(음수 = 침투)를 보여 준다. `GET /api/controller/sessions/{id}/tcp`
(약 20 ms, 8 KB)이며 `motioncreator/tcp_trace.py`가 기록한다. 디버그 전용이라 실패해도 로봇은 멈추지 않는다.

## 파지 계획: 포즈 적응 (재조준)

`pose_adaptive_grasp.RetargetedGraspPlanner`는 저작된 파지 구간의 양손 툴 경로를 **저작 당시 물체 좌표계 기준으로 표현**하고,
관측된 물체 자세(월드 좌표, 로봇의 현재 루트 자세로 변환)에 다시 얹은 뒤 **현재 자세에서 골반·발을 고정하고** IK를 푼다(팔·허리만 사용).
Cylinder Lab에서 사용자가 그린 경로의 틈 사이 형상은 그대로 유지되고, 물체/로봇 위치 오차만 보정된다. 계획에 약 0.5 s.
화면의 **포즈 적응 파지** 토글로 고정 파지(`FixedGraspPlanner`)와 전환한다. 기존 `plan_ramen_sequence`는 이 장면(높은 스택, 기울어진
박스)에서 IK가 풀리지 않아 쓰지 않았다.

**한계 (측정):** 팔 관절 오차가 0인데도 계획한 TCP와 실제 TCP가 5~6 cm 어긋난다. 저작 모션은 허리 피치를 ±0.5 rad까지 쓰는데
전신제어기는 허리를 몸통 자세 요청으로 근사해서 실현하므로(명령 대비 최대 0.17 rad, 정상 상태 0.08~0.11 rad 오차, 골반 기울기 최대 9°,
로봇이 14 cm 밀림) 툴이 틈에 들어가지 못하고 이웃 스택에만 닿는다(타깃 접촉력 0 N). 허리를 잠그면(`lock_waist`) 팔만으로는 닿지 않는다.
해결하려면 측정된 허리·루트 기준으로 팔을 실시간 보정(TCP 폐루프)하거나, 허리 사용이 적은 모션으로 다시 저작해야 한다.

**폐루프 TCP 보정 (실험, 기본 꺼짐):** `RetargetedGraspPlanner(closed_loop=True)`는 약 10 Hz로 *측정된* 골반·다리·허리 기준으로 팔을 다시 풀어
월드 기준 목표 경로에 TCP를 맞춘다(`MotionSegment.refine` 훅). 이 샘플 프로젝트에서는 IK가 77회 중 63회 실패하고 이웃 스택에 438 N이 걸렸다.

**저작된 파지 자체 점검 (측정):** 저작 모션을 MuJoCo 충돌 형상으로 재생해 툴 형상과 스택 사이 최소 거리를 쟀다. 타깃 `Mupama_Stack_L03_R04`까지
최소 **60 mm**(닿지 않음)이고, 이웃 `L03_R03`/`L03_R02` 안으로 **20~55 mm 침투**한다. 즉 이 샘플의 저작 경로는 타깃을 잡는 경로가 아니다.
`left_hand` 핸들은 손목 요 링크 +5.5 cm 지점이며 툴 끝이 아니므로, 파지점 검증은 TCP 점이 아니라 툴 충돌 형상으로 해야 한다.

## 편집기와 WBC의 서기 자세 일치 (WBC 서기 자세)

편집기 기본 서기는 무릎 0.24 rad, 골반 0.787 m이고 decoupled-WBC는 무릎 약 0.74 rad, 골반 0.747 m로 선다(측정: `scripts/measure_wbc_stand.py`,
`configs/wbc_stand.json`). 기본 서기에서 만든 상체 모션은 실행 시 약 4 cm 낮고 더 뒤에 놓인다. 편집기 포즈 패널의 **WBC 서기 자세**
버튼(`GET /api/wbc-stand`, `Robot.wbc_stand`)은 하체·허리를 WBC 값으로 맞춘 서기 자세를 불러온다(팔은 기본 서기와 동일). 제어기에서 실행할
모션은 이 자세에서 시작해 만든다. WBC 정책이나 기본 높이를 바꾸면 스크립트를 다시 실행한다.

**IK에서 발·골반 고정 평가 (측정, 포즈 적응 파지 실행 중):** 발은 거의 고정이다(위치 ≤ 8 mm, 요 ≤ 5°, 기울기 ≤ 3.6°) → 발 고정은 타당한 가정.
골반은 고정이 아니다(발 중점 기준 최대 약 10 cm 이동, 무게중심 지면 투영도 최대 44 mm 이동) → 골반 위치·자세를 현재 값에 고정한 계획은 하체가
정적이라고 가정하는 셈이다. 제어 가능한 하체 자유도는 기본 높이, 허리/몸통 자세, 이동 명령뿐이라 다리 관절을 푸는 것도 답이 아니다.

## 라면 꺼내기 모션 설계 실험 (`scripts/extraction_*.py`)

샘플 장면(스택 12개를 기울어진 상자에 4열×3층으로 채움)의 최상단 스택(`Mupama_Stack_L03_R04`)을 꺼내는 모션을 물리 시뮬로 탐색하는 도구다.
출시 런타임이 아니라 설계 보조이며 `build/`(무시됨)에 결과를 쓴다.

- **측정한 장면 사실:** 타깃과 몸쪽 이웃(`L03_R03`)의 틈 3.3 mm, 아래 이웃(`L02_R04`)과 −2.5 mm(겹침), 상자 폭 여유 5 cm, 높이 여유 15 mm.
  기본 크기에서는 주걱이 틈에 들어가지 못해 이웃을 날려 버린다.
- `extraction_lab.py`: 스택 단면 스케일(`scale_stacks`), `plan_scoop` 단면 경로 → 양손 TCP(Cylinder Lab 변환), WBC 서기 하체에 골반·발 고정 IK, 로봇은 운동학
  구동·스택은 자유로 물리 평가(`KinematicRig`), 사람이 말한 기법(수평 판을 조금씩 밀어 넣으며 흔들기 → 들어 올리며 몸쪽으로 기울이기 → 운반)을 매개변수화한
  `maneuver_points`, 시작 대기 자세(`neutral_pose`: 가슴 앞에 주걱 끝을 세움; 차렷은 주걱이 허벅지 높이라 상자 위로 접근 불가), `settled_scene`(정착한 배치 저장).
- `extraction_search.py`: 병렬 무작위/국소 탐색(`--scales`로 여러 스택 크기에서 동시에 평가). `save_candidates.py`(편집기 번들 + `task_stages.json` + 영상 + 색인),
  `render_extraction.py`(mp4), `settle_saved_candidates.py`.
- **결과(측정):** 스케일 0.93(틈 약 13 mm)에서 타깃을 10~22 cm 들어 몸쪽 25~55 cm로 가져오는 후보를 찾았고 편집기 물리 미리보기에서도 재현된다. 다만 한 크기에서만
  성공하는 후보가 대부분이며(0.91/0.93/0.95 모두 성공은 300개 중 1개), 평가는 운동학 구동 로봇이 수 kN으로 밀어 넣는 접촉에 의존한다.
  WBC 시뮬(적응 파지, 위치 잡기 건너뜀)에서는 주걱이 타깃에 닿아 양쪽 접촉력(2~6 N)이 생기지만 밀어 넣는 반력으로 로봇이 10~14 cm(폐루프 TCP 보정 시 최대 37 cm)
  뒤로 물러나 TCP 오차가 10~17 cm로 벌어지고 들어 올리지 못한다. 힘 한계(유한한 팔 힘)를 반영한 탐색과 반력을 견디는 하체 전략이 필요하다.
- `task_stages.json`의 `adaptive_sample_period_s`(적응 계획 샘플 간격, 흔들기 같은 미세 동작은 0.04), `closed_loop_tcp`(폐루프 TCP 보정)를 읽는다.

## 상태

- 확인됨(GPU PC, RTX 5070): sim 스택 + 가상 카메라(RGB-D, 정답) + FSM이 프록시 파지로 3사이클 무낙상 완료.
- 포즈 적응 파지(재조준)를 연결했다. 위치 오차는 보정되지만 위 한계 때문에 접촉 판정 모드의 파지는 여전히 실패한다.
- **파지 물리는 실패한다**(프록시 없이): 고정 파지(`FixedGraspPlanner`)는 저작된 한 자세만 재생하는데 보행 후 로봇이 밀리고 서는 높이가
  0.747 m(저작 0.79 m)라 ~8 cm 빗나간다. 포즈에 적응하는 파지 플래너가 필요하다. 사이클을 반복하면 위치 오차가 누적된다.
- 합치기 전 sim 모델은 기본 툴 쌍(주걱+받침)으로 고정돼 있어 번들의 `mounted_tools`(주걱+주걱)와 달랐다. 합친 뒤에는 번들을 따른다.
- 편집기 UI(`/controller`)에서도 같은 결과: 프록시 파지로 1사이클 완료(작업 시간 약 33 s), 접촉 판정 모드는 `grasp not confirmed`(접촉력 0/0 N). 파지 실패는 sim2sim 스택 고유의 문제가 아니다.
- 비전: 목업만. 실제 백엔드, 실기 어댑터, 실제 grip 센서 판정은 미구현. 실제 로봇과 연결해 본 적 없음.
