# 상자 태스크 생성과 CPU 물리 검증

2026-09-10 구현. 편집기 상단 **상자 태스크**에서 실행합니다. ARDY G1 가중치와 SONIC 인코더·디코더를 실제로 설치했으며, 두 모델 모두 CPU에서 실행합니다. 기존 편집기 환경 `.conda`와 별도로 `.conda-policy`를 사용합니다.

## 사용 순서

1. `./run.sh`로 실행하고 상자 태스크 화면을 엽니다.
2. **실제 상자**의 중심 위치·회전·크기·질량을 설정합니다. 이 값은 MuJoCo의 자유 물체에 적용됩니다.
3. **인식 결과**에 6D pose estimation 결과와 인식된 크기를 입력합니다. 모션 계획은 이 값을 사용합니다. 기본값은 실제 Y 폭 0.51 m / 인식 Y 폭 0.49 m입니다.
4. 목적지의 **상자 중심**과 팔레트 윗면 높이를 설정합니다. 예를 들어 상자 높이 0.40 m, 팔레트 높이 0.12 m이면 수평으로 안착한 상자 중심 높이는 0.32 m입니다. 크기 변경 시 중심 높이도 직접 맞춰야 합니다.
5. **태스크 프리셋 저장**으로 설정을 저장합니다. 다시 실행하면 저장된 태스크와 실행 기록을 불러올 수 있습니다.
6. **ARDY 모션 생성**을 실행합니다. 위치·회전·루트 경로 제약을 입력하며 자연어 인코더/8B LLM은 불러오지 않습니다. 생성 중에도 편집기 사용이 가능하며 취소할 수 있습니다.
7. 참조 모션을 재생하고 관절 범위 초과 및 손 접촉면 목표 오차를 확인합니다. **IK 배치 미리보기**는 빠른 기구학 배치 확인용입니다. 이 모드의 접근/운반 구간은 보행 정책으로 생성한 걸음이 아닙니다.
8. 생성된 참조 또는 기존 편집기의 NPZ를 선택하고 **GEAR-SONIC / CPU → MuJoCo 검증 실행**을 누릅니다.
9. 결과를 재생하면서 양손 압착력, 접선력, 접촉 미끄럼 속도, 손목 접촉 모멘트, 토크 포화 및 실패 단계를 확인합니다.

실제 상자는 갈색, 인식된 초기 상자는 청색 윤곽, 목표 접촉면은 작은 구형 마커입니다. 참조 재생에서 실제 상자는 임의로 손에 붙여 움직이지 않습니다. 물리 재생에서는 MuJoCo가 계산한 상자 위치와 회전을 표시합니다. 힘 화살표는 **상자가 손에 가하는 힘**이며 1 N당 5 mm로 표시하되 0.6 m에서 길이를 제한합니다.

## 물리 모델과 파지

기존 `assets/g1/g1.xml`은 변경하지 않습니다. 실행별로 별도의 `scene.xml`을 만듭니다.

- G1의 29개 토크 모터와 자유 루트를 유지합니다. 자유 루트에 외력을 가하거나 고정하지 않습니다.
- 기본 페이크핸드의 **실제 rubber-hand 메시를 충돌체로 추가**합니다. 메시 충돌은 MuJoCo의 볼록 형상에 따릅니다. 시각 메시의 세밀한 오목한 부분까지 정확한 접촉 형상으로 쓰는 방식은 아닙니다.
- 기본 관성에 SONIC의 공개 모터 상수에 따른 유효 rotor armature를 추가합니다. PD gains, action scale, 기본 관절각은 공개 배포 헤더에서 추출했고, 토크 제한은 현재 편집기 G1 모델 값을 유지합니다. 실제 하드웨어를 동정한 값은 아닙니다.
- 상자는 질량을 가진 강체이며 별도 free joint를 갖습니다. 손–상자 weld, equality, adhesion, mocap attachment, elastic band를 사용하지 않습니다.
- 손–상자는 명시적 `contact/pair`에 마찰계수를 설정해 기본 재질 혼합 규칙에 의해 값이 덮이지 않게 합니다. 바닥·팔레트 계수도 따로 적용합니다. `condim=3`으로 정상력과 두 방향 접선 마찰을 계산합니다.
- 파지점은 손목 중심이 아니라 페이크핸드 메시의 안쪽 지지면을 근사한 점입니다. 계획 시 이 오프셋을 손목 위치로 변환합니다. **49 cm는 두 접촉면 목표 사이 거리**이고, 추가 압착 간격은 인식 오차와 별도의 변수입니다.

49 cm를 명령한다고 51 cm 강체가 2 cm 줄어드는 것은 아닙니다. 충돌 제약과 유한한 관절 제어 토크가 반력을 만들며 실제 손 위치는 목표와 다를 수 있습니다. 접촉에는 MuJoCo의 유한한 수치적 침투가 있으므로 최대 침투량도 기록합니다. 상자의 찌그러짐, 손가락 내부 제어, 고무의 세부 변형, 힘 센서 기반 폐루프 압착 제어는 이번 PoC에 포함하지 않았습니다.

정적인 대칭 측면 파지의 참고값은 한 손당 `N ≥ mg / (2μ)`입니다. 이는 모멘트·가속도·불균일 접촉을 제외한 참고치입니다. 실제 성공 판정은 접촉과 물체 운동으로 합니다. 힘/모멘트 표시도 **상자 접촉 때문에 손목 원점에 생기는 외력 성분**이며 팔의 자중과 관성을 포함한 전체 손목 하중은 아닙니다.

## 생성과 제어의 연결

ARDY는 25 Hz 참조를 생성합니다. MuJoCo는 500 Hz로 적분하고 SONIC은 50 Hz로 관절 목표를 갱신합니다. CPU에서 계산이 늦어지면 시뮬레이션 시간의 진행도 늦어지며, 물리 시간을 벽시계에 맞추려고 스텝을 버리지 않습니다. 참조 회전은 quaternion SLERP로 샘플링합니다.

이 프로젝트의 SONIC 실행기는 **공식 ONNX 모델을 사용하는 CPU 관측/액션 어댑터**입니다. NVIDIA의 공식 C++/TensorRT 실행 파일을 실행한 것은 아닙니다. 원본 SONIC 릴리스의 G1 reference mode만 지원합니다. low-latency/v1.1 모델을 같은 자리에 바꾸면 안 됩니다.

어댑터가 따르는 공개 소스 계약:

- MuJoCo 순서 ↔ IsaacLab 순서의 명시적 29관절 permutation.
- G1 encoder mode는 `[0, 0, 0, 0]`입니다. one-hot `[1, 0, 0, 0]`이 아닙니다.
- encoder 입력 1762, token 64, decoder 입력 994, 출력 action 29.
- history는 오래된 프레임부터 10개이며 기본 관절각을 뺀 관측, 관절 속도, 이전 action, pelvis gyro와 gravity를 사용합니다.
- G1 미래 참조는 공개 C++의 `10frame_step5`를 따라 50 Hz 기준 0, 0.1, …, 0.9초를 봅니다. SMPL 모드의 lookahead 설명과 혼동하지 않습니다.
- action은 기준 관절각과 공개 scale로 변환한 후 유한한 PD 토크로 적용합니다. `PD 진단`은 별도 옵션이며 SONIC 검증 성공으로 기록하지 않습니다.

ARDY의 좌표는 +Y up / +Z forward, 이 프로젝트는 +Z up / +X forward입니다. 위치는 `(x,y,z) → (y,z,x)`로 변환하며 yaw와 ARDY heading은 같은 부호입니다. 공식 G1 converter와 현재 모델의 손목 FK가 일치하는지 수치 테스트합니다. ARDY의 제약은 생성 모델의 조건이며, 정확한 IK 해나 동적 실행 가능성을 보장하지 않습니다. 관절 범위 초과도 원본 참조에 그대로 기록하며 숨겨서 잘라내지 않습니다.

## 태스크 실행과 성공 판정

`approach → crouch → pregrasp → grasp → lift → stand → carry → lower → place → release → retreat`의 11단계입니다. 단계 끝에서 참조 시계를 멈추고 측정 조건이 0.2초 연속 유지될 때 다음 단계로 넘어갑니다. 조건을 기다리는 시간에도 SONIC과 물리는 계속 실행합니다.

| 단계 | 주요 측정 조건 |
|---|---|
| 접근/후퇴 | 루트 XY·방향·높이·속도. 후퇴 완료 시 상자가 여전히 안착되어 있어야 함 |
| 자세 변화 | 관절 목표 오차와 루트 높이 |
| 파지 | 양손 정상력, 반대 방향 측면 압착 및 낮은 접촉 미끄럼 속도 |
| 들어 올리기 | 양손 접촉 유지, 상자 상승, 받침의 지지력 감소 |
| 운반 | 양손 접촉 및 상자 목표 위치 도달 |
| 안착 | 팔레트 지지력, 물체 위치·방향·선속도·각속도 |
| 손 떼기 | 양손 접촉력 감소, 팔레트에서 안정된 상태 유지 |

넘어짐·낙하·시간 초과는 실패로 기록합니다. `job.status == completed`는 계산 종료라는 뜻이고, 성공은 `job.result.validated == true`로 확인해야 합니다. 현재 검증은 설정한 한 상자와 한 사이클에 대한 것입니다. 다음 상자 선택, SLAM/6D 추정, 실제 로봇 SDK navigation은 상위 시스템의 역할입니다. 새 추정값으로 태스크를 생성하고 성공 결과를 확인한 뒤 다음 사이클을 요청하는 API를 제공합니다. 재시도 정책이나 자동 다중 상자 큐는 구현하지 않았습니다.

PoC 시뮬레이션에서는 접근/운반/후퇴도 생성된 참조를 SONIC으로 추종합니다. 실제 시스템에 연결할 때는 navigation 도착 확인 후 전신 제어 권한을 SONIC에 넘기는 소유권 전환이 필요합니다. 두 제어기가 동시에 전신 모터 명령을 내리는 구성은 이 구현에 없습니다.

## 저장 형식과 API

기존 편집기의 JSON/NPZ 및 ProtoMotions 변환은 유지합니다. 새 데이터는 다음처럼 저장합니다.

```text
tasks/<task-id>/task.json                       # 다시 불러올 작업 설정과 계획
tasks/<task-id>/runs/<run-id>/request.json       # 실행 당시의 입력 스냅샷
                            reference.npz      # 기존 v2 참조 규약, pickle 없음
                            reference.metadata.json
                            ardy-conditions.npz # ARDY에 실제로 전달한 조건/마스크
                            constraint-errors.json
                            scene.xml           # 해당 실행의 물리 장면
                            report.json         # 제어 설정, 모델/참조 지문, 성공/실패
                            contact-log.json    # 50 Hz 접촉/단계 판정 로그
                            simulation.npz      # 25 Hz 실제 qpos/qvel/토크/상자 상태
                            replay.json         # 화면 재생 정보
                            status.json / result.json / worker.log
```

참조 NPZ의 quaternion은 wxyz입니다. 시뮬레이션 로그는 로봇 루트 wxyz, 상자 quaternion xyzw로 명시합니다. `simulation.npz`는 관측 로그이며 참조 NPZ와 구분합니다. `scene.xml`의 메시 경로는 이 컴퓨터의 `assets/g1/meshes`를 참조합니다.

설정과 시작 자세가 바뀌면 task 지문이 바뀌며 이전 생성 결과로 새 검증을 시작하는 요청을 거절합니다. 기존 실행 기록은 덮어쓰지 않습니다. report에는 실제 사용한 가중치 해시, 참조 해시, 물리 장면 해시, PD gains와 토크 제한도 남습니다.

| API | 역할 |
|---|---|
| `GET /api/tasks/defaults` | 입력 템플릿 |
| `GET /api/tasks/runtime` | CPU 환경/가중치 파일 및 기존 NPZ 목록 |
| `POST /api/tasks/plan` | `{spec, start_qpos?}` → 상자 기준 목표 |
| `POST /api/tasks` | `{spec, start_qpos?, id?}` → 프리셋 저장 |
| `GET /api/tasks`, `GET /api/tasks/{id}` | 목록과 재로딩 |
| `POST /api/tasks/{id}/runs` | `{kind: "ardy"}` 또는 `"ik_preview"` |
| 같은 URL | `{kind: "simulate", reference_run: "...", controller: "sonic"}` |
| 같은 URL | 기존 편집기 NPZ 사용 시 `reference_file: "파일명.npz"` |
| `GET /api/tasks/{id}/runs/{run}` | 진행률과 결과 |
| `POST /api/tasks/{id}/runs/{run}/cancel` | 취소 요청 |
| `GET /api/tasks/{id}/runs/{run}/replay` | 참조/물리 재생 |
| `GET /api/tasks/{id}/runs/{run}/files/{name}` | NPZ·로그·보고서 다운로드 |

`spec.perceived.pose`에 `{position: [x,y,z], quaternion_xyzw: [x,y,z,w]}`를 넣습니다. 상자 중심 좌표이며 단위는 m입니다. 입력 좌표는 미리 로봇 시뮬레이션 world frame으로 변환해야 합니다. 첫 PoC는 수평 받침 위 상자이며 up 방향 기울기는 10° 이내로 제한합니다.

## 설치 재현과 확인

이 컴퓨터에는 이미 설치했습니다. 새 환경에서는:

```bash
cd /home/kim/motioncreator
./scripts/setup-task-cpu.sh
npm --prefix frontend run build
./run.sh
```

`integrations/task-cpu-requirements.lock.txt`, `task-models.manifest.json`, `task-source-provenance.json`, `sonic-parameters.json`이 환경과 모델 출처를 기록합니다. 추론 가중치는 약 826 MiB이며, 학습 데이터·SMPL 데이터·텍스트 LLM은 받지 않았습니다. 기존 `.conda`와 `.conda-protomotions`의 패키지는 바꾸지 않았습니다.

모델 설치는 저장소에 고정된 크기와 SHA-256을 검증합니다. 기존 파일이 다르거나 다운로드가 손상되면 설치를 중단하며, 체크섬 기준을 자동으로 덮어쓰지 않습니다.

확인 항목:

- 기존 편집기/API 및 새 태스크 계약 테스트.
- 실제 SONIC CPU로 자유 G1의 5초 정지 자세 유지. 외부 지지력 없음.
- 공식 ARDY converter와 현재 G1의 관절/손목 FK 및 회전 좌표 변환 일치.
- 별도 두 패드 접촉 시험: 실제 51 cm / 명령 49 cm, 마찰 0.8에서 1 kg 상자를 유지, 마찰 0에서 낙하. **G1 팔레타이징 성공을 의미하는 시험은 아닙니다.**
- 실제 ARDY 22초 후보 생성 및 로컬 API를 통한 SONIC/MuJoCo 실행·재생 응답 확인. 초기 실행은 접근 후 웅크리기에서 넘어졌고, 최종 마찰 설정을 반영한 실행은 접근 조건 시간 초과로 종료됐습니다. 기본 후보의 팔레타이징 성공은 아직 검증되지 않았습니다.
- 브라우저 화면의 수동/자동 조작 테스트는 수행하지 않았습니다. 프런트엔드 타입 검사와 production build를 확인했습니다.

```bash
./scripts/test.sh
./scripts/test-task-cpu.sh
./scripts/test-protomotions.sh
npm --prefix frontend run build
```

## 현재 컴퓨터의 실행 기록

- 태스크: `8db098d67fcf` (CPU PoC · 실제 폭 51 cm / 인식 폭 49 cm).
- ARDY 실행 `4a30e2417fc6`: 25 FPS, 22초/551프레임, 전체 worker 시간 **59.36초**. 모델 로딩과 저장을 포함합니다.
- SONIC 실행 `3e21ab3b408b`: 물리 시간 **5.02초**, 제어/적분 루프 시간 **1.31초**. 결과는 `timeout:root_arrival`, `validated=false`입니다. 125개의 실제 상태 재생 프레임과 토크 채널을 저장했습니다. 렌더링을 포함한 실시간 성능 보장은 아닙니다.
- 취소 실행 `4cf4c969d5b3`: API 요청으로 취소되고 `cancelled` 상태로 기록되는 것을 확인했습니다.

이미 상자 앞에 도착한 자세부터 검토하려면 편집기에서 해당 시작 자세를 만들고 태스크 설정의 **편집기의 현재 자세를 시작점으로**를 누를 수 있습니다. 시작 자세 변경도 재생성·재검증 대상입니다.

## 근거 자료

- [ARDY 공식 코드와 G1 모델](https://github.com/nv-tlabs/ardy) — 수치 제약과 텍스트 조건을 지원하는 생성기. 이 구현은 수치 조건을 사용합니다.
- [GEAR-SONIC 공식 저장소](https://github.com/NVlabs/GR00T-WholeBodyControl) — 관측 순서, action 변환, 모델과 모터 상수의 출처.
- [MuJoCo 접촉 파라미터](https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters) 및 [명시적 contact pair](https://mujoco.readthedocs.io/en/stable/XMLreference.html#contact-pair) — 마찰과 접촉력 모델 설정.
