# ProtoMotions와 모션 저장·sim2sim 연결

확인일: 2026-09-08. 공식 ProtoMotions 3 커밋 `607ca7a0bb92e261120bcab8d9f97f28b3130ffc`의 실제 Python API로 변환과 로딩을 검증했습니다. 이 문서에서 설명하는 변환 환경에는 CPU PyTorch와 FK/모션 로더 의존성만 설치했습니다. 정책 학습이나 동역학 실행을 완료한 환경은 아닙니다.

## 형식 선택

**편집 원본 JSON과 참조 NPZ를 유지하고, ProtoMotions .motion/.pt를 파생 출력으로 추가합니다. BVH로 원본을 교체할 필요는 없습니다.**

| 형식 | 현재 역할 |
|---|---|
| 프로젝트 `.json` | 키프레임, 고정 상태, 현재 편집 자세와 박스 가이드 보존 |
| 참조 `.npz` | 로봇 관절·루트·링크 상태와 시간·단위·좌표계·접촉 의미를 보존하는 원본 참조 |
| ProtoMotions `.motion` | 대상 G1 모델로 FK를 다시 계산한 단일 모션 `RobotState` |
| ProtoMotions `.pt` | 공식 `MotionLib`로 패키징한 모션 라이브러리. 현재 내보내기는 모션 하나를 포함 |
| `.protomotions.json` | 대상 모델, 관절/링크 순서, 변환 규약, 원본 해시, 검증 범위 |
| BVH | 앞으로 Blender 등 DCC나 사람 모션과 교환할 때 선택적으로 추가할 애니메이션 형식 |

ProtoMotions의 [G1 CSV 안내](https://github.com/NVlabs/ProtoMotions/blob/607ca7a0bb92e261120bcab8d9f97f28b3130ffc/docs/source/getting_started/seed_g1_csv_preparation.rst)는 CSV → `.motion` → `.pt` 경로를 설명합니다. [BVH 안내](https://github.com/NVlabs/ProtoMotions/blob/607ca7a0bb92e261120bcab8d9f97f28b3130ffc/docs/source/getting_started/seed_bvh_preparation.rst)는 SOMA 사람 골격용 변환 경로이며 G1에 그대로 적용되는 형식 규약이 아닙니다. 이미 G1 관절각을 작성하고 있으므로 BVH를 거쳐 좌표·Euler 각·골격을 다시 변환하는 과정은 현재 필요하지 않습니다. BVH만으로 제어 주기, 모터 한계, 지지 상태 같은 제어 관련 의미를 모두 전달할 수도 없습니다.

## 적용한 변경

JSON 프로젝트 버전은 유지합니다. 편집기에서 새로 저장하는 NPZ는 Kimodo G1Skeleton34의 5개 배열(`posed_joints`, `global_rot_mats`, `local_rot_mats`, `root_positions`, `foot_contacts`)과 좌표계를 그대로 사용합니다. FPS, 모델 해시와 변환 정보는 같은 이름의 `.metadata.json`에 `reference_schema: motioncreator.reference.v2`로 기록합니다. 내부 로더는 이 배열을 아래 MuJoCo 참조 표현으로 복원하며, 이전 v1/v2 NPZ도 계속 읽습니다.

| 필드 | 의미 |
|---|---|
| `fps` | 참조 샘플링 주파수. 제어기 주기와는 별개 |
| `root_lin_vel_world`, `root_ang_vel_world` | 루트의 월드 기준 선속도·각속도 |
| `body_lin_vel_world`, `body_ang_vel_world` | 각 링크 원점의 월드 기준 속도. 질량중심 속도가 아님 |
| `body_parent_indices` | 저장된 `body_names` 순서 기준 부모 인덱스. 루트는 -1 |

복원된 `qvel[:, 3:6]`은 MuJoCo 자유 관절의 로컬 각속도입니다. 링크 속도는 MuJoCo Jacobian과 일반화 속도로 다시 계산합니다. 로더는 Kimodo 배열의 형상, 유한값, 회전 직교성, 로컬/전역 회전 일치 여부를 검사합니다. `control_binding`은 아직 연결한 제어기가 없으므로 null입니다.

ProtoMotions 출력에는 다음 절차를 적용합니다.

1. 확인한 소스 G1 모델과 대상 `g1_holo_compat.xml`의 해시 및 ProtoMotions 커밋을 검사합니다.
2. 29개 관절을 **이름으로 대응**합니다. 누락·중복·대상 관절 범위 초과는 실패로 처리합니다.
3. 루트 위치, 방향, 관절각, FPS와 길이를 보존합니다. 매 프레임 바닥 높이를 자동으로 바꾸지 않습니다.
4. 공식 `extract_kinematic_info`와 `fk_batch_mjcf_with_velocities`를 호출해 대상 링크 상태를 계산합니다. 대상은 `head`, `left_rubber_hand`, `right_rubber_hand` 등 편집기와 다른 링크를 포함합니다.
5. ProtoMotions의 **xyzw / 월드 링크 상태**로 저장합니다. NPZ의 wxyz 데이터를 파일명만 바꾸거나 무조건 배열 복사하지 않습니다.
6. 속도는 공식 전방 차분 API, horizon=1로 다시 계산합니다. 편집기의 중앙 차분 속도와 수치가 조금 다를 수 있으며 변환 보고서에 명시합니다.
7. 실제 `MotionLib`로 `.motion`을 로딩한 뒤 `.pt`를 저장합니다. 모든 출력에는 물리 균형·제어기 추종이 아직 미검증임을 기록합니다.

공식 [FK API](https://github.com/NVlabs/ProtoMotions/blob/607ca7a0bb92e261120bcab8d9f97f28b3130ffc/protomotions/components/pose_lib.py), [RobotState](https://github.com/NVlabs/ProtoMotions/blob/607ca7a0bb92e261120bcab8d9f97f28b3130ffc/protomotions/simulator/base_simulator/simulator_state.py), [MotionLib](https://github.com/NVlabs/ProtoMotions/blob/607ca7a0bb92e261120bcab8d9f97f28b3130ffc/protomotions/components/motion_lib.py)를 사용합니다. 여기서 API는 주로 Python의 모듈/클래스 인터페이스이며, URL에 모션을 보내면 WBC가 실행되는 REST 서비스라는 뜻은 아닙니다.

발 고정 플래그는 대응하는 ankle-roll 링크의 `rigid_body_contacts`로 매핑합니다. 이는 **작성자가 지정한 지지 상태**이며 반력 측정이나 충돌 검출 결과가 아닙니다. 모두 false인 경우 공식 MotionLib는 접촉 라벨을 제거하므로 `packaged_contacts_available`에 결과를 기록합니다. 실제 접촉 검증과 박스 하중은 이후 동역학 단계에서 처리해야 합니다.

## 사용 방법

편집기의 저장 영역에서 **저장 시 ProtoMotions .motion / .pt 추가**를 체크한 뒤 상단 **저장 / NPZ**를 누릅니다. JSON, NPZ, 일반 메타데이터와 함께 `.motion`, `.pt`, 변환 보고서가 `motions/`에 저장됩니다. 추가 변환이 실패해도 원본 JSON·NPZ는 저장하고 UI에 실패 이유를 표시합니다.

정지 자세는 같은 키프레임을 복제하고 예를 들어 3초를 지정하세요. ProtoMotions에서 시간 보간을 하려면 2개 이상의 샘플이 필요합니다. 한 프레임을 임의의 유지 시간으로 자동 확대하지 않습니다.

기존 NPZ만 변환하려면:

```bash
cd /home/kim/motioncreator
PYTHONPATH="$PWD" OPENBLAS_NUM_THREADS=1 .conda/bin/python -m motioncreator.cli protomotions motions/원본.npz
```

새 배치 모션과 함께 변환하려면:

```bash
PYTHONPATH="$PWD" OPENBLAS_NUM_THREADS=1 .conda/bin/python -m motioncreator.cli demo --fps 50 --protomotions
```

이미 동일한 이름의 변환 결과가 있으면 덮어쓰지 않습니다. 다른 출력 이름은 전용 환경에서 지정할 수 있습니다.

```bash
PYTHONPATH="$PWD" OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl .conda-protomotions/bin/python -m motioncreator.protomotions_bridge motions/원본.npz --output motions/다른이름.motion
```

파일만 복사해 사용할 때는 `.motion` 또는 `.pt`와 대응하는 `.protomotions.json`을 함께 전달하세요. `.pt`는 PyTorch 직렬화 파일이며 이 프로젝트에서는 직접 생성한 파일만 검증에 사용했습니다.

## 환경과 재현

- 편집기: `/home/kim/motioncreator/.conda`
- 변환·호환성 검사: `/home/kim/motioncreator/.conda-protomotions`
- 공식 소스: `/home/kim/motioncreator/external/ProtoMotions`
- 버전/모델 계약: `integrations/protomotions-profile.json`
- 의존성 잠금: `integrations/protomotions-requirements.lock.txt`

새 컴퓨터에서는 편집기를 설치한 다음 `./scripts/setup-protomotions.sh`를 실행합니다. 기존 다른 버전의 ProtoMotions checkout을 자동 변경하지 않습니다. 원본 소스와 라이선스를 보존하며, 프레임워크 코드 자체는 수정하지 않았습니다.

변환에는 시각 메시가 필요하지 않습니다. 현재 checkout의 메시 중 Git LFS 포인터인 파일은 실제 시뮬레이션을 시작할 때 별도로 받아야 합니다. 설치된 Git LFS로 대상 G1 메시를 받는 예시는 아래와 같습니다. 이것만으로 정책 실행에 필요한 전체 의존성·체크포인트가 설치되는 것은 아닙니다.

```bash
git -C external/ProtoMotions lfs pull --include='protomotions/data/assets/mesh/G1/**'
```

## WBC와 sim2sim에 연결할 때

ProtoMotions는 G1의 MuJoCo CPU와 Newton 등 여러 시뮬레이터에서 정책을 시험하는 경로를 제공하므로 향후 방향에 잘 맞습니다. 외부 WBC를 쓸 때는 해당 제어기의 관측/행동 어댑터가 필요합니다. 프레임워크 내부에서 학습한 정책의 시뮬레이터 교체 기능과 임의의 외부 정책 자동 호환은 구별해야 합니다. [공식 개요](https://github.com/NVlabs/ProtoMotions)

저장 FPS를 제어 주기로 간주하지 않습니다. 예를 들어 30fps 참조를 50Hz 제어기에 넣을 때 공식 `deployment.motion_utils.MotionPlayer`가 사용하는 위치 선형 보간·quaternion 보간 경로를 이용합니다. 이번 통합 테스트에서는 실제 MotionPlayer로 이 변환과 미래 참조 조회를 확인했습니다. 정책 추론이나 로봇 동역학을 실행한 검증은 아닙니다. [MotionPlayer](https://github.com/NVlabs/ProtoMotions/blob/607ca7a0bb92e261120bcab8d9f97f28b3130ffc/deployment/motion_utils.py)

다음 단계에는 사용할 WBC/체크포인트와 함께 아래 항목을 확정해야 합니다.

- 관절/링크 이름과 순서, heading/root 좌표계, 관측 정규화와 과거/미래 참조 구간
- 출력이 토크인지 목표 관절각인지, action scale과 기준 자세
- 제어 주기와 물리 step/decimation, Kp/Kd, 토크·속도 제한
- 해당 정책의 G1 모델 리비전, 발 접촉 형상·마찰, 추가 장착물과 박스 하중

ProtoMotions에서 내보낸 통합 ONNX 정책은 `deployment/test_tracker_mujoco.py` 경로가 있습니다. 실제 사용할 ONNX와 옆에 저장되는 설정 파일을 준비한 뒤 이 참조를 입력으로 연결하는 것이 다음 작업입니다. 사용자가 기존에 보유한 다른 WBC라면 그 입력 규약에 맞춰 연결해야 합니다. [공식 배포 안내](https://github.com/NVlabs/ProtoMotions/blob/607ca7a0bb92e261120bcab8d9f97f28b3130ffc/docs/source/tutorials/workflows/g1_deployment.rst)

## 검증

`./scripts/test-protomotions.sh`는 별도 CPU 환경에서 실제 공식 코드를 사용합니다. 대상 FK를 시각 메시만 제거한 동일 MJCF의 MuJoCo FK와 비교하고, 관절 순서 재배열, quaternion 순서, 모션 길이, 접촉 매핑, MotionLib 재로딩, 30→50Hz 배포 참조 보간을 검사합니다. 시각 메시 제거는 기구학 비교에만 사용하며 실제 충돌/동역학 모델 검증으로 해석하지 않습니다.

## 2026-09-10 모델 지문 정합성 확인

체크인된 현재 G1 XML의 지문(`7c0e0f81344c3cc3aa2b199b983c4ef049904c6e195d15199523c68f4ffc684e`)이 기존 변환 프로필 값과 달라 새 내보내기가 거절되는 문제를 확인했습니다. 현재 G1의 29관절과 공식 ARDY G1의 손목 FK/좌표 변환을 검증하고 현재 자산을 명시적인 소스 프로필로 등록했습니다. 이전에 사용한 알려진 지문도 별도 목록에 보존하여 과거 NPZ 변환을 유지하며, 임의의 다른 지문은 여전히 거절합니다. 변환 보고서에는 실제 소스 지문을 기록합니다.
