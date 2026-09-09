# G1 물리 모델과 MimicKit 적합성 검토

공식 저장소 확인일: 2026-09-08. 이 문서의 추천은 현재 목표가 '직접 만든 자세의 균형과 기존 WBC에서의 추종 가능성 확인'이고, 컴퓨터 자원이 제한되어 있다는 조건에 따른 판단입니다. MimicKit 설치나 정책 학습, 실제 동역학 시험은 수행하지 않았습니다.

## Menagerie G1은 무엇인가?

[MuJoCo Menagerie의 G1](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1)은 G1의 관절·질량·관성·충돌·구동 설정을 포함하는 MuJoCo용 물리 모델입니다. 모델과 시뮬레이터만으로 임의 자세를 안정화하는 제어기가 생기지는 않습니다. 공식 README는 일반 모델에 위치 구동기를 추가했으며 게인 조정이 필요하다고 명시합니다. MJX 변형에는 접촉 형상과 solver 설정, 더 낮은 PD 게인 등을 별도로 조정했습니다. [공식 모델 설명](https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/unitree_g1/README.md)

현재 편집기도 Unitree 공식 G1 MJCF에서 가져온 모델로 MuJoCo FK/IK를 계산합니다. Menagerie가 설명하는 원본은 `g1_29dof_rev_1_0.xml`이며 이 프로젝트의 원본은 로컬 `g1_description/g1_29dof.xml`이므로 동일 리비전이라고 단정하지 않습니다. 모델을 교체하려면 관절 순서·발 접촉 형상·질량/관성·구동기를 비교해야 합니다. 기존 모션의 모델 fingerprint도 달라집니다.

## MimicKit은 사용할 만한가?

**새 모션 추종 정책을 학습하려는 단계에는 적합합니다. 지금 자세의 정적 균형을 빠르게 판정하는 첫 도구로는 우선순위가 낮습니다.** MimicKit은 DeepMimic, AMP 등을 제공하는 강화학습 기반 모션 모방 제어기 학습 프레임워크입니다. 현재 설치 문서의 엔진은 Isaac Gym, Isaac Lab, Newton입니다. 현재 편집기의 Python MuJoCo 서버에 그대로 붙이는 검증 라이브러리로 설명되어 있지 않습니다. [공식 저장소](https://github.com/xbpeng/MimicKit/tree/main)

G1 지원도 실제로 있습니다. [G1 DeepMimic 실행 설정](https://github.com/xbpeng/MimicKit/blob/main/args/deepmimic_g1_ppo_args.txt)과 [G1 환경 설정](https://github.com/xbpeng/MimicKit/blob/main/data/envs/deepmimic_g1_env.yaml)은 G1 자산과 걷기 모션을 연결합니다. 기본 환경의 허용 접촉 목록에는 무릎도 포함되고 추종 오차 로그는 꺼져 있습니다. 따라서 그대로 실행한 성공 결과를 '양발로 서서 박스를 지지하는 시험 통과'로 해석하면 안 됩니다. 목표 작업에 맞춰 허용 접촉, 종료 기준, 오차 기록을 바꿔야 합니다.

참조 입력은 `.pkl`이며 루트 회전은 3D exponential map, 관절은 해당 XML의 트리 순서입니다. 현재 NPZ의 wxyz quaternion과 관절 이름을 대응시키는 변환이 필요합니다. README의 병렬 환경 4096개는 학습 예시 설정이며 필수 최소 개수는 아닙니다. 그래도 개별 자세 검사보다 정책 학습·튜닝에 더 많은 계산과 작업이 필요하다는 것이 이 프로젝트에서의 판단입니다. [입력 형식과 학습 안내](https://github.com/xbpeng/MimicKit#motion-data)

또한 참조 캐릭터를 직접 배치하는 시각화와 학습된 제어기로 물리 실행되는 캐릭터를 구별해야 합니다. 참조가 화면에서 넘어지지 않는 것 자체는 균형 검증이 아닙니다. [DeepMimic 환경 구현](https://github.com/xbpeng/MimicKit/blob/main/mimickit/envs/deepmimic_env.py)

학습된 제어기가 자세를 유지하면 '그 모델·제어기·하중·시험 조건에서 유지됐다'는 증거가 됩니다. 반대로 실패한 원인은 물리적 불가능 외에도 정책의 학습 범위, 게인, 입력 변환, 접촉 설정일 수 있습니다. 실패만으로 자세 자체가 불가능하다고 단정할 수 없습니다.

## 현재 프로젝트에서 추천하는 검증 순서

1. **정적 평형 검사**: MuJoCo에서 중력항과 접촉 Jacobian을 얻어 관절 토크·지면 반력을 계산합니다. 비구동 루트의 힘/모멘트 평형, 마찰, 발 지지면, 토크 한계를 함께 검사하고 여유를 표시합니다. CoM이 발 사이에 있는지는 빠른 보조 지표입니다.
2. **자세 유지 시험**: 자유 루트 상태에서 중력과 접촉을 켜고 먼저 PD+보상 토크, 이후 실제 WBC로 몇 초간 유지합니다. 참조 qpos를 매 프레임 덮어쓰지 않습니다. 기울기·미끄럼·추종 오차·토크 포화와 시험 조건을 저장합니다.
3. **전체 모션 추종 시험**: 웅크리기 → 들기 → 서기를 실제 WBC로 실행하고 박스 하중, 속도·가속도와 로코모션 전환까지 검사합니다.

정적 검사와 유지 시험은 기존 MuJoCo 환경에서 CPU로 시작할 수 있습니다. 구체적인 제약과 데이터 요구는 [물리 검증 설계](physics-validation-plan.md)에 있습니다. 새 범용 추종 정책을 학습할 필요가 생길 때 MimicKit을 별도 conda 환경에 도입하는 것이 적절합니다.

사용자의 예상처럼 정지·준정적 작업에서 접촉과 토크 여유가 있는 참조는 WBC가 추종하기에 유리할 것으로 기대합니다. 다만 보행 중에는 운동량과 다음 발 접촉을 이용하므로 모든 프레임에 정적 균형을 강제하면 유효한 동작도 제외됩니다. 자세별 정적 판정과 시간 흐름을 포함한 동적 판정을 분리해야 합니다.
