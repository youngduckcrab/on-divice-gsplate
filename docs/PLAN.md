# 온디바이스 Gaussian Splatting 앱 (Android) — 개발 계획서

| 항목 | 내용 |
|---|---|
| 문서 버전 | v0.1 (초안) |
| 작성일 | 2026-10-30 |
| 대상 플랫폼 | Android (arm64) — 1차 목표는 사용자 테스트 기기 1대 |
| 진행 방식 | Claude가 구현 → APK 배포 → 사용자가 설치·실험 → 피드백 → 반복 |
| 상태 | 계획 검토 중 → 승인 후 M0 착수 |

---

## 1. 개요

### 1.1 목표

스마트폰으로 주변을 30~60초 촬영하면, 서버 없이 **폰 안에서** 수 분 이내에 3D Gaussian Splatting(3DGS) 모델을 학습하고, 바로 돌려보며 확인할 수 있는 안드로이드 앱을 만든다.

레퍼런스는 Niantic의 **Scaniverse**다. Scaniverse는 AR 프레임워크의 카메라 포즈를 이용해 촬영하고, 학습을 기기 안에서 수행하며, iOS 기준 처리 시간은 약 1분 내외로 알려져 있다.

### 1.2 범위

**포함**
- ARCore 기반 촬영 (이미지 + 6DoF 카메라 포즈 + 깊이)
- 온디바이스 3DGS 학습 (Vulkan compute)
- 온디바이스 실시간 뷰어
- PLY / SPZ 내보내기 및 공유
- 실험용 기능: 학습 파라미터 조절, 성능 계측, 진단 리포트 내보내기

**제외 (현 단계)**
- iOS, 서버/클라우드 학습
- Play 스토어 배포 (APK 직접 설치로 진행)
- 메쉬 추출, 동적 장면, 건물 단위 대규모 장면
- ARCore 미지원 기기 (추후 온디바이스 SfM 경로로 검토)

### 1.3 성공 기준

> 아래 수치는 **초기 가설**이다. M0 기기 벤치마크와 M3 실측 결과로 재조정한다.

| 지표 | 1차 목표 (M3~M4) | 최종 목표 (M5~M7) |
|---|---|---|
| 촬영 시간 | 30~90초 | 30~60초 |
| 학습 시간 (촬영 종료 → 결과 확인) | ≤ 5분 | **≤ 2분** (Scaniverse급) |
| 화질 (hold-out 프레임 PSNR) | ≥ 22 dB | ≥ 25 dB |
| 뷰어 프레임레이트 | ≥ 30 fps @ 가우시안 30만 개 | ≥ 60 fps @ 가우시안 50만 개 |
| 학습 중 최대 메모리 | < 3 GB | < 2 GB |
| 결과 파일 크기 (SPZ) | — | < 20 MB |
| 안정성 | 표준 장면 3종 학습 완료 | 표준 시나리오에서 크래시·GPU 디바이스 로스트 0회 |

---

## 2. 진행 방식: 빌드 → 설치 → 피드백 루프

### 2.1 사이클

```
 ┌───────────── Claude ─────────────┐         ┌───────────── 사용자 ─────────────┐
 │ 1. 마일스톤 기능 구현             │         │ 4. 폰에서 APK 다운로드·설치       │
 │ 2. CI: 빌드 + 단위 테스트 +       │ ──────► │ 5. 릴리즈 노트의 체크리스트 수행   │
 │    GPU 커널 정합성 테스트          │         │ 6. 결과·스크린샷·진단 패키지를     │
 │ 3. GitHub Release에 APK 게시      │ ◄────── │    피드백 (Issue 또는 대화)        │
 │ 7. 피드백 분석 → 수정 / 다음 단계 │         └──────────────────────────────────┘
 └──────────────────────────────────┘
```

### 2.2 APK 배포

- **GitHub Actions**: push마다 APK 빌드와 테스트를 돌리고, 태그(`v0.X.Y`)를 push하면 GitHub Release를 만들어 APK를 첨부한다.
- 파일명은 `gsplat-v0.X.Y-<commit>.apk`. 앱의 "정보" 화면에 버전과 커밋 해시를 표시한다.
- **서명 키 고정**: 실험용 고정 키스토어를 저장소에 포함한다. 새 버전을 기존 앱 위에 덮어 설치할 수 있어서 매번 삭제할 필요가 없고 스캔 데이터도 유지된다. (공개 배포용 키가 아님)
- **설치 방법**: 폰 브라우저에서 Release 페이지 → APK 다운로드 → "출처를 알 수 없는 앱 설치" 허용. 저장소가 private이면 폰 브라우저에서 GitHub 로그인이 필요하다.

### 2.3 피드백 방법

매 릴리즈 노트에 **변경 사항 / 테스트 체크리스트 / 알려진 문제**를 적는다. 사용자는 아래 양식으로 GitHub Issue를 열거나 대화에 붙여 넣는다.

```
버전: v0.X.Y (앱 정보 화면의 커밋 해시)
기기: 모델명 / Android 버전
체크리스트: 1 ✅  2 ❌  3 ✅ ...
문제: 무엇을 했을 때 → 무엇이 일어났는지 (스크린샷·화면녹화 첨부)
수치: 앱이 표시한 학습 시간 / FPS / PSNR 등
첨부: 진단 패키지 zip
의견: 자유롭게
```

- **진단 패키지** (앱에서 버튼 하나로 생성, 수 MB 이내 목표): 기기·GPU 정보(JSON), 앱 로그, 학습 지표(손실·PSNR·단계별 시간·온도), 사용한 파라미터, 카메라 궤적, 저해상도 썸네일 몇 장.
- **스캔 데이터셋 전체 내보내기** (필요할 때만): nerfstudio `transforms.json` 형식. PC에서 gsplat / nerfstudio / Brush로도 학습해 결과를 비교할 수 있다.

### 2.4 역할 분담과 한계

| | 가능 | 불가능 |
|---|---|---|
| Claude | 빌드, 단위 테스트, **소프트웨어 Vulkan(Mesa lavapipe)으로 GPU 커널 정합성 테스트** (CPU 레퍼런스 구현과 forward/backward 수치 비교), 사용자 데이터셋으로 엔진의 Linux 빌드 재현 (느리지만 결정적) | 실제 기기에서의 실행, 성능·발열 측정, ARCore 동작 확인, 기기 드라이버 버그 확인 |
| 사용자 | 실제 기기 테스트, 촬영, 체감 품질 평가 | — |

그래서 엔진은 처음부터 **Android와 Linux에서 함께 빌드되는 C++ 코드**로 작성하고, 모든 단계에 계측을 넣는다.

---

## 3. 배경 조사와 시사점

| 레퍼런스 | 내용 | 시사점 |
|---|---|---|
| **Scaniverse** (Niantic) | AR 프레임워크 포즈 기반 촬영 → 온디바이스 학습. iOS에서 처리 약 1분 내외. 내부 구현은 비공개 | 목표 지점. 무거운 SfM을 생략하고 AR 포즈를 그대로 쓰는 것이 속도의 핵심으로 추정 |
| **PocketGS** (arXiv 2601.17354, 2026) | 모바일 온디바이스 3DGS 학습 논문. 정보량 기반 프레임 선별 + GPU BA + MVS로 기하 prior 구성, Adam 업데이트를 단일 fused kernel로 처리. iPhone 15에서 500 iteration ≈ 4분, 피크 메모리 < 3 GB 보고 | 좋은 초기화로 iteration 수를 줄이는 것이 핵심. fused optimizer. 메모리 예산의 근거 |
| **Brush** (Rust, Burn + wgpu) | 크로스플랫폼 3DGS 학습·뷰어, Android 지원(실험적). WebGPU에 float atomic이 없어 CAS loop로 에뮬레이션 | Android GPU에서 학습이 가능함을 보여줌. 커널 구조 참고 + 같은 기기 성능 기준선 |
| **gsplat** (nerfstudio, Apache-2.0) | CUDA 3DGS 라이브러리. 수식과 backward가 깔끔하게 정리됨 | 래스터라이저 수식의 주 레퍼런스 |
| **원본 3DGS** (Inria) | 비상업 라이선스 | 코드는 가져오지 않고 논문 수식만 참고 |
| **3DGS-MCMC, Taming 3DGS** | 가우시안 개수 예산을 고정한 densification | 모바일에서 메모리가 예측 가능 → 채택 후보 |
| **FastGS** (CVPR 2026) | 데스크톱에서 약 100초 학습. 다중 시점 일관성 기반 densify/prune | 적은 iteration에서의 densification 전략 참고 |
| **SPZ** (Niantic, MIT) | Scaniverse의 압축 포맷, PLY 대비 약 90% 작음 | 내보내기 포맷으로 채택 |

**핵심 시사점**
1. **SfM(COLMAP) 생략** — ARCore 포즈와 깊이를 그대로 써서 전처리 시간을 거의 0으로 만든다.
2. **좋은 초기화 = 적은 iteration** — ARCore Depth로 조밀한 포인트 클라우드를 만들어 초기화한다.
3. **고정 예산** — 가우시안 수 상한, 낮은 학습 해상도, 낮은 SH 차수로 시작해 점진적으로 올린다.
4. **모바일 GPU 특성 대응** — 메모리 대역폭이 데스크톱의 1/10~1/15 수준이므로 커널 fusion, fp16, sparse 업데이트가 중요하다.

---

## 4. 시스템 아키텍처

### 4.1 전체 파이프라인

```
[촬영]   ARCore 세션 — 프레임마다 이미지, 카메라 포즈, intrinsics, depth + confidence, 추적 상태
           │  키프레임 선택 (이동량 · 회전량 · 선명도 · 추적 품질)
           ▼
[저장]   scans/<id>/  images/*.jpg, depth/*.bin, frames.json (포즈·intrinsics), meta.json
           ▼
[초기화] depth 역투영 → 월드 좌표 포인트 → voxel 병합 → 색 · 크기 · 불투명도 초기값
           ▼
[학습]   Vulkan compute: 전처리 → 정렬 → forward 래스터 → loss → backward → Adam → (densify/prune)
           │  실시간 미리보기, 손실 / PSNR 그래프, 단계별 시간 · 온도 계측
           ▼
[결과]   뷰어 (Vulkan graphics)  ·  PLY / SPZ 내보내기  ·  공유
```

### 4.2 기술 스택

| 영역 | 선택 | 이유 |
|---|---|---|
| UI | Kotlin + Jetpack Compose | 표준 스택, 화면 구성이 빠름 |
| 촬영 | ARCore SDK (Kotlin API) + OpenGL ES 카메라 프리뷰 | 6DoF 포즈와 깊이를 바로 얻을 수 있음. ARCore 카메라 텍스처가 GLES 기반이라 공식 샘플 구조를 따름 |
| 엔진 | C++20 + Vulkan 1.1 (NDK, CMake) | 모바일 GPU를 가장 세밀하게 제어(subgroup, fp16, 메모리). Linux에서 같은 코드로 테스트 가능 |
| 셰이더 | GLSL → SPIR-V (빌드 시 glslc) | NDK에 포함된 툴체인 |
| 라이브러리 | VMA (MIT), glm (MIT), libspz (MIT), stb (public domain) | 라이선스가 단순 |
| CI | GitHub Actions (Android 빌드 + Linux/lavapipe 테스트) | 기기 없이 자동 검증 |
| 최소 사양 | Android 10 (API 29) 이상, arm64, ARCore 지원 기기, Vulkan 1.1 | 열 관리 API 등 상위 API는 런타임에 확인 후 사용 |

**엔진 구현 방식 비교 — 결정: A안**

| 안 | 장점 | 단점 |
|---|---|---|
| **A. 자체 C++/Vulkan 엔진** | 최대 성능과 제어권, 앱 통합이 단순(JNI), lavapipe로 CI 테스트 가능 | 구현량이 가장 많고 드라이버 차이를 직접 대응해야 함 |
| B. Brush(Rust/wgpu) 포크·임베드 | 동작하는 학습기를 빨리 확보 | wgpu 추상화 때문에 모바일 최적화에 한계(float atomic 없음 등), Rust + Burn 빌드 체인 추가, 촬영 파이프라인과의 통합 비용 |
| C. ExecuTorch / MNN 등 모바일 ML 런타임 | 익숙한 ML 스택 | 미분 가능한 래스터라이저를 결국 커스텀 op로 직접 작성해야 하고, 학습 지원이 미흡 |

A안으로 구현하고, Brush는 커널 구조 참고와 (필요하면) 같은 기기에서의 성능 기준선으로 활용한다.

### 4.3 저장소 구조 (예정)

```
on-divice-gsplate/
├── app/                    # Android 앱 (Kotlin, Compose, ARCore)
├── engine/                 # C++ Vulkan 엔진 — Android · Linux 공용
│   ├── include/  src/      # Vulkan 컨텍스트, 버퍼, 파이프라인, 3DGS 학습·렌더, IO
│   ├── shaders/            # GLSL compute / graphics 셰이더
│   ├── tests/              # CPU 레퍼런스 vs GPU 커널 정합성 테스트 (lavapipe)
│   └── cli/                # Linux CLI: 데이터셋 학습·벤치마크 재현
├── tools/                  # PC 보조 스크립트 (데이터셋 변환·분석·비교)
├── docs/
│   ├── PLAN.md             # 이 문서
│   └── experiments/        # 버전별 테스트 결과·피드백 기록
└── .github/                # CI 워크플로, 피드백 이슈 템플릿
```

### 4.4 데이터 포맷과 좌표계

- **스캔 폴더**: 4.1의 구조. `frames.json`에 프레임별 `timestamp, pose (4×4 camera-to-world), fx, fy, cx, cy, width, height, tracking_state, sharpness`를 저장한다.
- **좌표계**: ARCore 월드 좌표(오른손 좌표계, +Y가 중력 반대 방향, 미터 단위). ARCore 카메라는 OpenGL 관례(−Z가 전방)이고, 엔진 내부는 OpenCV 관례(+Z가 전방)로 변환해서 쓴다. 변환은 한 곳에서만 처리한다.
- **이미지와 포즈의 짝**: CPU 이미지는 센서 방향(가로) 기준이므로 `Camera.getImageIntrinsics()`와 디스플레이 회전이 적용되지 않은 `Camera.getPose()`를 짝지어 쓴다. M2에서 depth 재투영 테스트로 검증한다.
- **내보내기**: nerfstudio `transforms.json` (OpenGL 관례 camera-to-world) — PC 도구와 호환.

---

## 5. 핵심 컴포넌트 설계

### 5.1 촬영 (ARCore)

- **세션 설정**
  - 깊이: 지원 기기면 Raw Depth(+confidence) 사용, 미지원이면 특징점만 사용
  - 포커스 고정 시도 (초점 변화로 intrinsics가 흔들리는 것 방지)
  - CPU 이미지 해상도: 기본은 640×480. `Session.getSupportedCameraConfigs()`에서 기기가 지원하는 가장 높은 해상도를 고른다 (지원 목록은 M0 진단으로 확인)
- **키프레임 선택**
  - 추적 상태가 `TRACKING`인 프레임만 사용
  - 직전 키프레임 대비 이동량이 장면 깊이의 약 5% 이상이거나 회전 5~10° 이상일 때 채택
  - 선명도: Y 채널 Laplacian 분산이 최근 중앙값보다 크게 낮으면 제외. 포즈 차분으로 각속도를 구해 너무 빠르면 제외
  - 최대 키프레임 수 제한 (기본 150~250, 실험 파라미터로 노출)
- **촬영 가이드 UI**: 추적 상태, 키프레임 수, 속도·조도 경고, (물체 모드) 대상 주변 커버리지 표시
- **저장**: JPEG 인코딩은 백그라운드 스레드에서. depth / confidence 원본, ARCore 특징점도 함께 저장
- **후순위 고급 기능**: Shared Camera(Camera2)로 고해상도 프레임 획득, ARCore Recording(MP4)으로 세션을 기록해 두고 키프레임 설정만 바꿔 재처리

### 5.2 초기화

- 각 키프레임의 raw depth를 confidence 임계값으로 거른 뒤 월드 좌표로 역투영하고, voxel grid로 병합한다 (평균 색, 관측 횟수 누적). 여러 시점에서 반복 관측된 점만 남긴다.
- 초기값: scale = 이웃 점 간 거리(또는 voxel 크기), rotation = identity, opacity = 0.1~0.5, SH DC = 픽셀 색.
- **Depth 미지원 기기**: ARCore 특징점을 누적하고 장면 범위 안에 랜덤 점을 보충한다 (품질·속도 저하 예상).
- (M6) PocketGS처럼 다중 시점 일관성 검사나 경량 MVS로 prior 품질을 높이는 방안을 검토한다.

### 5.3 학습 엔진 (Vulkan compute)

**iteration 1회의 흐름**

1. **전처리** — scale·quaternion으로 3D 공분산 → EWA 투영으로 2D conic, 반경, 타일 범위 계산 → SH에서 색 계산
2. **정렬** — 깊이로 먼저 정렬한 뒤 타일 ID로 안정 정렬한다. 32-bit 키만 써서 64-bit 정수(`shaderInt64`) 지원에 의존하지 않는다.
   - 기기마다 subgroup 크기가 16~128로 다르므로 이에 의존하지 않는 radix sort를 쓴다. GPU 간 forward-progress 보장이 필요한 onesweep 방식은 쓰지 않는다.
3. **Forward 래스터** — 16×16 타일, 픽셀별 front-to-back alpha blending. 최종 투과율과 마지막 기여 인덱스를 저장한다.
4. **Loss** — (1−λ)·L1 + λ·D-SSIM (λ = 0.2). SSIM은 separable convolution compute 셰이더로 fused 처리한다.
5. **Backward 래스터** — back-to-front로 다시 순회하며 가우시안별 그래디언트를 계산한다. 그래디언트 누적 방식 3가지를 구현하고 기기별 벤치마크로 고른다.
   - (a) subgroup 합산 후 float atomic add (`VK_EXT_shader_atomic_float` 지원 시)
   - (b) subgroup 합산 후 CAS loop로 float atomic 에뮬레이션
   - (c) **atomic-free**: (타일, 가우시안) 쌍별 부분 그래디언트를 정렬 전 인덱스 위치에 기록 → 가우시안마다 연속 구간을 합산
6. **Backward 전처리** — 2D 그래디언트를 3D 파라미터(위치, scale, rotation, opacity, SH) 그래디언트로 변환
7. **Adam** — 단일 fused kernel. 이번 iteration에 화면에 보이지 않은 가우시안은 건너뛴다(sparse). 위치 학습률은 장면 크기 기준으로 지수 감쇠
8. **Densify / Prune** (N iteration마다) — 개수 예산을 고정한다. 3DGS-MCMC식 재배치(relocation)를 1순위로 하고, Taming식 점수 기반 방식과 비교 실험한다.

**점진 스케줄**: 학습 해상도 1/4 → 1/2 → 1, SH 0차 → 1차(→ 2차), 가우시안 예산 점증.

**메모리 예산** (가우시안 1개당)
- 파라미터는 SH 1차 기준 23 float(약 92 B). 그래디언트와 Adam 모멘트 2개까지 포함하면 약 0.4 KB. SH 3차면 약 1 KB.
- 가우시안 30만 개 → 120~300 MB. 여기에 타일 쌍·정렬 버퍼 수십~100 MB, 학습 이미지(200장 × 640×480 RGBA ≈ 245 MB).
- **전체 1 GB 안팎**을 목표로 한다.

**안정성**: Android GPU 드라이버는 너무 긴 dispatch를 멈춤으로 판단해 `VK_ERROR_DEVICE_LOST`를 낼 수 있다. dispatch를 짧게 나누고, 학습 중 UI 미리보기는 낮은 빈도로 그린다.

### 5.4 뷰어

- Vulkan graphics 파이프라인: 가우시안마다 인스턴스 쿼드를 그리고, 프래그먼트 셰이더에서 2D 가우시안을 평가한다. premultiplied alpha blending.
- 정렬: 학습기와 같은 GPU radix sort. 카메라가 크게 움직였을 때만 다시 정렬하는 옵션.
- 렌더링용 데이터는 fp16으로 압축해 대역폭을 줄인다.
- 조작: 오빗 / 팬 / 줌, 시점 초기화, 학습 카메라 위치로 이동.
- HUD: FPS, 가우시안 수, GPU 시간.

### 5.5 내보내기

- **PLY**: 표준 3DGS 속성 레이아웃 — SuperSplat 등 외부 뷰어와 호환
- **SPZ**: libspz 사용
- Android 공유 시트(FileProvider)로 전송
- **스캔 데이터셋 zip** (nerfstudio 형식): PC 비교·디버깅용

### 5.6 발열 · 전력 · 메모리

- 사용 가능한 API를 런타임에 확인해 쓴다: 열 상태 리스너(API 29+), `PowerManager.getThermalHeadroom()`(API 30+), ADPF `PerformanceHintManager`(API 31+).
- 열 여유가 줄어들면 학습 해상도나 작업량을 줄이고, 미리보기를 멈추고, 필요하면 일시 정지 후 재개한다.
- v1은 학습 중 화면 켜짐을 유지한다. 이후 Foreground Service로 백그라운드 학습을 검토한다.
- 학습 시작 전에 가용 메모리를 확인해 가우시안 예산을 자동으로 정한다. OOM이나 디바이스 로스트가 나면 마지막 체크포인트에서 재개한다.

### 5.7 실험 기능 (사용자가 직접 실험하는 부분)

- **학습 프리셋**: 빠름 / 균형 / 고품질
- **고급 설정**: iteration 수, 해상도 스케줄, 가우시안 예산, SH 차수, densify 방식, 학습률, loss 가중치, atomic 전략, 타일 크기
- 결과마다 사용한 파라미터·기기 상태·지표를 `metrics.json`으로 함께 저장한다. 같은 스캔으로 설정만 바꿔 다시 학습하고 결과를 나란히 비교하는 화면을 제공한다.
- **벤치마크 모드**: 고정된 합성 장면으로 단계별 커널 시간을 잰다 (기기 간·버전 간 비교 기준)
- **개발자 오버레이**: 단계별 GPU 타임스탬프, 메모리, 온도

---

## 6. 성능 예산 추정 (가설)

- 데스크톱 고성능 GPU에서 640×480 이미지, 가우시안 20만 개 규모의 iteration은 수 ms 수준이다. 모바일 GPU는 연산 성능이 약 1/20, 메모리 대역폭이 약 1/10~1/15이므로 **iteration당 50~120 ms**를 예상한다.
- 그러면 2분 동안 1,000~2,400 iteration을 돌릴 수 있다. PocketGS는 좋은 prior로 500 iteration에서도 쓸 만한 결과를 보고했다.
- **전략**
  1. depth 기반 초기화로 1,000~2,000 iteration 안에 수렴시킨다.
  2. 초반 iteration은 저해상도로 싸게 돌린다.
  3. 커널 최적화로 iteration 시간 자체를 줄인다.
- M0 마이크로벤치마크(정렬, 대역폭)와 M3 실측으로 이 추정치를 갱신한다.

---

## 7. 마일스톤

각 마일스톤은 **APK 1개 이상**을 배포하고, 사용자 테스트 결과로 완료 여부를 판단한다. 기간은 날짜 대신 **예상 피드백 사이클 수**로 적는다.

### M0 — 기반 구축 & 기기 진단 `v0.1` · 1~2 사이클

**목표**: 빌드·배포 루프를 확립하고, 사용자 기기의 GPU / ARCore 능력을 파악한다.

**구현**
- Android 프로젝트 골격 (Compose, NDK/CMake, JNI), Vulkan 초기화
- GitHub Actions: APK 빌드 · Release 게시, Linux 엔진 빌드 · lavapipe 테스트
- **기기 진단 화면**: SoC / GPU, 드라이버 버전, Vulkan 버전 · 확장 · 기능 (float atomic, fp16, int64, subgroup 크기·연산, 메모리 힙), ARCore 설치·지원 여부, Depth API 지원, 지원 CameraConfig 목록, RAM, 열 상태
- **GPU 마이크로벤치마크**: 메모리 대역폭, 100만 키 radix sort, atomic 방식별 처리량
- 진단 결과 JSON 공유, 피드백 이슈 템플릿

**사용자 테스트**
1. APK 설치 · 실행
2. 진단 화면 확인 → JSON 공유
3. 벤치마크 실행 (실행 전후 발열 상태 함께 기록)
4. (v0.1.1에서) 새 버전 덮어 설치가 되는지 확인

**완료 기준**: 사용자 기기에서 실행되고 진단 JSON을 받음 → 기기에 맞춘 설계 결정(atomic 전략, 학습 해상도, 가우시안 예산) 확정

### M1 — 스플랫 뷰어 `v0.2` · 1~2 사이클

**목표**: 렌더링 경로와 GPU 정렬을 검증하고, 이후 결과 확인용 도구를 확보한다.

**구현**: PLY / SPZ 로더, 파일 선택, 내장 합성 샘플, Vulkan 렌더러 + GPU 정렬, 터치 조작, HUD

**사용자 테스트**
1. 내장 샘플이 제대로 보이는지
2. 외부 PLY / SPZ 파일 로드 (다른 앱이나 공개 샘플에서 받은 파일)
3. 가우시안 수별 FPS 기록
4. 깜빡임 · 정렬 오류 · 색 이상이 있으면 스크린샷

**완료 기준**: 가우시안 30만 개에서 30 fps 이상, 외부 뷰어와 시각적으로 같은 결과

### M2 — 촬영 & 데이터셋 `v0.3` · 2~3 사이클

**목표**: 학습에 쓸 수 있는 품질의 이미지 + 포즈 + 깊이를 확보한다.

**구현**: ARCore 촬영 화면, 키프레임 선택, 촬영 가이드 UI, 스캔 목록, 저장 · 내보내기, **디버그 3D 뷰** (카메라 궤적 + depth 포인트 클라우드를 M1 뷰어로 표시), 재투영 검증 도구

**사용자 테스트**
1. 표준 장면 3종 촬영 (§8)
2. 디버그 뷰에서 궤적과 포인트 클라우드가 실제 장면과 맞는지 확인
3. 데이터셋 zip 공유 (선택: PC에서 nerfstudio / gsplat으로 학습 시도)
4. 촬영 중 끊김이나 경고가 어떻게 느껴지는지

**완료 기준**: 포인트 클라우드가 이중상 없이 정합됨. (선택) PC 학습 결과가 정상

### M3 — 온디바이스 학습 v1 `v0.4` · 3~5 사이클 ★ 핵심 난관

**목표**: 폰에서 처음부터 끝까지 학습되는 첫 결과.

**구현**: Vulkan 전처리 · 정렬 · forward · backward · Adam, L1 loss, depth 초기화, 고정 가우시안 수(densify 없음), SH 0차, 학습 화면(미리보기, 손실/PSNR 그래프, 단계별 시간), 결과 저장 → 뷰어. CI에서 CPU 레퍼런스 대비 그래디언트 수치 검증.

**사용자 테스트**
1. M2 스캔으로 학습 실행 → 완료까지 걸린 시간
2. 결과를 뷰어로 확인 (스크린샷 + 학습 이미지와 비교)
3. 진단 패키지 공유 (iteration 시간, 온도, 메모리)
4. 같은 스캔으로 2회 학습 → 재현성 확인

**완료 기준**: 표준 장면이 알아볼 수 있게 재구성되고, 크래시 없이 완료되며, hold-out PSNR이 측정됨

### M4 — 품질 향상 `v0.5` · 2~3 사이클

**구현**: D-SSIM loss, 예산 고정 densification (MCMC / Taming 비교), SH 1~2차, 해상도 · SH 점진 스케줄, opacity 정규화, 프리셋 & 고급 설정 UI, 결과 비교 화면

**사용자 테스트**: 프리셋별 시간 · PSNR · 화질 비교, 고급 설정으로 자유 실험

**완료 기준**: hold-out PSNR ≥ 22 dB, 학습 ≤ 5분

### M5 — 속도 최적화 `v0.6` · 2~4 사이클

**구현**: GPU 타임스탬프 프로파일링, sparse Adam, fp16 경로, 기기별 atomic 전략 자동 선택, 타일 크기 · workgroup 튜닝, 커널 fusion, ADPF 열 관리

**사용자 테스트**: 벤치마크 모드 + 표준 장면 학습 시간, 3회 연속 학습 시 발열과 성능 저하

**완료 기준**: 학습 ≤ 2~3분, 연속 실행 시 성능 저하 < 30%

### M6 — 견고성 `v0.7` · 2~3 사이클

**구현**: 학습 중 카메라 포즈 미세 보정, 프레임별 노출 · 화이트밸런스 보정, floater 제거, 크롭 박스 · 배경 제거(물체 모드), 추적 품질이 나쁜 구간 자동 제외, (선택) 촬영 중에 학습을 미리 시작

**사용자 테스트**: 어려운 장면 (조명 변화, 반사, 넓은 공간)

**완료 기준**: 표준 장면과 어려운 장면 1종 이상에서 눈에 띄는 개선

### M7 — 사용성 마무리 `v1.0` · 1~2 사이클

**구현**: 스캔 갤러리 (썸네일, 이름 변경, 삭제), SPZ / PLY 공유, 백그라운드 학습 (Foreground Service), 크래시 로그 수집, 설정 정리, (가능하면) 다른 기기 테스트

**완료 기준**: 촬영 → 학습 → 공유 전체 과정을 설명 없이 할 수 있음

### 확장 후보 (v1.0 이후)

촬영 중 실시간 학습 · 미리보기, LOD · 압축 학습, 2DGS · 메쉬 추출, ARCore 미지원 기기를 위한 온디바이스 SfM, 여러 기기 지원 매트릭스

---

## 8. 실험 · 평가 계획

- **표준 장면 3종** (버전마다 같은 장면을 다시 촬영하거나 같은 스캔을 재사용)
  - S1 물체: 책상 위 물체 1개 주위를 360°
  - S2 실내: 방 한쪽 구석 또는 책장
  - S3 야외: 작은 조형물이나 화단
- **지표**: hold-out PSNR / SSIM (키프레임 8장마다 1장을 평가용으로 분리), 학습 시간 (전체 · iteration당 · 단계별), 최대 메모리, 온도 변화, 가우시안 수, 파일 크기, 뷰어 FPS
- **비교 기준** (가능할 때): 같은 데이터셋으로 PC(gsplat 등)에서 학습한 결과, Scaniverse 등 기존 앱으로 같은 장면을 처리한 시간과 화질
- **기록**: `docs/experiments/`에 버전별 결과 표와 피드백 요약을 쌓아 성능 회귀 여부를 추적

---

## 9. 리스크와 대응

| 리스크 | 영향 | 대응 |
|---|---|---|
| GPU 계열별 드라이버 차이 (Adreno / Mali·Immortalis / Xclipse / PowerVR) | 커널 오동작, 성능 편차 | M0 진단으로 확인, 기능별 fallback 경로, lavapipe 정합성 테스트 |
| float atomic 미지원 또는 느림 | backward 성능 저하 | CAS 에뮬레이션 / atomic-free 경로 |
| ARCore 포즈 drift · 스케일 오차 | 흐릿함, 이중상 | 짧은 촬영 권장, 학습 중 포즈 보정(M6), 추적 품질 낮은 프레임 제외 |
| CPU 이미지 해상도 제한 (기본 640×480) | 디테일 한계 | 고해상도 CameraConfig, Shared Camera (후순위) |
| 자동 노출 · 화이트밸런스 변화 | 색 얼룩 | 프레임별 색 보정 (M6) |
| 발열 스로틀링 | 학습 시간 증가 | ADPF 연동, 동적 작업량 조정 |
| 메모리 부족 / GPU 워치독 | 앱 종료 | 예산 자동 결정, dispatch 분할, 체크포인트 재개 |
| 원격 디버깅의 어려움 | 수정 지연 | 진단 패키지, 단계별 계측, 사용자 데이터셋을 Linux 엔진으로 재현 |
| 목표 시간 (≤ 2분) 미달 | 핵심 목표 미달 | 단계별 목표 (5분 → 3분 → 2분), 초기화 품질에 투자, 해상도·예산 트레이드오프를 프리셋으로 제공 |
| 라이선스 | 배포 제약 | 원본 3DGS 코드 미사용, Apache-2.0 / MIT 계열만 사용 |

---

## 10. 결정이 필요한 사항 (사용자 확인)

1. **테스트 기기의 정확한 모델명** — 같은 제품명이라도 출시 지역에 따라 SoC가 다를 수 있다 (예: 갤럭시 일부 모델은 Snapdragon/Adreno 또는 Exynos/Xclipse). GPU 계열에 따라 최적화 경로가 달라진다.
2. **주 대상 장면** — 물체 중심 / 실내 공간 / 야외 중 우선순위
3. **피드백 채널** — GitHub Issue 또는 이 대화
4. **GitHub Actions 사용 가능 여부와 저장소 공개 여부** — private이면 폰에서 GitHub 로그인 후 다운로드
5. (가능하면) 비교용으로 Scaniverse 등 기존 앱을 쓸 수 있는지

---

## 11. 참고 자료

- PocketGS: On-Device Training of 3D Gaussian Splatting for High Perceptual Modeling — https://arxiv.org/abs/2601.17354
- Brush (3D reconstruction for all) — https://github.com/ArthurBrussee/brush
- gsplat — https://github.com/nerfstudio-project/gsplat
- 3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023) — https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- 3D Gaussian Splatting as Markov Chain Monte Carlo (3DGS-MCMC, NeurIPS 2024)
- Taming 3DGS: High-Quality Radiance Fields with Limited Resources (SIGGRAPH Asia 2024) — https://arxiv.org/abs/2406.15643
- FastGS: Training 3D Gaussian Splatting in 100 Seconds — https://arxiv.org/abs/2511.04283
- SPZ 포맷 (Niantic, MIT) — https://github.com/nianticlabs/spz
- Scaniverse 온디바이스 스플래팅 소개 — https://80.lv/articles/capture-photorealistic-3d-scenes-on-your-phone-with-gaussian-splatting
- ARCore 개발 문서 (CameraConfig, Depth / Raw Depth, Shared Camera, Recording & Playback) — https://developers.google.com/ar/develop
