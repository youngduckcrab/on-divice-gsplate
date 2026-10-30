# M0 결과 — Galaxy Z Fold6 (v0.1.0)

- 측정일: 2026-10-30, 빌드 `58213f9`
- 기기: SM-F956N, Android 16 (SDK 36), Snapdragon 8 Gen 3 for Galaxy (SM8650), Adreno 750
- 측정 조건: **충전 중, 배터리 13%, 열 상태 MODERATE (headroom 0.90)**, 커버 화면(968×2376) → 성능 수치는 보수적으로 봐야 함

## 원본 요약 (사용자 제공)

| 항목 | 값 |
|---|---|
| RAM | 총 10.9 GB, 측정 시 가용 1.8 GB |
| Vulkan | 1.3.128, 드라이버 빌드 2025-09-19 |
| Subgroup | 64 (가변 64~128), 연산: basic, vote, arithmetic, ballot, shuffle, shuffle_relative, quad (clustered 없음) |
| float atomic add | **미지원** |
| fp16 / int64 / 16bit storage | 지원 / 지원 / 지원 |
| 공유 메모리 | 32 KB |
| 최대 storage buffer (바인딩 1개) | **128 MB** |
| ARCore | 설치됨, Depth API·Raw depth 지원, ToF 없음 |
| CPU 이미지 해상도 | 640×480, 1280×720, **1920×1080** (모두 30 fps) |
| 메모리 대역폭 (복사) | 29.7 GB/s |
| FP32 | 853 GFLOPS |
| FP16 | 258 GFLOPS (FP32보다 낮음 → 벤치마크 문제로 추정, 아래 참고) |
| atomic u32 | 경합 높음 1.45 / 낮음 0.38 Gadds/s |
| atomic f32 CAS | 경합 높음 0.09 / 낮음 0.24 Gadds/s |
| radix sort 1M 키 (32-bit) | 20.9 ms, 50 Mkeys/s, **정확** |

## 해석과 설계 결정

1. **float atomic이 없다 → backward 그래디언트 누적 방식 변경.**
   CAS 에뮬레이션은 경합이 높을 때 u32 atomic보다 16배 느리다(0.09 vs 1.45 Gadds/s). 한 가우시안에 많은 픽셀이 기여하는 3DGS backward는 경합이 높은 패턴이므로 CAS를 그대로 쓰면 병목이 된다. M3에서 다음 세 가지를 구현해 비교한다.
   - (b) subgroup(64-wide) `subgroupAdd`로 먼저 합친 뒤 CAS — 경합을 최대 64배 줄임 (arithmetic 연산 지원 확인됨)
   - (c) atomic-free: (타일, 가우시안) 쌍별 부분합 기록 후 구간 합산
   - (d) **고정소수점 정수 atomic**: 그래디언트를 스케일링해 int32 `atomicAdd` — u32 atomic이 빠르므로 유력. 정밀도는 파라미터별 스케일로 관리. int64 atomic 지원 여부를 v0.2.0 진단에 추가해 확인
2. **storage buffer 바인딩 1개당 128 MB 제한.** 큰 배열은 128 MB 이하로 나눈다. 학습 이미지는 storage buffer 대신 이미지 배열(텍스처)에 둔다. SH 계수는 fp16으로 저장(가우시안 100만 개 SH 3차 = 96 MB).
3. **ARCore CPU 이미지 1920×1080 가능.** 촬영은 1080p로 저장하고 학습은 1/2(960×540)부터 시작한다. 깊이는 ToF 없이 depth-from-motion(raw depth + confidence)만 사용.
4. **정렬은 정확하지만 느리다 (50 Mkeys/s).** 드라이버 정확성은 확인됨. 학습에는 4~8배 빨라야 한다.
   - 뷰어(M1): 16-bit log-depth 키로 패스 수를 8→4로 줄임
   - 학습(M3): 깊이 선정렬 후 타일 ID만으로 안정 정렬(12~13 bit → 3~4 패스), subgroup ballot 기반 순위 계산으로 교체
5. **메모리.** 측정 시 가용 RAM이 1.8 GB였다. 학습 전체 메모리 목표를 1.5 GB 이하로 잡고, 시작 전 가용 메모리로 가우시안 예산을 정한다.
6. **발열.** 시작부터 MODERATE였다 (충전 중 + 배터리 부족). 이 상태의 수치는 최댓값이 아니다. v0.2.0에서 충전기 분리·냉각 상태로 재측정한다.
7. **FP16 < FP32 문제.** 벤치마크가 GLSL `fma()`를 썼는데, 이 함수는 정확한 fused 연산을 요구해서 Adreno의 fp16 경로에서 에뮬레이션됐을 가능성이 크다. 실제 셰이더 코드처럼 `a * b + c`로 바꿔 재측정한다 (v0.2.0).

## 성능 추정 갱신
대역폭 29.7 GB/s(보수적)는 계획서의 가정(데스크톱 대비 1/10~1/15)보다 낮다. 냉각 상태 재측정 후에도 30~40 GB/s 수준이라면 iteration당 시간 추정을 50~120 ms에서 80~150 ms로 올리고, 해상도 스케줄(960×540 이하에서 대부분의 iteration 수행)과 가우시안 예산(20만 개 내외)을 더 보수적으로 잡는다. 확정은 M3 실측 후.
