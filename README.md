# on-device gsplat (Android)

스마트폰에서 촬영부터 3D Gaussian Splatting 학습, 확인까지 서버 없이 처리하는 안드로이드 앱 (Scaniverse 같은 빠른 온디바이스 3D 스캔이 목표, 촬영 대상은 사람).

- 개발 계획서: [docs/PLAN.md](docs/PLAN.md)
- 릴리즈 노트: [docs/releases/](docs/releases/)
- 현재 단계: **M0 — 기반 구축 & 기기 진단** (`v0.1.0`)

## 구조

| 경로 | 내용 |
|---|---|
| `app/` | Android 앱 (Kotlin, Jetpack Compose, ARCore). 네이티브 코드는 `app/src/main/cpp/jni_bridge.cpp` |
| `engine/` | C++20 / Vulkan 엔진. Android와 Linux에서 같은 코드로 빌드됨 |
| `engine/shaders/` | GLSL compute 셰이더 (빌드 시 SPIR-V로 컴파일되어 바이너리에 포함) |
| `engine/tests/` | GPU 커널 정합성 테스트 (CI에서는 Mesa lavapipe 소프트웨어 Vulkan 사용) |
| `.github/workflows/build.yml` | 엔진 테스트 + APK 빌드 + 새 버전이면 GitHub Release 생성 |

## 빌드

APK는 GitHub Actions가 빌드합니다. `version.properties`의 버전에 해당하는 Release가 없으면 `docs/releases/v<버전>.md`를 노트로 Release를 만들고 APK를 첨부합니다.

엔진만 Linux에서 빌드·테스트하려면 (Vulkan 드라이버, `glslc` 필요):

```sh
cmake -S engine -B build/engine -G Ninja -DCMAKE_BUILD_TYPE=Release -DGS_BUILD_TESTS=ON
cmake --build build/engine
ctest --test-dir build/engine --output-on-failure
./build/engine/gs_probe          # 기기 정보 + 벤치마크 JSON 출력
```
