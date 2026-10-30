# 예제 데이터셋 v1 — 가상 촬영

카메라와 촬영할 사람이 없어도 학습 파이프라인을 개발·검증할 수 있도록 만든 **가상 촬영 데이터셋**입니다.
폰을 세로로 들고 피사체 주위를 두 바퀴(높이를 바꿔서) 도는 상황을 흉내 냈고, 실제 촬영의 불완전함(걷는 속도 변화, 흔들림, 살짝 기울어짐)도 넣었습니다.
렌더링으로 만들었기 때문에 **카메라 위치가 정확**합니다. 학습이 잘못되면 데이터가 아니라 학습 코드 문제라는 뜻이라 개발용으로 유리합니다.

| 파일 | 피사체 | 용도 | 출처 / 라이선스 |
|---|---|---|---|
| `gsplat-capture-head.zip` | 실제 사람 머리 스캔(흉상), 받침대 위, 실내 | 얼굴 품질, 인물 마스크 학습 (P1 대용) | Lee Perry-Smith head scan, Infinite-Realities, CC BY 3.0 (three.js 예제) |
| `gsplat-capture-body.zip` | 전신 캐릭터, T자 자세, 실내 | 전신 촬영 (P2 대용) | Adobe Mixamo "Michelle" (three.js 예제) — 내부 테스트 전용, 재배포 금지 |
| `gsplat-capture-lizard.zip` | 실제 Scaniverse 스캔(공원의 도마뱀 조각상) | 사실적인 장면 전체 학습 (P0 대용) | Niantic hornedlizard.spz, MIT |
| `gsplat-capture-*.mp4` | 위 데이터셋의 미리보기 영상 | 눈으로 확인용 | — |

## 구성 (zip 안)

- `images/frame_00000.jpg` … 180장, 720×1280 세로, 핀홀 카메라(세로 화각 65°, 폰 메인 카메라와 비슷)
- `transforms.json` — nerfstudio 형식. `transform_matrix`는 카메라→월드, OpenGL 관례(x 오른쪽, y 위, 카메라는 −z 방향을 봄). 월드는 Y가 위, 단위 미터 (ARCore와 같음)
- `masks/` — 인물(피사체) 마스크, 8-bit (head/body만)
- `depth/`, `depth_confidence/` — **ARCore와 비슷하게 만든 깊이**: 160×90(세로), 거리 제곱에 비례하는 노이즈, 구멍, 경계 번짐. 16-bit 밀리미터, 0 = 없음 (head/body만)
- `depth_gt/` — 정확한 깊이, 1/4 해상도, 16-bit 밀리미터 (평가용)
- `points3d.ply` — 색이 있는 점 구름 6만 개, 약 3 mm 노이즈 (lizard만, SfM 결과 대용). 절반은 조각상, 절반은 주변 공원(멀수록 많이)에서 뽑음 — 배경에 점이 없으면 배경을 학습할 수 없음. `transforms.json`의 `ply_file_path`가 가리킴
- `preview.mp4`, `README.txt`

## 다시 만들기

`tools/datasets/`의 스크립트로 같은 데이터를 다시 만들 수 있습니다 (카메라 경로는 시드 고정, 모델은 버전 고정). GitHub Actions의 `datasets` 워크플로가 이 릴리즈를 갱신합니다.
