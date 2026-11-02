# 뷰어 배경 사진 출처

뷰어의 사진 배경(`app/src/main/assets/backdrops/`)은 모두 [Poly Haven](https://polyhaven.com)의 HDRI이고, **CC0(퍼블릭 도메인)** 입니다. 만든 분들께 감사드립니다.

`tools/backdrops/make_backdrops.py`가 HDR 원본을 받아 화면용 JPEG(노출·톤매핑)과 `manifest.json`(배치, 빛의 SH 계수, 주광 방향·비중·색)을 만듭니다. 원본은 Poly Haven을 그대로 옮겨 둔 공개 GitHub 저장소(gkjohnson/3d-demo-data, google/filament, playcanvas/engine, mrdoob/three.js)에서 받습니다.

| 이름 | 분류 | Poly Haven 원본 | 앱 안의 크기 |
|---|---|---|---|
| 카페 | 실내 | [comfy_cafe](https://polyhaven.com/a/comfy_cafe) | 2048×1024 |
| 밝은 스튜디오 | 실내 | [brown_photostudio_02](https://polyhaven.com/a/brown_photostudio_02) | 2048×1024 |
| 빈티지 방 | 실내 | [photo_studio_loft_hall](https://polyhaven.com/a/photo_studio_loft_hall) | 2048×1024 |
| 궁전 홀 | 실내 | [vestibule](https://polyhaven.com/a/vestibule) | 2048×1024 |
| 박물관 | 실내 | [hall_of_mammals](https://polyhaven.com/a/hall_of_mammals) | 2048×1024 |
| 크리스마스 | 실내 | [christmas_photo_studio_04](https://polyhaven.com/a/christmas_photo_studio_04) | 2048×1024 |
| 노을 바다 | 야외 | [the_sky_is_on_fire](https://polyhaven.com/a/the_sky_is_on_fire) | 2048×1024 |
| 해돋이 들판 | 야외 | [spruit_sunrise](https://polyhaven.com/a/spruit_sunrise) | 4096×2048 |
| 정원 | 야외 | [chinese_garden](https://polyhaven.com/a/chinese_garden) | 2048×1024 |
| 베네치아 골목 | 야외 | [venetian_crossroads](https://polyhaven.com/a/venetian_crossroads) | 2048×1024 |
| 런던 시장 | 야외 | [leadenhall_market](https://polyhaven.com/a/leadenhall_market) | 2048×1024 |
| 밤 광장 | 야외 | [st_peters_square_night](https://polyhaven.com/a/st_peters_square_night) | 4096×2048 |
| 숲길 | 야외 | [rainforest_trail](https://polyhaven.com/a/rainforest_trail) | 2048×1024 |
| 시골길 | 야외 | [flower_road](https://polyhaven.com/a/flower_road) | 2048×1024 |

해돋이 들판은 화면용 그림을 three.js에 있는 4K 그림(같은 HDRI를 톤매핑한 것)에서, 빛은 1K HDR에서 계산합니다.
