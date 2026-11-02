# 뷰어 장소(3D) 출처

뷰어의 **장소**는 사진이 아니라 실제로 3D 촬영한 가우시안 스플랫 장면입니다. 모델과 장소의 가우시안을 함께 정렬해 그리므로, 모델이 가구·벽 사이에 서고 시점을 돌리면 앞뒤가 맞게 가려지고 보입니다.

| 앱의 이름 | 장면 | 만든 사람 | 라이선스 | 원본 |
|---|---|---|---|---|
| 아파트 거실, 아파트 안뜰 | SA3D_R&D_XP47 | Stephane Agullo (https://sa3d.fr) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | https://superspl.at/view?id=cdcec084 |
| 마을회관 잔디밭 | Knock Community Hall | scbenoit (https://superspl.at/user?id=scbenoit) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | https://superspl.at/scene/0ff2e6dc |

두 장면 모두 PlayCanvas 엔진 예제(github.com/playcanvas/engine, `examples/assets/splats`)에 SOG 형식으로 들어 있는 것을 받아 썼습니다. 만드신 분들께 감사드립니다.

## 바꾼 점 (CC BY 4.0의 "변경 사항 표시")
`tools/environments/make_environments.py`가
- 거의 투명한 가우시안을 빼고, 서 있을 자리 둘레로 잘라내고(아파트: 집과 안뜰, 마을회관: 건물과 앞뒤 마당), 마을회관은 193만 → 110만 개로 줄였습니다 (불투명도 × 넓이가 큰 것부터)
- 시점에 따른 색(SH 1차 이상)을 뺐습니다 (모델 둘레로 돌리기 때문)
- 앱용 압축 형식(.gsenv)으로 다시 담았습니다

뷰어는 모델 발밑의 바닥을 조금 어둡게 해 그림자를 넣습니다.
