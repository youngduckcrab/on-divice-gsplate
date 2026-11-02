# 뷰어 장소(3D) 출처

뷰어의 **장소**는 사진이 아니라 3D 가우시안 스플랫 장면입니다. 모델과 장소의 가우시안을 함께 정렬해 그리므로, 모델이 가구·벽 사이에 서고 시점을 돌리면 앞뒤가 맞게 가려지고 보입니다. 장소는 두 종류입니다.

- **3D 촬영한 장면** (아래 표): 남이 실제 장소를 찍어 공개한 것
- **직접 만든 장면** (CG 거실, 사진 스튜디오): 이 저장소의 스크립트로 Blender에서 짓고 조명을 계산해 가우시안으로 바꾼 것

## 3D 촬영한 장면

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

## 직접 만든 장면 (Blender)
`tools/environments/blender/`의 스크립트가 Blender(파이썬 모듈 `bpy`)로 방을 짓고, Cycles로 빛(해·하늘·조명, 직접광과 반사광)을 계산해 표면마다 굽고(bake), 표면을 2~3 cm 간격의 납작한 가우시안으로 바꿉니다 (`place_kit.py`). 빛은 미리 계산해 색에 넣었으므로 폰에서는 다른 장소와 똑같이 그리기만 합니다.

| 앱의 이름 | 스크립트 | 들어간 것 |
|---|---|---|
| CG 거실 | `living_room.py` | 큰 창으로 해가 드는 거실: 소파, 의자 둘, 푸프, 러그, 커피 테이블과 꽃병, 책장, 사이드보드와 그림, 스탠드, 펜던트 조명, 화분, 창밖 잔디·생울타리·나무 |
| 사진 스튜디오 | `studio.py` | 종이 배경지(스윕), 옥타박스(주광)·스트립(역광)·배경 조명, V-플랫 반사판, 삼각대 위 카메라, 노트북 카트, 분장 거울과 전구, 옷걸이, 소파, 액자 |

벽·바닥·가구 대부분은 스크립트에서 직접 모델링했습니다. 일부 가구는 Khronos glTF 샘플 에셋(github.com/KhronosGroup/glTF-Sample-Assets)을 썼습니다.

| 에셋 | 쓰인 곳 | 저작권자 / 만든 사람 | 라이선스 |
|---|---|---|---|
| GlamVelvetSofa | CG 거실 소파 | Wayfair, LLC / Eric Chadwick | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| ChairDamaskPurplegold | CG 거실 의자 | Wayfair / Eric Chadwick | CC BY 4.0 |
| SpecularSilkPouf | CG 거실 푸프 | Wayfair, LLC / Eric Chadwick | CC BY 4.0 |
| SheenChair | CG 거실 의자 | Wayfair, LLC / Eric Chadwick | CC0 |
| GlassVaseFlowers | CG 거실 꽃병의 꽃 | Eric Chadwick, Rico Cilliers | CC0 |
| DiffuseTransmissionPlant | CG 거실 화분 | Darmstadt Graphics Group GmbH / Eric Chadwick (CC BY 4.0), Rico Cilliers (CC0) | CC BY 4.0 |
| SheenWoodLeatherSofa | 스튜디오 소파 | Darmstadt Graphics Group GmbH / Eric Chadwick (CC BY 4.0), Fran Calvente (CC0) | CC BY 4.0 |

바꾼 점: 에셋을 방에 배치하고 크기·방향을 맞췄으며, 꽃병의 유리는 뺐습니다. 표면을 잘게 나눠 조명을 구운 뒤 가우시안으로 바꿨습니다 (재질의 반사광은 빼고 확산광만).

다시 만들기 (PC, 파이썬 3.11):
```
pip install bpy==5.0.1 numpy
# 위 에셋 .glb 파일을 한 폴더에 받아 두고
python tools/environments/blender/living_room.py <에셋 폴더> <출력 폴더> preview   # 미리보기 3장
python tools/environments/blender/living_room.py <에셋 폴더> <출력 폴더> bake      # living_room.ply
gs_place <출력 폴더>/living_room.ply app/src/main/assets/environments/living_room.gsenv
```
(4코어 CPU에서 굽기에 거실 약 30분, 스튜디오 약 15분)

