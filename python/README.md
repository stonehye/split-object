# 인접 사물 분리
## 설치
```shell
conda create -n split-object python=3.8
conda activate split-object
pip install -r requirements.txt
```
## 실행
* 입력: rgb, mask, depthmap 이미지
* 출력: 사물 별 mask 이미지, label
```shell
# 이미지 한장 테스트
test.py -rgb ./2021-11-01_15=22=44=823%rgb.jpg \
        -m ./2021-11-01_15=22=44=823%final_mask.png \
        -d 2021-11-01_15=22=44=823%depth.png
```
```shell
usage: test.py [-h] -rgb RGB -m MASK -d DEPTH [-o OUTPUT_DIR] [-rs REGION_SIZE]

인접 사물 분리 (Segmentation)

optional arguments:
  -h, --help            show this help message and exit
  -rgb RGB, --rgb RGB   입력 컬러 이미지 (required)
  -m MASK, --mask MASK  입력 마스크 이미지 (required)
  -d DEPTH, --depth DEPTH
                        입력 깊이 이미지 (required)
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        결과 출력 디렉토리 경로 (default = ./)
  -rs REGION_SIZE, --region_size REGION_SIZE
                        영역 병합 시 최소 영역 임계값 (객체 크기 대비 비율, default = 0.05)

```