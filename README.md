# Sign Avatar – Gloss → Motion Baseline

AI-Hub 한국어 수어 데이터셋을 사용해서  
**수어 gloss 시퀀스 → 모션 시퀀스 (T, J, 3)** 를 직접 생성하고,  
이를 아바타 엔진에서 사용할 수 있는 `"tracks"` JSON 형태로 내보내는 베이스라인 프로젝트입니다.

- 입력: gloss 시퀀스 (예: `["왼쪽"]`)
- 출력: 정규화된 모션 시퀀스 `(T, J, 3)`  
  → bone 단위 `"tracks"` JSON으로 변환해 아바타에 적용

---

## Features

- AI-Hub 수어 데이터 (OpenPose keypoints, morpheme 라벨) 기반
- gloss 시퀀스 → 모션 시퀀스 **Transformer** 모델
- 3D keypoints (x, y, z) 정규화 및 리샘플링
- 문장 단위 / segment(gloss) 단위 둘 다 학습 가능
- 학습된 체크포인트로:
  - `(T, J, 3)` 모션 `.npy` 생성
  - 아바타 엔진용 `"tracks"` JSON 생성

---

## Project Structure

```bash
sign-avatar/
├─ README.md
├─ requirements.txt        # (선택) Python 의존성
├─ configs/
│  └─ train_baseline.yaml  # 학습 설정
├─ data/
│  ├─ raw/
│  │  ├─ keypoints/        # AI-Hub OpenPose *_keypoints.json
│  │  └─ morpheme/         # *_morpheme.json (형태소/비수지)
│  ├─ processed/
│  │  ├─ meta_train.json           # 문장 단위 또는 segment 단위 메타
│  │  ├─ meta_val.json
│  │  ├─ meta_train_segments.json  # (선택) segment 단위 메타
│  │  └─ gloss_vocab.json          # gloss → id 사전
│  └─ outputs/
│     ├─ checkpoints/      # 모델 체크포인트
│     └─ samples/          # 생성 모션/트랙스 샘플
├─ scripts/
│  ├─ build_metadata.py            # 문장 단위 메타 생성
│  ├─ build_metadata_segments.py   # segment(gloss) 단위 메타 생성
│  └─ build_vocab.py               # gloss_vocab.json 생성
└─ src/
   ├─ data/
   │  ├─ keypoints_io.py          # *_keypoints.json 로더
   │  ├─ keypoints_processing.py  # normalize/resample (T,J,3)
   │  ├─ dataset.py               # Gloss2MotionDataset (PyTorch)
   │  └─ vocab.py                 # GlossVocab (encode/디코딩)
   ├─ models/
   │  └─ transformer_gloss2motion.py  # Gloss→Motion Transformer
   ├─ training/
   │  ├─ losses.py                # MSE 기반 motion loss
   │  ├─ trainer.py               # train_one_epoch / evaluate
   │  └─ train_baseline.py        # 학습 main 스크립트
   ├─ inference/
   │  ├─ generate_motion.py       # 체크포인트 → (T,J,3) 생성
   │  └─ export_tracks.py         # (T,J,3) → tracks JSON
   └─ utils/
      ├─ config.py                # YAML config 로딩
      ├─ logging.py               # 실험 폴더/로그 경로 생성
      └─ seed.py                  # 랜덤 시드 고정
```
---

## 1. Setup
### 1.1. Environment
git clone <this-repo-url> sign-avatar
cd sign-avatar

### (선택) 가상환경
```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```


###  의존성 설치
```
pip install -r requirements.txt
```


### 의존성 예시:
```
torch
numpy
pyyaml
```


(필요에 따라 tqdm, matplotlib 등 추가)

## 2. Data Preparation
### 2.1. Keypoints (OpenPose) 위치

AI-Hub [라벨]01_crowd_keypoint.zip 등을 풀면 대략:
```
.../1.Training/01/NIA_SL_WORD0001_REAL01_D/
  ├─ NIA_SL_WORD0001_REAL01_D_000000000000_keypoints.json
  ├─ NIA_SL_WORD0001_REAL01_D_000000000001_keypoints.json
  └─ ...
```

이런 식의 *_keypoints.json 들이 생성됩니다.

이 전체를 data/raw/keypoints/ 아래로 두면 됩니다. 예:
```
mkdir -p data/raw/keypoints
```

 예시
```
cp -r "/path/to/수어 영상/1.Training" data/raw/keypoints/
```

중요: NIA_SL_XXX_YYYY_D_000000000000_keypoints.json 같은 파일이
data/raw/keypoints/** 경로 어딘가에 있으면 됩니다.
build_metadata.py가 재귀적으로 찾아서 prefix를 잡습니다.

### 2.2. Morpheme (형태소 라벨) 위치

AI-Hub [라벨]01_crowd_morpheme.zip 등을 풀면 (예시):
```
.../2.Validation/라벨링데이터_231204_add/01_real_sen_morpheme/morpheme/17/
  └─ NIA_SL_SEN0001_REAL17_D_morpheme.json
```

이런 *_morpheme.json 파일들을 data/raw/morpheme/ 아래에 둡니다.

예:
```
mkdir -p data/raw/morpheme
```

### 예시 (Validation)
cp -r "/path/to/004.수어영상/2.Validation/라벨링데이터_231204_add/01_real_sen_morpheme/morpheme" \
      data/raw/morpheme/val

### (선택) Training morpheme 있을 경우
cp -r "/path/to/train_morpheme_root" data/raw/morpheme/train

## 3. Build Metadata (morpheme ↔ keypoints 매핑)
### 3.1. 문장 단위 메타 (meta_train.json, meta_val.json)

build_metadata.py는:

morpheme JSON에서

원본 비디오 이름 (예: NIA_SL_SEN0001_REAL17_D.mp4)

gloss 시퀀스 리스트 (["왼쪽", ...])

각 gloss의 시작/끝 시간

keypoints 쪽에서

해당 영상 id (NIA_SL_SEN0001_REAL17_D)와 매칭되는
..._000000000000_keypoints.json prefix

를 찾아 문장 단위 샘플 리스트를 생성합니다.

Validation 메타 생성 예시
```
python scripts/build_metadata.py \
  --morpheme-root data/raw/morpheme/val \
  --keypoints-root data/raw/keypoints \
  --output data/processed/meta_val.json \
  --fps 30
```

Training 메타 생성 예시
```
python scripts/build_metadata.py \
  --morpheme-root data/raw/morpheme/train \
  --keypoints-root data/raw/keypoints \
  --output data/processed/meta_train.json \
  --fps 30
```

### 3.2. Segment(gloss) 단위 메타 (meta_train_segments.json) (선택)

각 문장을 segment (gloss) 단위로 잘라서,
각 segment를 독립적인 학습 샘플로 사용하는 메타를 만들 수 있습니다.

```
python scripts/build_metadata_segments.py \
  --morpheme-root data/raw/morpheme/train \
  --keypoints-root data/raw/keypoints \
  --output data/processed/meta_train_segments.json \
  --fps 30
```


이렇게 생성된 메타를 쓰면:

gloss_seq는 보통 [단일_gloss]

segment.start_frame, segment.end_frame 범위만 잘라서 모션 클립을 학습

## 4. Build Gloss Vocab

gloss_vocab.json은 gloss 문자열을 integer id로 매핑하는 사전입니다.
meta 파일들에 있는 gloss_seq를 모두 모아 생성합니다.
```
python scripts/build_vocab.py \
  --meta data/processed/meta_train.json data/processed/meta_val.json \
  --output data/processed/gloss_vocab.json \
  --min-freq 1
```

--min-freq 1 : 1번 이상 등장한 gloss를 모두 포함
(데이터가 많으면 2 이상으로 조정해도 됩니다.)

## 5. Config (configs/train_baseline.yaml)

학습에 사용할 경로/하이퍼파라미터를 설정합니다.

예시:
```
experiment_name: "gloss2motion_baseline"

data:
  meta_train: "data/processed/meta_train.json"          # or meta_train_segments.json
  meta_val:   "data/processed/meta_val.json"
  gloss_vocab: "data/processed/gloss_vocab.json"
  target_T: 60        # 모션 시퀀스 프레임 수 (리샘플링 후)
  max_gloss_len: 30
  batch_size: 8
  num_workers: 4

model:
  d_model: 512
  nhead: 8
  num_encoder_layers: 4
  num_decoder_layers: 4
  dim_feedforward: 1024
  num_joints: 49      # keypoints_processing.py 에서 사용하는 J와 일치해야 함
  dropout: 0.1

train:
  epochs: 50
  lr: 1e-4
  weight_decay: 1e-4
  device: "cuda"

logging:
  output_dir: "data/outputs"
  save_every: 1
```

num_joints 는 keypoints_processing.py에서 실제로 만들어지는 joint 개수와 반드시 일치해야 합니다.
예) 상체 7개 + 오른손 21개 + 왼손 21개 = 49

## 6. Training
python -m src.training.train_baseline --config configs/train_baseline.yaml


체크포인트는 자동으로
data/outputs/<experiment_name>_YYYYMMDD_HHMMSS/checkpoints/ 에 저장됩니다.

검증 loss 기준으로 가장 좋은 모델은 best.pt 로 별도 저장됩니다.

## 7. Inference – Gloss → Motion (T, J, 3)
### 7.1. 모션 시퀀스 .npy 생성

학습된 체크포인트로 gloss 시퀀스를 넣어 (T, J, 3) 모션을 생성합니다.
```
python -m src.inference.generate_motion \
  --checkpoint data/outputs/gloss2motion_baseline_YYYYMMDD_HHMMSS/checkpoints/best.pt \
  --config configs/train_baseline.yaml \
  --vocab data/processed/gloss_vocab.json \
  --gloss "왼쪽" \
  --output data/outputs/samples/left_motion.npy \
  --device cuda
```

--gloss는 공백으로 구분된 gloss 시퀀스 문자열입니다.
예: "왼쪽 어디 있다_의문"

결과 .npy 파일은 np.ndarray 형식의 (T, J, 3) 배열입니다.

## 8. Inference – Motion → Tracks JSON (아바타 엔진용)
### 8.1. (T, J, 3) → "tracks" JSON
python -m src.inference.export_tracks \
  --motion-file data/outputs/samples/left_motion.npy \
  --output data/outputs/samples/left_tracks.json \
  --name "왼쪽" \
  --fps 30


left_tracks.json 예시 구조:
```
{
  "name": "왼쪽",
  "duration": 2.0,
  "tracks": [
    {
      "bone": "rightShoulder",
      "type": "vector",
      "times": [0.0, 0.0333, 0.0666, ...],
      "values": [
        0.0, 0.0, 0.0,
        0.01, -0.02, 0.00,
        ...
      ]
    },
    ...
  ]
}
```

BONE_NAMES 리스트 (export_tracks.py 안)는
아바타 엔진에서 사용하는 bone 이름/순서에 맞게 수정하면 됩니다.

이 JSON을 엔진에서 읽어서
bone 별로 times/values(x,y,z)에 맞춰 애니메이션을 재생하면 됩니다.

## 9. Segment-based Training (선택)

문장 전체가 아니라 각 segment(gloss) 단위로 학습하고 싶다면:

segment 메타 생성
```
python scripts/build_metadata_segments.py \
  --morpheme-root data/raw/morpheme/train \
  --keypoints-root data/raw/keypoints \
  --output data/processed/meta_train_segments.json \
  --fps 30
```

config에서 meta_train 을 segment 메타로 교체
```
data:
  meta_train: "data/processed/meta_train_segments.json"
  meta_val:   "data/processed/meta_val.json"
  ...
```

동일하게 train_baseline.py 실행

이렇게 하면:

각 segment (예: "왼쪽")만 잘라낸 모션 클립이 학습되므로
gloss 단위 모션 패턴을 더 명확하게 학습할 수 있습니다.

## 10. Notes / TODO

현재 모델은 단순한 non-autoregressive Transformer 베이스라인입니다.

gloss 시퀀스를 인코딩하고,
고정 길이 T 타임스텝에 대해 전체 모션을 한 번에 예측합니다.

- 향후 확장 아이디어:

1. Diffusion 기반 text-to-motion 모델로 교체 
2. 문장 단위에서 segment 모션들을 자연스럽게 이어 붙이는 coarticulation 모듈
3. joint 선택/가중치 튜닝, smoothness regularization 등
