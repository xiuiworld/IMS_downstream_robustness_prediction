# 익명화 비디오의 Downstream Task Robustness Prediction

이 프로젝트는 **익명화(anonymization)가 적용된 비디오에서 downstream task 성능 저하를 빠르게 예측하는 surrogate framework**를 목표로 합니다.

최종적으로는 `YOLOv8 + DeepSORT` 같은 실제 downstream pipeline을 매번 전부 실행하지 않고도, 주어진 익명화 방식과 영상 정보를 바탕으로 detection / tracking 성능 저하를 예측하는 시스템을 만드는 것이 목적입니다.

현재 저장소는 **데이터 준비, target 생성, 학습 입력 생성, 세 모델 공통 학습/평가 파이프라인, single-task ablation, 결과 집계, 추론 속도 측정**까지 포함합니다. 모델 구조 튜닝은 계속 가능하지만, ParamMLP / VisualBaseline / FusionMultiTask를 같은 조건에서 비교하고 RQ2 ablation까지 실행하는 최종 실험 골격은 고정되어 있습니다. 최신 `models/`와 `training/` 코드는 modality guard, cache integrity validation, Video Swin Tiny 기반 visual encoder, single-task ablation, uncertainty-weighted Huber loss, AMP 학습/평가 옵션을 포함합니다.

## 프로젝트 목표

- localized anonymization이 downstream task에 미치는 영향을 정량화하기
- 실제 downstream 성능 저하량을 surrogate target으로 구축하기
- parameter-only, visual-only, fusion 기반 predictor를 비교하기
- multi-task learning, OOD generalization, inference efficiency를 같은 실험 프로토콜에서 비교하기

## 최종적으로 다루게 될 전체 흐름

```text
MOT17 raw data
  -> clip 생성
  -> anonymized clip 생성
  -> downstream evaluation
  -> target 생성
  -> surrogate 학습용 데이터 정리
  -> baseline / fusion model 학습
  -> 평가 및 ablation
  -> OOD generalization 검증
```

즉, 이 저장소는 단순한 전처리 저장소가 아니라, 장기적으로는 아래 단계를 모두 연결하는 연구 저장소를 목표로 합니다.

- dataset building
- target generation
- surrogate training
- evaluation and reporting

## 현재 구현된 범위

현재 바로 실행 가능한 단계는 다음과 같습니다.

- MOT17 시퀀스를 clip 단위로 생성
- GT box 내부 localized anonymization 적용
- `YOLOv8 + DeepSORT` 기반 target 생성
- target 분포 분석 및 요약 리포트 생성
- surrogate model용 CSV / `.pt` / cache 생성
- parameter-only baseline 학습
- visual-only baseline 학습
- fusion / multi-task surrogate model 학습
- 통합 evaluation 스크립트
- RQ2 single-task ablation
- seed별 결과 집계
- checkpoint 기반 inference FPS 측정

## 프로젝트 구조

```text
.
├─ configs/        # 경로, split, 파라미터 설정
├─ data/           # raw / interim / processed 데이터
├─ experiments/    # 학습 결과, checkpoint, 실험 산출물
├─ models/         # ParamMLP, VisualBaseline, FusionMultiTask, single-task ablation
├─ scripts/        # 데이터 준비, target 생성, 분석, 입력 생성 코드
├─ training/       # 공통 dataset/loss/metric/train/evaluate/aggregate/benchmark
├─ TrackEval/      # tracking metric 계산용
└─ README.md
```

`scripts/05_create_dataloaders.py`가 만든 cached artifact를 기준으로 `training/` 아래 공통 학습/평가 코드가 동작합니다.

## 모델 및 학습 코드 기준

`models/`는 입력 modality별 모델 정의만 담당하고, `training/model_factory.py`가 모델 이름과 modality를 고정 매핑합니다. canonical 실험에서는 `--modality`를 직접 덮어쓰지 않는 것이 원칙입니다.

```text
param_mlp                 -> param_only
visual_baseline           -> visual_only
fusion_multitask          -> fusion
visual_single_task_map    -> visual_only, delta_map only
visual_single_task_hota   -> visual_only, delta_hota only
fusion_single_task_map    -> fusion, delta_map only
fusion_single_task_hota   -> fusion, delta_hota only
```

모델 구현 기준은 다음과 같습니다.

- `ParamMLP`: anonymization type embedding과 severity를 입력으로 받아 `[delta_map, delta_hota]`를 예측합니다. OOD masking된 sample은 severity를 0으로 처리합니다.
- `VisualBaseline`: 기본 backbone은 `torchvision` Video Swin Tiny입니다. Kinetics-400 pretrained weight, Swin input normalization, early layer freezing을 실험 옵션으로 지원합니다. `simple3d`는 smoke/debug용 경량 fallback입니다.
- `FusionMultiTask`: parameter branch와 visual branch를 결합해 두 target을 동시에 예측합니다.
- `VisualSingleTask`, `FusionSingleTask`: RQ2 ablation용이며 한 번에 `delta_map` 또는 `delta_hota` 하나만 active metric으로 평가합니다.

`training/train.py`는 `train_config.json`, `model_summary.json`, `train_history.json`, `latest.pt`, `best.pt`를 저장합니다. validation monitor는 single-task의 경우 `active_rmse`, multi-task의 경우 `rmse_mean`을 우선 사용합니다.

## Colab 사용 기준

이 프로젝트는 **Google Colab + Google Drive** 환경을 기본 실행 환경으로 염두에 두고 있습니다.

권장 방식은 아래와 같습니다.

- 코드와 작은 설정 파일은 GitHub 또는 Drive에 저장
- 큰 데이터는 Colab local runtime(` /content `)으로 복사해서 사용
- 긴 실행으로 생기는 manifest / stats / snapshot은 Drive에 동기화

Drive 위의 많은 이미지 파일을 직접 읽는 방식은 느릴 수 있으므로, 가능하면 **Drive -> Colab local 복사 후 실행**하는 편이 안정적입니다.

### Colab 준비 예시

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
%cd /content
!rm -rf /content/downstream_robustness_prediction
!rsync -av /content/drive/MyDrive/downstream_robustness_prediction/ /content/downstream_robustness_prediction/
%cd /content/downstream_robustness_prediction
```

`data.zip`을 따로 관리한다면 아래처럼 압축 해제해서 사용할 수 있습니다.

```bash
%cd /content
!rm -rf /content/downstream_robustness_prediction
!mkdir -p /content/downstream_robustness_prediction
!rsync -av --exclude 'data.zip' /content/drive/MyDrive/downstream_robustness_prediction/ /content/downstream_robustness_prediction/
!unzip -q /content/drive/MyDrive/downstream_robustness_prediction/data.zip -d /content/downstream_robustness_prediction/
%cd /content/downstream_robustness_prediction
```

### 패키지 설치 예시

```bash
!pip install -q ultralytics deep-sort-realtime lap opencv-python pandas numpy pyyaml tqdm scipy matplotlib
!apt-get update -qq
!apt-get install -y ffmpeg
```

### Drive sync 설정 예시

```bash
%env TARGET_MANIFEST_SYNC_DIR=/content/drive/MyDrive/downstream_robustness_prediction/data/interim/manifests
```

이 설정을 사용하면 `target_manifest`, `target_stats`, `run_config_snapshot` 등 주요 산출물을 Drive와 동기화하면서 실행할 수 있습니다.

## 현재 기준 실행 순서

프로젝트 루트에서 아래 순서로 실행하면 됩니다.

```bash
python scripts/01_generate_original_clips.py
python scripts/02_apply_obfuscation.py
python scripts/03_generate_targets.py --yolo_device 0
python scripts/04_analyze_target_outputs.py
python scripts/05_create_dataloaders.py --img_size 224 --clip_len 30 --batch_size 8 --num_workers 2
```

처음에는 full run 대신 일부만 테스트하는 smoke run이 더 안전합니다.

```bash
python scripts/03_generate_targets.py \
  --semantic_run_id smoke \
  --max_original_clips 10 \
  --max_obf_clips 20 \
  --yolo_device 0
```

## 최종 실험 프로토콜

최종 비교 대상은 아래 세 모델입니다.

```text
1. param_mlp
2. visual_baseline
3. fusion_multitask
```

RQ2 ablation용 single-task 모델은 아래 네 개입니다.

```text
1. visual_single_task_map
2. visual_single_task_hota
3. fusion_single_task_map
4. fusion_single_task_hota
```

공정 비교를 위해 `training/dataset.py`, `training/train.py`, `training/evaluate.py`, `training/losses.py`, `training/metrics.py`는 공통으로 사용합니다. 학습 target은 z-score delta이고, 평가는 raw delta 단위의 RMSE / MAE / Spearman으로 수행합니다.

권장 canonical full sweep은 3 seeds이며, RQ1 세 모델과 RQ2 single-task ablation을 함께 실행합니다. `training.run_seed_sweep`의 기본 모델 목록은 RQ1+RQ2 전체이며, `--rq1_only`를 주면 RQ1 세 모델만 실행합니다.

```bash
python -m training.run_seed_sweep \
  --models param_mlp,visual_baseline,fusion_multitask \
  --include_ablations \
  --seeds 42,123,2026 \
  --loss uncertainty_huber \
  --visual_backbone swin_tiny \
  --device cuda
```

위 명령은 visual/fusion 모델에 대해 Video Swin Tiny Kinetics-400 pretrained 초기화와 early layer freezing을 기본으로 사용합니다. RQ2 single-task 결과는 `active_rmse`, `active_mae`, `active_spearman` 컬럼을 기준으로 해석합니다.

RQ2 single-task ablation만 별도로 돌릴 때는 아래처럼 실행합니다.

```bash
python -m training.run_seed_sweep \
  --models visual_single_task_map,visual_single_task_hota,fusion_single_task_map,fusion_single_task_hota \
  --seeds 42,123,2026 \
  --loss uncertainty_huber \
  --visual_backbone swin_tiny \
  --device cuda
```

```bash
python -m training.run_seed_sweep \
  --models param_mlp,visual_baseline,fusion_multitask \
  --seeds 42 \
  --loss huber \
  --visual_backbone simple3d \
  --device cpu \
  --epochs 1 \
  --train_extra --max_train_batches 2 --max_val_batches 1 --allow_empty_val
```

개별 모델을 직접 실행할 때의 canonical 명령은 다음과 같습니다.

```bash
python -m training.train --model param_mlp --loss uncertainty_huber --seed 42

python -m training.train \
  --model visual_baseline \
  --loss uncertainty_huber \
  --visual_backbone swin_tiny \
  --swin_pretrained \
  --freeze_early_layers \
  --seed 42

python -m training.train \
  --model fusion_multitask \
  --loss uncertainty_huber \
  --visual_backbone swin_tiny \
  --swin_pretrained \
  --freeze_early_layers \
  --seed 42
```

학습이 끝난 checkpoint는 같은 공통 평가 코드로 평가합니다.

```bash
python -m training.evaluate \
  --checkpoint experiments/<experiment_name>/best.pt \
  --split test
```

여러 seed 결과는 아래처럼 집계합니다. 기본 group은 `run_id`, model/task, loss, target mode, epochs, batch size, backbone, feature dim, Swin normalization/pretrain/freeze 설정을 포함하므로 서로 다른 설정이 평균에 섞이지 않습니다. RQ1/RQ2 pairwise 비교와 OOD-only 비교는 `rq_summary.csv`에 저장됩니다. 최종 표 생성에는 smoke run이 섞이지 않은 clean `experiments_root`를 사용하세요.

```bash
python -m training.aggregate_results --split test
```

추론 효율성은 checkpoint 기준으로 측정하고, YOLOv8+DeepSORT 기준 latency는 별도 측정값을 `--reference_time_ms`로 넣습니다. 제안서/보고서에 직접 사용할 속도 비교는 `--input_source raw_frames --batch_size 1` 기준으로 측정합니다. 기본값인 cached benchmark는 shared video tensor 로딩, device transfer, model forward를 포함하지만 raw frame decode/resize는 제외하므로, speedup ratio를 계산하려면 `--allow_cached_speedup`을 명시해야 합니다.

```bash
python -m training.benchmark_inference \
  --checkpoint experiments/<experiment_name>/best.pt \
  --split test \
  --input_source raw_frames \
  --batch_size 1 \
  --reference_time_ms <YOLO_DEEPSORT_MS>
```

`model_summary.json`에는 total/trainable parameter count가 저장됩니다. Fusion 계열은 visual trunk 외에 parameter branch와 fusion head가 추가되므로, 보고서에서는 total parameter와 visual trunk parameter를 구분해서 해석해야 합니다.

## 주요 산출물

현재 파이프라인을 돌리면 주로 아래 결과가 생성됩니다.

- `data/interim/manifests/clip_manifest.csv`
- `data/interim/manifests/target_manifest.csv`
- `data/interim/manifests/target_stats.csv`
- `data/interim/manifests/run_config_snapshot.json`
- `data/processed/` 아래 surrogate 학습용 입력 파일
- `experiments/<experiment_name>/train_config.json`
- `experiments/<experiment_name>/model_summary.json`
- `experiments/<experiment_name>/train_history.json`
- `experiments/<experiment_name>/best.pt`
- `experiments/<experiment_name>/latest.pt`
- `experiments/<experiment_name>/eval_<split>/<split>_metrics.json`
- `experiments/<experiment_name>/eval_<split>/<split>_predictions.csv`
- `experiments/aggregate_<split>/results_summary.csv`
- `experiments/aggregate_<split>/summary_by_model.csv`
- `experiments/aggregate_<split>/rq_summary.csv`
- `experiments/aggregate_<split>/benchmark_summary.csv`
- `experiments/aggregate_<split>/rq2_deployment_summary.csv`

장기적으로는 여기에 아래 산출물이 추가될 수 있습니다.

- 최종 보고서용 표/그림
- downstream reference latency 측정 결과
- OOD test 해석 리포트

## 로드맵

- Phase 1: dataset preparation, anonymization, target generation
- Phase 2: surrogate-ready dataloader and training input construction
- Phase 3: baseline / fusion model training
- Phase 4: evaluation, ablation, OOD generalization, final reporting

README 역시 이 로드맵에 맞춰 점진적으로 업데이트될 예정입니다.

## 참고

- 주요 경로와 파라미터는 `configs/config.yaml`에서 관리합니다.
- `TrackEval/` 디렉터리가 프로젝트 루트에 있어야 합니다.
- 생성 결과물(`data/`, `.csv`, `.json`, `.pt`, cache, checkpoint`)은 보통 GitHub에 올리지 않고 로컬이나 Drive에 두는 편이 적절합니다.
