# 전체 프로젝트 파일구조

본 문서는 현재 구현된 저장소 구조를 기준으로 정리합니다. `/train` 관련 코드는 실제 디렉터리명 기준으로 `training/`에 있고, `/model` 관련 코드는 `models/`에 있습니다.

## 1. 전체 디렉토리 트리

```text
project_root/
├─ configs/
│  └─ config.yaml                       # 경로, split, target/model input 생성 설정
│
├─ data/
│  ├─ raw/                              # 원본 데이터 보관: 직접 수정 금지
│  │  ├─ MOT17/                         # MOT17-FRCNN 시퀀스 원본
│  │  └─ dataset_split.txt              # Train / Val / Test split 메타데이터
│  │
│  ├─ interim/
│  │  ├─ clips/
│  │  │  ├─ original/                   # 원본 클립
│  │  │  └─ obfuscated/                 # blur / pixelate / h264_local 등 적용 클립
│  │  ├─ manifests/
│  │  │  ├─ clip_manifest.csv
│  │  │  ├─ target_manifest.csv
│  │  │  ├─ target_stats.csv
│  │  │  └─ run_config_snapshot.json
│  │  └─ logs/
│  │
│  └─ processed/
│     ├─ evaluation/                    # YOLOv8 + DeepSORT 실행 결과와 metric
│     ├─ targets/
│     │  └─ surrogate_targets.csv       # 학습용 raw delta target
│     └─ model_inputs/
│        ├─ param_only/
│        │  ├─ train.pt
│        │  ├─ val.pt
│        │  └─ test.pt
│        ├─ visual_only/
│        │  ├─ train.pt
│        │  ├─ val.pt
│        │  └─ test.pt
│        ├─ fusion/
│        │  ├─ train.pt
│        │  ├─ val.pt
│        │  ├─ test.pt
│        │  └─ normalization_stats.json
│        ├─ shared_video_cache*_T*_S*_*/samples/
│        ├─ shared_sample_manifest.json
│        ├─ learning_schema.json
│        └─ cache_fingerprint.json
│
├─ models/
│  ├─ __init__.py                       # 공개 model class export
│  ├─ _validation.py                    # type_id 범위 검증
│  ├─ param_mlp.py                      # parameter-only baseline
│  ├─ visual_baseline.py                # Video Swin Tiny / Simple3D visual baseline
│  ├─ fusion_multitask.py               # parameter + visual multi-task fusion
│  └─ single_task.py                    # RQ2 single-task ablation 모델
│
├─ training/
│  ├─ __init__.py
│  ├─ dataset.py                        # cache 기반 Dataset/DataLoader와 integrity validation
│  ├─ model_factory.py                  # model name -> modality/model class 매핑
│  ├─ train.py                          # 공통 학습 entrypoint
│  ├─ evaluate.py                       # checkpoint 평가와 prediction CSV 저장
│  ├─ losses.py                         # Huber/MSE/MAE/weighted/uncertainty loss
│  ├─ metrics.py                        # RMSE/MAE/Spearman/OOD/zero-target metric
│  ├─ run_seed_sweep.py                 # RQ1/RQ2 seed sweep 실행
│  ├─ aggregate_results.py              # seed별 metric/benchmark 집계
│  ├─ benchmark_inference.py            # cached/raw frame latency/FPS 측정
│  └─ utils.py                          # seed, JSON, batch 이동, checkpoint helper
│
├─ scripts/
│  ├─ 01_generate_original_clips.py
│  ├─ 02_apply_obfuscation.py
│  ├─ 03_generate_targets.py
│  ├─ 04_analyze_target_outputs.py
│  └─ 05_create_dataloaders.py
│
├─ experiments/
│  ├─ <experiment_name>/
│  │  ├─ train_config.json
│  │  ├─ model_summary.json
│  │  ├─ train_history.json
│  │  ├─ latest.pt
│  │  ├─ best.pt
│  │  ├─ eval_<split>/
│  │  │  ├─ eval_config.json
│  │  │  ├─ <split>_metrics.json
│  │  │  └─ <split>_predictions.csv
│  │  └─ benchmark_<split>/
│  │     ├─ benchmark_metrics.json
│  │     └─ benchmark_rows.csv
│  └─ aggregate_<split>/
│     ├─ results_summary.csv
│     ├─ summary_by_model.csv
│     ├─ rq_summary.csv
│     ├─ benchmark_summary.csv
│     ├─ rq2_deployment_summary.csv
│     └─ aggregate_config.json
│
├─ TrackEval/                           # tracking metric 계산용 외부 코드
├─ README.md
├─ STRUCTURE.md
└─ TIMELINE.md
```

`run_id`를 지정한 실행은 `target_manifest_<run_id>.csv`, `surrogate_targets_<run_id>.csv`, `train_<run_id>.pt`처럼 suffix가 붙은 산출물을 사용합니다. `run_id`가 비어 있거나 `canonical`이면 suffix 없는 canonical artifact를 사용합니다.

## 2. 데이터 흐름

1. `data/raw/`: MOT17 원본 시퀀스와 split 정의를 읽습니다.
2. `scripts/01_generate_original_clips.py`: 원본 시퀀스를 30프레임 단위 clip으로 분할합니다.
3. `scripts/02_apply_obfuscation.py`: GT box 내부에 localized anonymization을 적용합니다.
4. `scripts/03_generate_targets.py`: YOLOv8 + DeepSORT를 실행하고 원본 대비 `delta_map`, `delta_hota` target을 생성합니다.
5. `scripts/04_analyze_target_outputs.py`: target 분포, zero/nonzero target, censoring 통계를 확인합니다.
6. `scripts/05_create_dataloaders.py`: `param_only`, `visual_only`, `fusion` split index와 shared video cache를 생성합니다.
7. `training/train.py`: cached artifact를 읽어 모델을 학습하고 checkpoint를 저장합니다.
8. `training/evaluate.py`: checkpoint를 평가해 metric JSON과 prediction CSV를 저장합니다.
9. `training/aggregate_results.py`: seed별 평가/benchmark 결과를 RQ별 summary로 집계합니다.

## 3. 모델 구조

`training/model_factory.py`는 모델 이름과 modality를 다음처럼 고정합니다.

```text
param_mlp                 -> param_only
visual_baseline           -> visual_only
fusion_multitask          -> fusion
visual_single_task_map    -> visual_only
visual_single_task_hota   -> visual_only
fusion_single_task_map    -> fusion
fusion_single_task_hota   -> fusion
```

- `ParamMLP`: anonymization type embedding과 severity를 concat하여 `delta_map`, `delta_hota`를 예측합니다.
- `VisualBaseline`: `swin_tiny` 또는 `simple3d` video encoder를 사용합니다. `swin_tiny`는 Kinetics-400 pretrained weight, input normalization, early layer freezing을 옵션으로 지원합니다.
- `FusionMultiTask`: parameter encoder와 visual encoder feature를 결합한 multi-task predictor입니다.
- `VisualSingleTask` / `FusionSingleTask`: `delta_map` 또는 `delta_hota` 하나만 active output으로 학습/평가하는 RQ2 ablation 모델입니다.

## 4. 학습 및 평가 구조

`training/train.py`는 모델별 코드 중복 없이 공통 loop를 사용합니다.

- `--model`에 따라 `model_factory.py`에서 모델과 modality를 결정합니다.
- `--modality`를 직접 넘길 수는 있지만, 모델의 canonical modality와 다르면 실행을 중단합니다.
- 학습 target은 기본적으로 z-score delta이며, metric은 raw delta로 denormalize해서 계산합니다.
- 지원 loss는 `huber`, `weighted_huber`, `mse`, `mae`, `uncertainty_huber`입니다.
- visual/fusion 모델은 `--visual_backbone swin_tiny`, `--swin_pretrained`, `--freeze_early_layers`, `--swin_input_norm` 옵션을 가집니다.
- CUDA 환경에서는 `--amp`로 mixed precision 학습/평가/benchmark를 사용할 수 있습니다.
- cache integrity validation은 기본으로 켜져 있고, smoke/debug 실행에서만 `--skip_integrity`로 끌 수 있습니다.

`training/evaluate.py`는 checkpoint 안의 model args, target mode, normalization stats를 우선 사용합니다. checkpoint와 다른 `run_id`로 평가하려면 `--allow_run_id_override`를 명시해야 합니다.

## 5. Google Colab 연동 및 동기화 전략

대량의 개별 이미지 파일을 Drive에서 직접 읽으면 I/O 병목이 커집니다. 권장 방식은 코드와 작은 manifest/config는 Drive에 두고, `data.zip` 또는 raw/interim/processed 데이터는 Colab local runtime으로 복사한 뒤 실행하는 것입니다.

```python
from google.colab import drive
drive.mount('/content/drive')

PROJECT_DRIVE = "/content/drive/MyDrive/downstream_robustness_prediction"
DATA_ZIP_DRIVE = "/content/drive/MyDrive/downstream_robustness_prediction/data.zip"
```

```bash
rm -rf /content/downstream_robustness_prediction
rsync -av --exclude 'data.zip' "$PROJECT_DRIVE/" /content/downstream_robustness_prediction/
unzip -q "$DATA_ZIP_DRIVE" -d /content/downstream_robustness_prediction/
cd /content/downstream_robustness_prediction
```

긴 target 생성 실행에서는 주요 manifest를 Drive로 동기화하기 위해 아래 환경 변수를 사용할 수 있습니다.

```bash
export TARGET_MANIFEST_SYNC_DIR=/content/drive/MyDrive/downstream_robustness_prediction/data/interim/manifests
```
