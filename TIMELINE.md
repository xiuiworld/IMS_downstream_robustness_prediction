# Project Timeline

본 문서는 프로젝트 제안서의 `Project Timeline`과 이를 바탕으로 요약된 `Phase`별 목표를 통합하여 재구성한 전체 진행 계획입니다.

## 1. 개요

- **목표**: 익명화된 비디오의 다운스트림 태스크(YOLOv8+DeepSORT) 성능 저하를 예측하는 프레임워크 개발
- **총 기간**: 12주
- **단계**: 총 3개의 Phase로 구성 (데이터 준비 → 모델 학습 → 평가 및 마무리)
- **현재 구현 기준**: Phase 1의 데이터/target 생성, Phase 2의 공통 학습 파이프라인과 모델 7종, Phase 3의 평가/집계/추론 벤치마크 코드가 저장소에 반영되어 있습니다.

## 2. Phase 1: 데이터 준비 및 전처리 (1~4주차)

이 단계의 핵심 목표는 "학습할 모델을 만드는 것"이 아니라, 모델 학습에 필요한 **"정답지(Ground Truth) 데이터셋을 구축하는 것"**입니다.

- **Week 1-2**:
    - MOT17 데이터셋 준비 및 Train/Val/Test 분할 확정.
    - 원본 시퀀스를 30프레임 단위의 클립으로 분할(`01_generate_original_clips.py`).
    - GT 박스 추출 등 베이스라인 파이프라인 코딩.
- **Week 3-4**:
    - 원본 클립의 보행자 GT 박스 내부에 국부적 난독화(Blur, Pixelate, H264_local) 적용.
    (`02_apply_obfuscation.py`).
    - YOLOv8+DeepSORT를 실행하여 원본 대비 실제 성능 저하량(Delta mAP, Delta HOTA) 산출.
    (`03_generate_targets.py`).
    - 활성 궤적 3개 미만 클립 제외 등 데이터 필터링(Censoring) 및 통계 분석.
    (`04_analyze_target_outputs.py`).
    - surrogate 학습용 `param_only`, `visual_only`, `fusion` cache와 split index 생성.
    (`05_create_dataloaders.py`).
- **우선순위 대체 계획 (Priority Fallback)**:
    - (참고) 데이터셋 생성 과정에서 병목이 발생할 경우, 모델 구조의 절제 연구(Ablation)를 우선시하기 위해 파라미터 그리드를 축소하여 실험을 진행합니다.

## 3. Phase 2: 모델 학습 및 실험 (5~8주차)

이 단계에서는 Phase 1에서 구축된 정답지 데이터를 바탕으로 **대체 예측 모델(Surrogate Model)을 학습시키고 구조를 검증**합니다.

- **Week 5-6**:
    - **Baseline 1 (ParamMLP)** 학습: 난독화 type embedding과 severity만을 입력으로 받아 `delta_map`, `delta_hota`를 예측하는 모델.
    (`models/param_mlp.py`, `training.train --model param_mlp`)
    - **Baseline 2 (VisualBaseline)** 학습: Video Swin Tiny 또는 smoke/debug용 Simple3D backbone으로 시각적 비디오 특징만을 입력받는 모델.
    (`models/visual_baseline.py`, `training.train --model visual_baseline`)
    - 공통 학습 loop, modality 검증, cache integrity validation, z-score target 학습과 raw delta metric 계산을 `training/`에 통합.
- **Week 7-8**:
    - **제안 모델(FusionMultiTask)** 학습: parameter branch와 visual branch를 융합하고, 탐지/추적 저하를 동시에 예측하는 multi-task 모델.
    (`models/fusion_multitask.py`, `training.train --model fusion_multitask`)
    - **절제 연구(Ablations)** 진행: `visual_single_task_map`, `visual_single_task_hota`, `fusion_single_task_map`, `fusion_single_task_hota`를 사용해 multi-task 구조와 single-task 구조를 비교.
    (`models/single_task.py`)
    - 3 seeds canonical sweep을 `training.run_seed_sweep`로 실행하고, 기본 loss는 `uncertainty_huber`를 사용.

## 4. Phase 3: 평가, 검증 및 마무리 (9~12주차)

학습된 제안 모델의 최종 성능을 다양한 지표로 **평가하고, 한계점을 검증하며, 연구 결과를 문서화**합니다.

- **Week 9-10**:
    - **성능 평가**: 예측 오차(RMSE) 및 실제 성능 저하와의 순위 상관관계(Rank Correlation) 측정.
    (`training.evaluate`)
    - **효율성 평가**: 추론 속도(FPS) 측정 및 기존 순차적 파이프라인(YOLO+DeepSORT 실행) 대비 배포 시의 효율성 검증.
    (`training.benchmark_inference`)
    - seed별 결과, RQ1/RQ2 pairwise 비교, benchmark 결과를 CSV로 집계.
    (`training.aggregate_results`)
- **Week 11**:
    - **일반화 성능(OOD Generalization) 검증**: 모델이 학습하지 않은 완전히 새로운 열화 방식(예: 국부 H.264 압축 분할셋)에 대해 예측 성능이 유지되는지(Out-of-Distribution 실험) 확인.
    - OOD metric은 `ood_rmse_*`, `ood_mae_*`, single-task의 `active_ood_rmse` 컬럼을 기준으로 해석.
- **Week 12**:
    - 전체 실험 결과 및 시사점을 종합하여 **최종 연구 보고서(Final report)** 작성.
    - 프로젝트 결과 발표(Presentation) 준비 및 마무리.

## 5. 현재 실행 단위와 산출물

최신 학습/평가 파이프라인의 실행 단위는 다음과 같습니다.

```bash
python -m training.train --model param_mlp --loss uncertainty_huber --seed 42
python -m training.evaluate --checkpoint experiments/<experiment_name>/best.pt --split test
python -m training.aggregate_results --split test
```

canonical full sweep은 RQ1 모델 3종과 RQ2 ablation 4종을 3 seeds로 실행합니다.

```bash
python -m training.run_seed_sweep \
  --models param_mlp,visual_baseline,fusion_multitask \
  --include_ablations \
  --seeds 42,123,2026 \
  --loss uncertainty_huber \
  --visual_backbone swin_tiny \
  --device cuda
```

각 실험은 `train_config.json`, `model_summary.json`, `train_history.json`, `latest.pt`, `best.pt`를 생성합니다. 평가 결과는 `eval_<split>/`, 벤치마크 결과는 `benchmark_<split>/`, 전체 집계 결과는 `aggregate_<split>/` 아래에 저장됩니다.
