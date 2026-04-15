import os
import shutil
import pandas as pd
import yaml
from pathlib import Path
import configparser
from tqdm import tqdm
import warnings

# =============================================================================
# 1. 설정 및 초기화
# =============================================================================
# Manifest 파일의 고정 컬럼 (데이터가 없을 때도 헤더를 유지하기 위함)
MANIFEST_COLUMNS = [
    "clip_id", "sequence_name", "split", "start_frame", "end_frame",
    "degradation_type", "degradation_param", "file_path", "active_trajectories"
]

def load_config(config_path="configs/config.yaml"):
    """YAML 설정 파일을 로드합니다."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def parse_mot_gt(gt_path):
    """
    MOT17의 gt.txt 파일을 파싱합니다.
    포맷: frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, vis
    """
    columns = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'class', 'vis']
    df = pd.read_csv(gt_path, header=None, names=columns)
    # 보행자(class=1)만 필터링
    pedestrians = df[df['class'] == 1]
    return pedestrians

def get_sequence_info(seq_dir):
    """seqinfo.ini 파일에서 시퀀스의 총 프레임 길이와 이미지 확장자를 추출합니다."""
    config = configparser.ConfigParser()
    config.read(seq_dir / "seqinfo.ini")
    seq_length = int(config["Sequence"]["seqLength"])
    im_ext = config["Sequence"]["imExt"]
    return seq_length, im_ext

# =============================================================================
# 2. 클립 생성 및 검증 메인 로직
# =============================================================================
def process_sequence(seq_name, split, config, raw_dir, clip_out_dir):
    """단일 시퀀스에 대해 슬라이딩 윈도우를 적용하고 클립을 생성합니다."""
    seq_dir = raw_dir / seq_name
    gt_path = seq_dir / "gt" / "gt.txt"
    img_dir = seq_dir / "img1"
    
    clip_len = config['clip_generation']['clip_length']
    stride = config['clip_generation']['stride']
    min_traj = config['clip_generation']['min_active_trajectories']
    
    gt_df = parse_mot_gt(gt_path)
    seq_length, im_ext = get_sequence_info(seq_dir)
    
    manifest_records = []
    
    for start_frame in range(1, seq_length - clip_len + 2, stride):
        end_frame = start_frame + clip_len - 1
        
        # 1. 활성 궤적 수 검증
        clip_gt = gt_df[(gt_df['frame'] >= start_frame) & (gt_df['frame'] <= end_frame)]
        active_trajectories = clip_gt['id'].nunique()
        if active_trajectories < min_traj:
            continue
            
        # 2. 물리적 이미지 프레임 누락 검증 (엄격한 검사)
        missing_frames = []
        for f in range(start_frame, end_frame + 1):
            if not (img_dir / f"{f:06d}{im_ext}").exists():
                missing_frames.append(f)
                
        if missing_frames:
            warnings.warn(f"[{seq_name}] {start_frame}~{end_frame} 구간에 누락된 프레임이 존재하여 건너뜁니다. (Missing: {missing_frames})")
            continue
            
        # 3. 클립 디렉토리 생성 및 파일 복사
        clip_id = config['naming']['clip_id_format'].format(
            sequence=seq_name, start_frame=start_frame, end_frame=end_frame
        )
        clip_path = clip_out_dir / clip_id
        clip_path.mkdir(parents=True, exist_ok=True)
        
        for f in range(start_frame, end_frame + 1):
            src_img = img_dir / f"{f:06d}{im_ext}"
            dst_img = clip_path / f"{f:06d}{im_ext}"
            shutil.copy2(src_img, dst_img)
        
        # 4. Manifest 레코드 생성
        record = {
            "clip_id": clip_id,
            "sequence_name": seq_name,
            "split": split,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "degradation_type": "original",
            "degradation_param": None,
            "file_path": clip_path.as_posix(), # Windows 환경에서도 호환되는 POSIX 경로 사용
            "active_trajectories": active_trajectories
        }
        manifest_records.append(record)
        
    return manifest_records

# =============================================================================
# 3. 실행 파이프라인
# =============================================================================
def main():
    print("[INFO] 클립 생성 파이프라인 시작...")
    config = load_config()
    
    raw_dir = Path(config['paths']['raw_dir']).resolve()
    clip_out_dir = Path(config['paths']['clip_output_dir']) / "original"
    manifest_dir = Path(config['paths']['manifest_dir'])
    
    clip_out_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    
    all_manifest_records = []
    
    dataset_splits = config['dataset']
    for split_name, sequences in dataset_splits.items():
        print(f"\n[INFO] {split_name.upper()} 분할 처리 중...")
        
        for seq in tqdm(sequences, desc=f"Sequences in {split_name}"):
            records = process_sequence(seq, split_name, config, raw_dir, clip_out_dir)
            all_manifest_records.extend(records)
            
    # 빈 데이터셋이어도 명시적 컬럼을 지정하여 스키마 유지
    manifest_df = pd.DataFrame(all_manifest_records, columns=MANIFEST_COLUMNS)
    manifest_path = manifest_dir / "clip_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    
    print(f"\n[SUCCESS] 파이프라인 완료. 총 {len(manifest_df)}개의 원본 클립이 생성되었습니다.")
    print(f"[SUCCESS] Manifest 파일 저장 위치: {manifest_path}")

if __name__ == "__main__":
    # 스크립트 실행 기준을 항상 프로젝트 루트로 고정
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)
    
    main()