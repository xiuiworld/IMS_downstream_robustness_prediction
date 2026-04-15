import os
import cv2
import numpy as np
import pandas as pd
import yaml
import tempfile
import subprocess
import configparser
import shutil
from pathlib import Path
from tqdm import tqdm

# =============================================================================
# 1. 설정 및 초기화
# =============================================================================
def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_sequence_info(seq_dir):
    """seqinfo.ini 파일에서 이미지 확장자를 추출합니다."""
    config = configparser.ConfigParser()
    config.read(seq_dir / "seqinfo.ini")
    return config["Sequence"]["imExt"]

def get_gt_boxes_for_frame(gt_df, frame_idx):
    """특정 프레임의 보행자 GT 박스 좌표(x, y, w, h)를 반환합니다."""
    frame_gt = gt_df[gt_df['frame'] == frame_idx]
    if frame_gt.empty:
        return np.array([])
    boxes = frame_gt[['bb_left', 'bb_top', 'bb_width', 'bb_height']].values
    return np.maximum(boxes, 0).astype(int)

def check_ffmpeg_installed():
    """시스템에 ffmpeg가 설치되어 있는지 확인합니다."""
    if shutil.which("ffmpeg") is None:
        raise EnvironmentError("ffmpeg가 설치되어 있지 않거나 PATH에 추가되어 있지 않습니다. 시스템 재시작 또는 터미널 재실행이 필요합니다.")

# =============================================================================
# 2. 난독화(Degradation) 처리 엔진
# =============================================================================
def apply_blur_to_frames(frames, kernel_size):
    return [cv2.GaussianBlur(img, (kernel_size, kernel_size), 0) for img in frames]

def apply_pixelate_to_frames(frames, block_size):
    res = []
    for img in frames:
        h, w = img.shape[:2]
        temp = cv2.resize(img, (w // block_size, h // block_size), interpolation=cv2.INTER_LINEAR)
        res.append(cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST))
    return res

def generate_h264_frames(frames, crf, fps=30):
    """
    클립 전체 프레임을 임시 mp4(libx264)로 인코딩한 뒤 다시 추출하여
    실제 H.264의 시간적/공간적 압축 열화(Compression Artifacts)를 시뮬레이션합니다.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        
        # 1. 원본 프레임을 임시 폴더에 쓰기 (JPEG 이중 열화 방지를 위해 PNG 무손실 사용)
        for i, frame in enumerate(frames):
            cv2.imwrite(str(temp_dir_path / f"{i:06d}.png"), frame)
            
        # 2. ffmpeg를 이용한 H.264 CRF 인코딩
        input_pattern = str(temp_dir_path / "%06d.png")
        output_mp4 = str(temp_dir_path / "compressed.mp4")
        
        cmd = [
            "ffmpeg", "-y", "-framerate", str(fps),
            "-start_number", "0",
            "-i", input_pattern,
            "-c:v", "libx264", "-crf", str(crf),
            "-pix_fmt", "yuv420p", output_mp4
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        # 3. 인코딩된 영상에서 프레임 다시 읽기
        cap = cv2.VideoCapture(output_mp4)
        compressed_frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            compressed_frames.append(frame)
        cap.release()
        
        return compressed_frames

# =============================================================================
# 3. 마스크 생성 및 블렌딩
# =============================================================================
def create_feathered_mask(image_shape, boxes, feather_ksize=21):
    mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.float32)
    for (x, y, w, h) in boxes:
        y_end, x_end = min(y + h, image_shape[0]), min(x + w, image_shape[1])
        mask[y:y_end, x:x_end] = 1.0
        
    if len(boxes) > 0:
        mask = cv2.GaussianBlur(mask, (feather_ksize, feather_ksize), 0)
    return np.expand_dims(mask, axis=-1)

# =============================================================================
# 4. 메인 파이프라인
# =============================================================================
def main():
    print("[INFO] 환경 점검 중...")
    check_ffmpeg_installed()
    
    print("[INFO] 국소적 난독화(Localized Obfuscation) 파이프라인 시작...")
    config = load_config()
    
    raw_dir = Path(config['paths']['raw_dir'])
    interim_dir = Path(config['paths']['interim_dir'])
    manifest_path = Path(config['paths']['manifest_dir']) / "clip_manifest.csv"
    
    clip_len = config['clip_generation']['clip_length']
    
    # 멱등성 보장: 기존 manifest에서 'original'만 베이스로 가져옴
    df = pd.read_csv(manifest_path)
    original_clips = df[df['degradation_type'] == 'original'].copy()
    
    # 최종 저장될 레코드 리스트 (original 레코드 먼저 삽입)
    final_manifest_records = original_clips.to_dict('records')
    
    obf_config = config['obfuscation']
    tasks = []
    for k in obf_config['blur_kernels']: tasks.append(('blur', k))
    for b in obf_config['pixelate_blocks']: tasks.append(('pixelate', b))
    
    # 클립 단위 순회
    for idx, row in tqdm(original_clips.iterrows(), total=len(original_clips), desc="Processing Clips"):
        clip_id = row['clip_id']
        seq_name = row['sequence_name']
        start_f, end_f = row['start_frame'], row['end_frame']
        split = row['split']
        orig_clip_dir = Path(row['file_path'])
        
        # 확장자 파싱 및 GT 로드
        im_ext = get_sequence_info(raw_dir / seq_name)
        gt_path = raw_dir / seq_name / "gt" / "gt.txt"
        columns = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'class', 'vis']
        gt_df = pd.read_csv(gt_path, header=None, names=columns)
        gt_df = gt_df[gt_df['class'] == 1]
        
        # 1. 원본 프레임 및 마스크 메모리 일괄 로드
        orig_frames = []
        masks = []
        load_success = True
        for frame_idx in range(start_f, end_f + 1):
            img_path = orig_clip_dir / f"{frame_idx:06d}{im_ext}"
            img = cv2.imread(str(img_path))
            if img is None:
                load_success = False
                break
            orig_frames.append(img)
            boxes = get_gt_boxes_for_frame(gt_df, frame_idx)
            masks.append(create_feathered_mask(img.shape, boxes))
            
        if not load_success or len(orig_frames) != clip_len:
            print(f"\n[WARNING] {clip_id} 프레임 누락으로 스킵됩니다.")
            continue
            
        # 2. 난독화 태스크 구성 (test 셋에만 h264_local 추가)
        clip_tasks = tasks.copy()
        if split == 'test':
            for v in obf_config['compression_crfs']:
                clip_tasks.append(('h264_local', v))
                
        # 3. 태스크별 난독화 처리 및 저장 (try-except-finally 구조를 통한 엄격한 롤백 보장)
        for deg_type, param in clip_tasks:
            deg_clip_id = config['naming']['degraded_clip_format'].format(clip_id=clip_id, deg_type=deg_type, deg_param=param)
            out_dir = interim_dir / "clips" / "obfuscated" / deg_type / deg_clip_id
            
            # [수정됨] 완벽한 재개(Resume) 지원: 파일뿐만 아니라 폴더 내 모든 항목을 포함하여 엄격히 비교
            expected_files = {f"{f:06d}{im_ext}" for f in range(start_f, end_f + 1)}
            if out_dir.exists():
                existing_items = {p.name for p in out_dir.iterdir()}
                if expected_files == existing_items:
                    new_record = row.copy()
                    new_record['degradation_type'] = deg_type
                    new_record['degradation_param'] = param
                    new_record['file_path'] = out_dir.as_posix()
                    final_manifest_records.append(new_record.to_dict())
                    continue
                
            write_success = False
            temp_dir = None
            try:
                if deg_type == 'blur':
                    deg_frames = apply_blur_to_frames(orig_frames, param)
                elif deg_type == 'pixelate':
                    deg_frames = apply_pixelate_to_frames(orig_frames, param)
                elif deg_type == 'h264_local':
                    deg_frames = generate_h264_frames(orig_frames, param)
                    
                if len(deg_frames) != clip_len:
                    raise ValueError(f"생성된 프레임 수({len(deg_frames)})가 불일치합니다.")
                    
                # 원자적 쓰기(Atomic Write)를 위한 임시 폴더 생성
                out_dir.parent.mkdir(parents=True, exist_ok=True)
                temp_dir = Path(tempfile.mkdtemp(dir=out_dir.parent, prefix="tmp_"))
                
                # 임시 폴더에 이미지 쓰기 처리
                for i in range(clip_len):
                    final_img = (deg_frames[i] * masks[i] + orig_frames[i] * (1 - masks[i])).astype(np.uint8)
                    save_path = temp_dir / f"{start_f + i:06d}{im_ext}"
                    if not cv2.imwrite(str(save_path), final_img):
                        raise IOError(f"이미지 저장 실패: {save_path}")
                        
                write_success = True
                
            except Exception as e:
                print(f"\n[ERROR] {clip_id} - {deg_type}({param}) 처리 중 오류 발생: {e}")
                
            finally:
                if write_success:
                    backup_dir = None
                    backup_created = False
                    temp_renamed = False
                    try:
                        # 1. 기존 폴더 백업
                        if out_dir.exists():
                            backup_dir = out_dir.with_name(out_dir.name + "_backup")
                            if backup_dir.exists():
                                shutil.rmtree(backup_dir)
                            out_dir.rename(backup_dir)
                            backup_created = True
                        
                        # 2. 임시 폴더를 실제 폴더명으로 변경 (원자적 교체)
                        temp_dir.rename(out_dir)
                        temp_renamed = True
                        
                        # [수정됨] 3. 교체 성공 시 즉시 Manifest 레코드 추가 (백업 삭제 실패로 인한 데이터 유실 방지)
                        new_record = row.copy()
                        new_record['degradation_type'] = deg_type
                        new_record['degradation_param'] = param
                        new_record['file_path'] = out_dir.as_posix()
                        final_manifest_records.append(new_record.to_dict())
                        
                        # 4. 교체 성공 시 백업 폴더 완전 삭제 (실패해도 데이터 무결성에는 영향 없음)
                        if backup_dir and backup_dir.exists():
                            try:
                                shutil.rmtree(backup_dir)
                            except Exception as cleanup_err:
                                print(f"[WARNING] {deg_clip_id} 백업 폴더({backup_dir.name}) 삭제 실패 (무시 가능): {cleanup_err}")
                        
                    except Exception as rename_err:
                        print(f"\n[ERROR] {deg_clip_id} 폴더 교체 중 시스템 오류 발생: {rename_err}")
                        # 상태 기반 안전한 롤백 처리
                        try:
                            if backup_created and not temp_renamed:
                                if out_dir.exists():
                                    shutil.rmtree(out_dir)
                                backup_dir.rename(out_dir)
                        except Exception as rollback_err:
                            print(f"\n[FATAL] {deg_clip_id} 롤백 중 치명적 오류 발생 (수동 복구 필요): {rollback_err}")
                            
                        # 임시 폴더 삭제 시도
                        if temp_dir and temp_dir.exists():
                            try:
                                shutil.rmtree(temp_dir)
                            except Exception:
                                pass
                else:
                    # 실패 시 임시 폴더만 삭제 (기존에 성공해둔 out_dir는 건드리지 않음)
                    if temp_dir and temp_dir.exists():
                        try:
                            shutil.rmtree(temp_dir)
                        except Exception:
                            pass
                    print(f"[INFO] {deg_clip_id} 생성이 롤백되었습니다.")

    # 전체 완료 후 덮어쓰기 (멱등성 보장)
    pd.DataFrame(final_manifest_records).to_csv(manifest_path, index=False)
    
    # 추가된 클립 수 계산
    new_count = len(final_manifest_records) - len(original_clips)
    print(f"\n[SUCCESS] 난독화 파이프라인 완료. 총 {new_count}개의 유효한 난독화 클립이 추가되었습니다.")
    print(f"[SUCCESS] 덮어쓰기 완료된 Manifest: {manifest_path}")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)
    main()