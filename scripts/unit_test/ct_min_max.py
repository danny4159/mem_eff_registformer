import os
import numpy as np
import nibabel as nib

# 폴더 경로
data_root = "/SSD5_8TB/Daniel/09_registformer_hugging_face/mem_eff_registformer/data/synthrad2023_mr-ct_pelvis"

# 환자 리스트
patients = sorted(os.listdir(data_root))
patients = [p for p in patients if os.path.isdir(os.path.join(data_root, p))]

for patient in patients:
    img_path = os.path.join(data_root, patient, "ct.nii.gz")
    if not os.path.exists(img_path):
        print(f"[{patient}] mr.nii.gz 없음 - 건너뜀")
        continue

    # 이미지 로드
    img_np = nib.load(img_path).get_fdata()
    orig_min, orig_max = img_np.min(), img_np.max()
    print(f"[{patient}] min={orig_min:.3f}, max={orig_max:.3f}")
