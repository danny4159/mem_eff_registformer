import os
import numpy as np
import nibabel as nib
import torch
from monai.transforms import Compose, ScaleIntensityRange, EnsureType

# 폴더 경로
data_root = "/SSD5_8TB/Daniel/09_registformer_hugging_face/mem_eff_registformer/data/synthrad2023_mr-ct_pelvis"

# 환자 리스트
patients = sorted(os.listdir(data_root))
patients = [p for p in patients if os.path.isdir(os.path.join(data_root, p))]

# 변환 정의
transform = Compose([
    EnsureType(),
    ScaleIntensityRange(
        a_min=-1024.0, a_max=2000.0,
        b_min=-1.0, b_max=1.0,
        clip=True
    )
])

# 역변환 정의    # ScaleIntensityRange는 InvertibleTransform를 상속받지 않아서, inverse()와 같이 간단히 호출해서 denorm 불가. CustomTransform을 구현하면 가능.
denorm_transform = Compose([
    EnsureType(),
    ScaleIntensityRange(
        a_min=-1.0, a_max=1.0,
        b_min=-1024.0, b_max=2000.0,
        clip=True
    )
])

# 각 환자 처리
for patient in patients:
    img_path = os.path.join(data_root, patient, "ct.nii.gz")
    if not os.path.exists(img_path):
        print(f"[{patient}] ct.nii.gz 없음 - 건너뜀")
        continue

    # 이미지 로드
    img_np = nib.load(img_path).get_fdata()
    orig_min, orig_max = img_np.min(), img_np.max()
    print(f"[{patient}] 원본 min={orig_min:.1f}, max={orig_max:.1f}")

    # 클리핑
    clipped_np = np.clip(img_np, a_min=-1024.0, a_max=2000.0)
    print(f"[{patient}] clip 후: min={clipped_np.min():.1f}, max={clipped_np.max():.1f}")

    # 텐서로 변환
    img_tensor = torch.tensor(clipped_np, dtype=torch.float32)

    # 정규화
    norm_tensor = transform(img_tensor)
    print(f"[{patient}] 정규화: min={norm_tensor.min():.3f}, max={norm_tensor.max():.3f}")

    # 역정규화
    restored_tensor = denorm_transform(norm_tensor)
    print(f"[{patient}] 복원: min={restored_tensor.min():.1f}, max={restored_tensor.max():.1f}\n")
