import os
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image

from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    EnsureTyped,
    Compose,
    SqueezeDimd,
    Resized,
    ResizeWithPadOrCropd,
    RandSpatialCropd,
    Lambda,
    Lambdad,
)
from monai.data import Dataset, DataLoader, GridPatchDataset, PatchIterd, ShuffleBuffer
import monai
from monai.inferers import SliceInferer


# 경로 설정
data_path = "/SSD5_8TB/Daniel/09_registformer_hugging_face/mem_eff_registformer/data/synthrad2023_mr-ct_pelvis/1PA001/mr.nii.gz"
train_files = [{"img": data_path}]

# 저장 디렉토리 설정
output_dir = "/SSD5_8TB/Daniel/09_registformer_hugging_face/mem_eff_registformer/scripts/unit_test/output"
os.makedirs(output_dir, exist_ok=True)

def save_image_as_png(tensor, filename, output_dir):
    """텐서를 PNG 이미지로 저장하는 함수"""
    # 텐서를 numpy로 변환
    if isinstance(tensor, torch.Tensor):
        img_array = tensor.detach().cpu().numpy()
    else:
        img_array = np.array(tensor)
    
    # 차원이 3차원이면 첫 번째 차원 제거 (C, H, W) → (H, W)
    if len(img_array.shape) == 3:
        img_array = img_array.squeeze(0)
    
    # -1~1 범위를 0~255 범위로 변환
    if img_array.min() < 0:  # -1~1 범위인 경우
        img_array = (img_array + 1) / 2  # -1~1 → 0~1
        img_array = (img_array * 255).astype(np.uint8)  # 0~1 → 0~255
    else:  # 0~1 범위인 경우
        img_array = (img_array * 255).astype(np.uint8)  # 0~1 → 0~255
    
    # PIL Image로 변환하여 저장
    img = Image.fromarray(img_array)
    filepath = os.path.join(output_dir, filename)
    img.save(filepath)
    print(f"이미지 저장됨: {filepath}")

# ----------- 1. Volume-Level Transform -----------
train_mr_transform = Compose([
    LoadImaged(keys=["img"]),  # (H, W, D)
    EnsureChannelFirstd(keys=["img"]),  # (H, W, D) → (C, H, W, D)
    ScaleIntensityd(keys="img"),  # [0~1 Norm]
    Lambdad(keys=["img"], func=lambda x: x * 2 - 1),  # 0~1 → -1~1
    EnsureTyped(keys=["img"]),  
])

train_dataset = Dataset(data=train_files, transform=train_mr_transform)

# ----------- 2. Iteration(2D Slice) Definition -----------
patch_func = PatchIterd( # How to iterate 3D data.
    keys=["img"],
    patch_size=(None, None, 1),  # (C, H, W, D) → (C, H, W, 1)
    start_pos=(0, 0, 0)
)

patch_transform = Compose([
    SqueezeDimd(keys=["img"], dim=-1),  # (C, H, W, 1) → (C, H, W)
    RandSpatialCropd(keys=["img"], roi_size=[128,128], random_center=True, random_size=False), # (C, H, W) → (C, 128, 128)
])

patch_dataset = GridPatchDataset(
    data=train_dataset,
    patch_iter=patch_func,
    transform=patch_transform,
    with_coordinates=False,
)

# ----------- 4. DataLoader 설정 -----------
train_loader = DataLoader(
    patch_dataset,  # 모든 슬라이스 처리
    batch_size=1,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

# First data: (B, C, H, W) → (1, 1, 128, 128)
# 최종 결과: 3D volume이 2D slice로 변환되어 128x128 크기로 통일됨

# ----------- 5. 단순 tanh 모델 정의 -----------
class TanhModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.act = torch.nn.Tanh()

    def forward(self, x):
        return self.act(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TanhModel().to(device)

# ----------- 6. Training loop (Shape 출력만) -----------
for epoch in range(1):  # 테스트용 1 epoch
    print(f"\n--- Epoch {epoch + 1} ---")
    for step, batch_data in enumerate(train_loader):
        inputs = batch_data["img"].to(device)  # shape: [B, C, H, W]
        outputs = model(inputs)
        print(f"Step {step + 1} | Input shape: {inputs.shape} → Output shape: {outputs.shape}")
        
        # 입력 이미지와 출력 이미지 저장
        save_image_as_png(inputs[0], f"input_step_{step+1:04d}.png", output_dir)
        # save_image_as_png(outputs[0], f"output_step_{step+1:04d}.png", output_dir)
        
        if step >= 20:  # 21개 이미지 생성 (더 많은 이미지 확인)
            break



# ----------- 7. Validation SliceInferer 실행 -----------

# Validation 경로 설정
val_path = "/SSD5_8TB/Daniel/09_registformer_hugging_face/mem_eff_registformer/data/synthrad2023_mr-ct_pelvis/1PA004/mr.nii.gz"
val_files = [{"img": val_path}]

# Validation transform 정의 (train과 동일하게 H, W만 resize)
val_mr_transform = Compose([
    LoadImaged(keys=["img"]),  # (H, W, D) → (H, W, D) 
    EnsureChannelFirstd(keys=["img"]),  # (H, W, D) → (C, H, W, D) 
    ScaleIntensityd(keys="img"),  # [0~1 정규화]
    Lambdad(keys=["img"], func=lambda x: x * 2 - 1),  # 0~1 → -1~1
    EnsureTyped(keys=["img"]), 
])

val_dataset = Dataset(data=val_files, transform=val_mr_transform)
val_loader = DataLoader(val_dataset, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())

# SliceInferer 정의
slice_inferer = SliceInferer( # 3D 데이터를 2D 슬라이스로 변환하는 클래스. (Sliding window inference를 상속받아)
    roi_size=(128, 128),        # Training시 사용한 크기랑 맞춰줘. 이것이 곧 Sliding window 크기.
    sw_batch_size=4,            
    spatial_dim=2,              # D축(z축)으로 슬라이싱
    device=device,
    padding_mode="replicate",
)

# 모델 평가 모드
model.eval()
with torch.no_grad():
    for val_data in val_loader:
        val_img = val_data["img"].to(device)  # (B, C, H, W, D) 
        print(f"\n[Validation] Input shape: {val_img.shape}")

        val_output = slice_inferer(val_img, model)  # → (B, C, H, W, D) 
        print(f"[Validation] Output shape: {val_output.shape}")
        print(f"[Validation] Number of slices (D): {val_output.shape[-1]}")
        break  # 하나만 확인