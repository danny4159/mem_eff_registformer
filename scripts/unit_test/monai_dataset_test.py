import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    RandSpatialCropd,
    ScaleIntensityd,
    SaveImaged,
    LambdaD,
)
from monai.data import Dataset, DataLoader
import os
import nibabel as nib
import monai


def print_minmax_func(tag):
    def _func(x):
        print(f"{tag} shape: {tuple(x.shape)}, min/max: {x.min().item():.4f} / {x.max().item():.4f}")
        return x
    return _func

def main():
    # ---- 데이터 경로 ----
    train_img_path = "/SSD5_8TB/Daniel/09_registformer_hugging_face/mem_eff_registformer/data/synthrad2023_mr-ct_pelvis/1PA001/mr.nii.gz"
    val_img_path = "/SSD5_8TB/Daniel/09_registformer_hugging_face/mem_eff_registformer/data/synthrad2023_mr-ct_pelvis/1PA004/mr.nii.gz"

    train_files = [{"image": train_img_path}]
    val_files = [{"image": val_img_path}]

    print(f"\n🔍 Training file → {train_img_path}")
    print(f"🔍 Validation file → {val_img_path}\n")

    # ---- Training Transform ----
    train_tf = Compose([
        LoadImaged(keys="image"),
        LambdaD(keys="image", func=print_minmax_func("📦 After Load")),
        EnsureChannelFirstd(keys="image"),
        ScaleIntensityRanged(
            keys="image",
            a_min=0, a_max=255,     # CT 값 범위 기준
            b_min=-1, b_max=1,
            clip=True
        ),
        LambdaD(keys="image", func=print_minmax_func("📦 After Norm")),
        RandSpatialCropd(
            keys="image",
            roi_size=(196, 196, 1),
            random_center=True,
            random_size=False,
        ),
        LambdaD(keys="image", func=print_minmax_func("✂️ After Crop")),
    ])

    # ---- Validation Transform ----
    val_tf = Compose([
        LoadImaged(keys="image"),
        LambdaD(keys="image", func=print_minmax_func("📦 Val After Load")),
        EnsureChannelFirstd(keys="image"),
        ScaleIntensityRanged(
            keys="image",
            a_min=0, a_max=255,
            b_min=-1, b_max=1,
            clip=True
        ),
        LambdaD(keys="image", func=print_minmax_func("📦 Val After Norm")),
    ])

    # ---- Dataset & Loader ----
    train_ds = Dataset(data=train_files, transform=train_tf)
    val_ds = Dataset(data=val_files, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    first_data = monai.utils.misc.first(train_loader)
    print("=====================================================")
    print(f"🔍 First data shape: {tuple(first_data['image'].shape)}")
    print(f"🔍 First data min/max: {first_data['image'].min().item():.4f} / {first_data['image'].max().item():.4f}")

    # ---- Dummy 모델 ----
    model = torch.nn.Tanh()

    # ---- Training 테스트 ----
    print("\n" + "="*50)
    print("🏋️ TRAINING DATASET (1PA001)")
    print("="*50)

    for step, batch in enumerate(train_loader):
        x = batch["image"]
        print(f"\n🧪 Training Step {step} - Input shape : {tuple(x.shape)}")
        print(f"🧪 Training input min/max: {x.min().item():.4f} / {x.max().item():.4f}")

        out = model(x)
        print(f"🧯 Training output after tanh min/max: {out.min().item():.4f} / {out.max().item():.4f}")

        print("✅ Training completed (no denorm)")
        break

    # ---- Validation 테스트 ----
    print("\n" + "="*50)
    print("🔍 VALIDATION DATASET (1PA004)")
    print("="*50)

    for step, batch in enumerate(val_loader):
        x = batch["image"]
        print(f"\n🧪 Validation Step {step} - Input shape : {tuple(x.shape)}")
        print(f"🧪 Validation input min/max: {x.min().item():.4f} / {x.max().item():.4f}")

        out = model(x)
        print(f"🧯 Validation output after tanh min/max: {out.min().item():.4f} / {out.max().item():.4f}")

        # SaveImaged 사용하되 메타데이터 없이
        output_batch = {
            "image": out
        }

        # 저장 transform - 메타데이터 없이
        save_tf = Compose([
            SaveImaged(
                keys="image",
                output_dir="./",
                output_postfix="out",
                output_ext=".nii.gz",
                print_log=True,
                separate_folder=False,
                writer="NibabelWriter"
            )
        ])  
        save_tf(output_batch)
        print("✅ Validation output saved to current directory")
        break


if __name__ == "__main__":
    main()
