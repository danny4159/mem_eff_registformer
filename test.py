import argparse
import torch
import json
from src.padain_synthesis import (
    PadainSynthesisModel, 
    PadainSynthesisConfig, 
    DEFAULT_TRAINING_ARGS,
    DEFAULT_TRAINING_PARAMS
)
from src.padain_synthesis.dataset import PadainSynthesisDataset
from src.padain_synthesis.trainer import create_padain_synthesis_trainer, load_trained_model
from torch.utils.data import DataLoader
import h5py
import nibabel as nib
import numpy as np
import os

def save_nii(volume, save_path):
    nii_img = nib.Nifti1Image(volume, affine=np.eye(4))
    nib.save(nii_img, save_path)

def run_inference_and_save(model, dataset, h5_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with h5py.File(h5_path, 'r') as f:
        # 첫 번째 그룹명 자동 추출
        first_group_name = list(f.keys())[0]
        group = f[first_group_name]
        patient_names = list(group.keys())

    # 환자별 인덱스 매핑 생성
    patient_to_indices = {}
    for patient, start, count in zip(dataset.patient_keys, dataset.cumulative_slice_counts, dataset.slice_counts):
        patient_to_indices[patient] = list(range(start, start + count))

    for patient in patient_names:
        indices = patient_to_indices[patient]
        real_a_slices, real_b_slices, fake_b_slices = [], [], []
        for idx in indices:
            sample = dataset[idx]
            real_a = sample['input'].cpu().numpy()
            real_b = sample['target'].cpu().numpy()
            # merged_input 생성 (input, reference 모두 (C, H, W)라면 dim=0, (B, C, H, W)라면 dim=1)
            merged_input = torch.cat([sample['input'], sample['reference']], dim=0).unsqueeze(0).to(model.device)
            with torch.no_grad():
                fake_b = model(merged_input).last_hidden_state[0].cpu().numpy()
            # shape 출력
            print(f"[{patient}] idx={idx} real_a shape: {real_a.shape}, real_b shape: {real_b.shape}, fake_b shape: {fake_b.shape}")
            real_a_slices.append(real_a)
            real_b_slices.append(real_b)
            fake_b_slices.append(fake_b)

        real_a_vol = np.stack(real_a_slices, axis=0)  # (D, 1, H, W)
        real_b_vol = np.stack(real_b_slices, axis=0)
        fake_b_vol = np.stack(fake_b_slices, axis=0)

        # (D, 1, H, W) → (D, H, W)
        real_a_vol = np.squeeze(real_a_vol, axis=1)
        real_b_vol = np.squeeze(real_b_vol, axis=1)
        fake_b_vol = np.squeeze(fake_b_vol, axis=1)

        # (D, H, W) → (H, W, D)
        real_a_vol = np.transpose(real_a_vol, (1, 2, 0))
        real_b_vol = np.transpose(real_b_vol, (1, 2, 0))
        fake_b_vol = np.transpose(fake_b_vol, (1, 2, 0))

        # width(가로, axis=1) 뒤집기
        real_a_vol = np.flip(real_a_vol, axis=1)
        real_b_vol = np.flip(real_b_vol, axis=1)
        fake_b_vol = np.flip(fake_b_vol, axis=1)

        print(f"[{patient}] final real_a_vol shape: {real_a_vol.shape}")
        print(f"[{patient}] final real_b_vol shape: {real_b_vol.shape}")
        print(f"[{patient}] final fake_b_vol shape: {fake_b_vol.shape}")

        save_nii(real_a_vol, os.path.join(output_dir, f"{patient}_real_a.nii.gz"))
        save_nii(real_b_vol, os.path.join(output_dir, f"{patient}_real_b.nii.gz"))
        save_nii(fake_b_vol, os.path.join(output_dir, f"{patient}_fake_b.nii.gz"))

def run_inference(model, dataset, batch_size=1):
    loader = DataLoader(dataset, batch_size=batch_size)
    model.eval()
    results = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            output = model(**batch)
            results.append(output)
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True, help='학습된 모델 체크포인트 경로')
    parser.add_argument('--test_h5_path', type=str, required=True, help='테스트 H5 파일 경로')
    parser.add_argument('--config_path', type=str, required=True, help='모델 config 파일 경로')
    parser.add_argument('--params_path', type=str, required=True, help='params 파일 경로')
    parser.add_argument('--output_dir', type=str, default='results', help='결과 저장 폴더')
    parser.add_argument('--data_config_path', type=str, required=True, help='data_config 파일 경로')
    args = parser.parse_args()

    # config, params 파일에서 로드
    with open(args.config_path, 'r') as f:
        config_dict = json.load(f)
    config = PadainSynthesisConfig(**config_dict)

    with open(args.params_path, 'r') as f:
        params = json.load(f)

    with open(args.data_config_path, "r") as f:
        data_config = json.load(f)

    exclude_keys = [
        "train_file", "val_file", "test_file",
        "batch_size", "num_workers", "pin_memory",
        "resize_size"
    ]
    # 여기서 None 값도 제외!
    filtered_data_config = {k: v for k, v in data_config.items() if k not in exclude_keys and v is not None}

    # 필수 positional 인자 추출
    required_keys = [
        "data_group_1", "data_group_2", "data_group_3", "data_group_4", "data_group_5", "is_3d", "padding_size"
    ]
    required_args = {k: data_config.get(k) for k in required_keys}

    # 나머지 키워드 인자 추출
    exclude_keys = [
        "train_file", "val_file", "test_file",
        "batch_size", "num_workers", "pin_memory",
        "resize_size"
    ] + required_keys
    optional_args = {k: v for k, v in data_config.items() if k not in exclude_keys and v is not None}

    # 데이터셋 생성
    test_dataset = PadainSynthesisDataset(
        h5_file_path=args.test_h5_path,
        **required_args,
        **optional_args
    )

    # 모델 로드
    model = load_trained_model(args.checkpoint_path, config=config)

    # 기본 훈련 설정 사용 (배치 크기 동적 설정)
    training_args = DEFAULT_TRAINING_ARGS.copy()
    training_args.update({
        "per_device_train_batch_size": data_config["batch_size"],
        "per_device_eval_batch_size": 1,  # 평가는 항상 배치 크기 1
        "output_dir": args.output_dir,
    })

    # Loss function weights and parameters (배치 크기 동적 설정)
    training_params = DEFAULT_TRAINING_PARAMS.copy()
    training_params.update({
        "batch_size": data_config["batch_size"],  # data_config에서 가져온 배치 크기
    })

    # Trainer 생성
    trainer = create_padain_synthesis_trainer(
        model=model,
        train_dataset=None,
        training_args=training_args,
        params=training_params,
        eval_dataset=test_dataset,
    )

    # 평가
    outputs = run_inference(model, test_dataset, batch_size=1)

    run_inference_and_save(model, test_dataset, args.test_h5_path, args.output_dir)

if __name__ == '__main__':
    main()
