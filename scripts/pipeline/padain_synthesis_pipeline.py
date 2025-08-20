#!/usr/bin/env python3
"""
Padin Synthesis Pipeline Inference Script

이 스크립트는 학습된 padin_synthesis 모델을 사용하여 MR 이미지로부터 CT 이미지를 생성합니다.
NII.gz 파일을 입력으로 받아 자동으로 전처리하고 inference를 수행한 후 결과를 저장합니다.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # GPU 3번 고정 사용

import sys
import argparse
import datetime
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import nibabel as nib
import numpy as np
from src.common.base_pipeline import BasePipeline
from src.padain_synthesis.modeling_padain_synthesis import PadainSynthesisModel, PadainSynthesisConfig
from src.padain_synthesis.trainer import load_trained_model
from src.padain_synthesis.config import MODEL_CONFIG


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Padin Synthesis Pipeline Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/synthrad2023_mr-ct_pelvis/test/1PA004/mr.nii.gz",
        help="Path to input MR NII.gz file"
    )
    
    parser.add_argument(
        "--reference_path", 
        type=str,
        default="data/synthrad2023_mr-ct_pelvis/test/1PA004/ct.nii.gz",
        help="Path to reference CT NII.gz file"
    )
    
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="None",  #"/home/milab/SSD5_8TB/Daniel/09_registformer_hugging_face/mem_eff_registformer/training_output/padain_synthesis/With_Augmentation/20250819_094314/weights/epoch5", # None
        help="Path to trained model checkpoint (if not provided, random weights will be used)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference"
    )
    
    parser.add_argument(
        "--run_name",
        type=str,
        default="padain_synthesis_inference",
        help="Name for this inference run"
    )
    
    parser.add_argument(
        "--output_root",
        type=str,
        default="pipeline_outputs",
        help="Root directory for saving outputs"
    )
    
    return parser.parse_args()


def main():
    """Main inference function."""
    args = parse_arguments()
    
    print("="*80)
    print("PADIN SYNTHESIS PIPELINE INFERENCE")
    print("="*80)
    
    # GPU 상태 확인
    print(f"🔧 GPU 설정: CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"🔧 사용 가능한 GPU: {torch.cuda.device_count()}")
    print(f"🔧 현재 GPU 디바이스: {torch.cuda.current_device()}")
    if torch.cuda.is_available():
        print(f"🔧 GPU 이름: {torch.cuda.get_device_name(0)}")
    
    print(f"Input MR path: {args.input_path}")
    print(f"Reference CT path: {args.reference_path}")
    print(f"Checkpoint path: {args.checkpoint_path}")
    print(f"Device: {args.device}")
    print(f"Run name: {args.run_name}")
    print(f"Output root: {args.output_root}")
    print("="*80)
    
    # Check if input files exist
    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input MR file not found: {args.input_path}")
    if not os.path.exists(args.reference_path):
        raise FileNotFoundError(f"Reference CT file not found: {args.reference_path}")
    
    # Load pipeline
    if args.checkpoint_path == "None" or not os.path.exists(args.checkpoint_path):
        print(f"⚠️  WARNING: Checkpoint not found or set to 'None': {args.checkpoint_path}")
        print("🎲 Using random weights for inference...")
        pipeline = BasePipeline.from_random_weights(
            config_class=PadainSynthesisConfig,
            model_class=PadainSynthesisModel,
            model_config=MODEL_CONFIG,
            device=args.device,
            model_name="PadainSynthesis"
        )
    else:
        print(f"📦 Loading trained model from: {args.checkpoint_path}")
        pipeline = BasePipeline.from_pretrained(
            checkpoint_path=args.checkpoint_path,
            config_class=PadainSynthesisConfig,
            model_class=PadainSynthesisModel,
            load_model_func=load_trained_model,
            device=args.device,
            model_name="PadainSynthesis"
        )
    
    # Create output directory structure: output_root/padain_synthesis/run_name/timestamp
    network_name = "padain_synthesis"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_root, network_name, args.run_name, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}")
    
    # Run inference
    print("🚀 Starting inference...")
    try:
        output_path = pipeline.inference_from_nii(
            input_path=args.input_path,
            reference_path=args.reference_path,
            output_dir=output_dir
        )
        print(f"✅ Inference completed successfully!")
        print(f"💾 Result saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Inference failed: {str(e)}")
        raise
    
    print("="*80)
    print("INFERENCE COMPLETED")
    print("="*80)


if __name__ == "__main__":
    main() 