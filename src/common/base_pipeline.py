import torch
import numpy as np
import os
from transformers import Pipeline
import nibabel as nib
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    ScaleIntensityRanged, ScaleIntensityd, Lambdad, EnsureTyped
)
from monai.inferers import SliceInferer
from .datasets import denormalize_ct

class BasePipeline(Pipeline):
    """
    Base Pipeline for medical image synthesis.
    Can be used by any network architecture.
    """
    
    def __init__(self, model, config, device="cuda", model_name="unknown"):
        super().__init__(model=model, tokenizer=None, device=device)
        self.config = config
        self.model_name = model_name
        self.model.eval()
        
        # Setup data transforms for automatic preprocessing
        self._setup_transforms()
        
        # SliceInferer 설정
        self.slice_inferer = SliceInferer(
            roi_size=(256, 256),  # Same as training (increased from 128 to 256)
            sw_batch_size=4,
            spatial_dim=2,
            device=device,
            padding_mode="replicate",
        )

    def _setup_transforms(self):
        """Setup MONAI transforms for NII file preprocessing (same as val/test processing)."""
        
        # MR transforms (simple -1 to 1 normalization)
        self.mr_transform = Compose([
            LoadImaged(keys="mr"),
            EnsureChannelFirstd(keys="mr"),
            ScaleIntensityd(keys="mr"),  # [0~1 Norm]
            Lambdad(keys=["mr"], func=lambda x: x * 2 - 1),  # 0~1 → -1~1
            EnsureTyped(keys="mr"),
        ])
        
        # CT transforms (clipping + normalization) 
        self.ct_transform = Compose([
            LoadImaged(keys="ct"),
            EnsureChannelFirstd(keys="ct"),
            ScaleIntensityRanged(
                keys="ct",
                a_min=-1000.0, a_max=2000.0,
                b_min=-1.0, b_max=1.0,
                clip=True
            ),
            EnsureTyped(keys="ct"),
        ])

    @classmethod
    def from_pretrained(cls, checkpoint_path, config_class, model_class, load_model_func, 
                       config_path=None, device="cuda", model_name="unknown"):
        """
        Create pipeline from pretrained model.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config_class: Configuration class (e.g., PadainSynthesisConfig)
            model_class: Model class (e.g., PadainSynthesisModel)
            load_model_func: Function to load trained model
            config_path: Path to config file (optional)
            device: Device to use
            model_name: Name of the model for identification
        """
        if config_path is None:
            config_path = os.path.join(checkpoint_path, "config.json")
        config = config_class.from_json_file(config_path)
        model = load_model_func(checkpoint_path, config=config)
        return cls(model, config, device=device, model_name=model_name)
    
    @classmethod
    def from_random_weights(cls, config_class, model_class, model_config, 
                           device="cuda", model_name="unknown"):
        """Create pipeline with random weights (for testing when no checkpoint is provided)."""
        config = config_class(**model_config)
        model = model_class(config)
        model.to(device)
        print(f"⚠️  {model_name} model initialized with random weights!")
        return cls(model, config, device=device, model_name=model_name)

    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def preprocess(self, inputs, **kwargs):
        # input, reference: numpy array (1, H, W)
        input_data = inputs.get('input')
        reference_data = inputs.get('reference')
        
        merged_input = torch.cat([torch.from_numpy(input_data), torch.from_numpy(reference_data)], dim=0).unsqueeze(0).to(self.device)
        print('[pipeline.py] merged_input shape:', merged_input.shape)
        return merged_input

    def _forward(self, model_inputs, **kwargs):
        with torch.no_grad():
            outputs = self.model(model_inputs)
        return outputs

    def postprocess(self, model_outputs, **kwargs):
        output = model_outputs.last_hidden_state[0].cpu().numpy()
        print('[pipeline.py] output shape:', output.shape)
        return output

    def __call__(self, input, reference, **kwargs):
        # Backward compatibility for direct input/reference calls
        inputs = {'input': input, 'reference': reference}
        return super().__call__(inputs, **kwargs)
    
    def load_and_preprocess_nii(self, input_path, reference_path):
        """
        Load and preprocess NII.gz files with the same transforms as val/test processing.
        
        Args:
            input_path: Path to input MR NII.gz file
            reference_path: Path to reference CT NII.gz file
            
        Returns:
            tuple: (processed_input, processed_reference, original_input_nii, original_reference_nii)
        """
        print(f"📂 Loading NII files...")
        print(f"   Input (MR): {input_path}")
        print(f"   Reference (CT): {reference_path}")
        
        # Load MR data
        mr_data = {"mr": input_path}
        mr_processed = self.mr_transform(mr_data)
        mr_array = mr_processed["mr"].numpy()  # Shape: (1, H, W, D)
        
        # Load CT data
        ct_data = {"ct": reference_path}
        ct_processed = self.ct_transform(ct_data)
        ct_array = ct_processed["ct"].numpy()  # Shape: (1, H, W, D)
        
        print(f"✅ Loaded MR shape: {mr_array.shape}")
        print(f"✅ Loaded CT shape: {ct_array.shape}")
        
        # Store original NII objects for header information
        original_mr_nii = nib.load(input_path)
        original_ct_nii = nib.load(reference_path)
        
        return mr_array, ct_array, original_mr_nii, original_ct_nii
    
    def inference_from_nii(self, input_path, reference_path, output_dir):
        """
        Perform inference on NII.gz files with SliceInferer (same as danny_train.py).
        
        Args:
            input_path: Path to input MR NII.gz file
            reference_path: Path to reference CT NII.gz file
            output_dir: Directory to save output
            
        Returns:
            str: Path to saved output file
        """
        # Load and preprocess NII files
        mr_array, ct_array, original_mr_nii, original_ct_nii = self.load_and_preprocess_nii(
            input_path, reference_path
        )
        
        print(f"🔄 Using SliceInferer with roi_size=(256, 256) for inference...")
        
        # Convert to torch tensors and add batch dimension
        mr_tensor = torch.from_numpy(mr_array).unsqueeze(0).to(self.device)  # (1, 1, H, W, D)
        ct_tensor = torch.from_numpy(ct_array).unsqueeze(0).to(self.device)  # (1, 1, H, W, D)
        
        # Merge input and reference for SliceInferer (same as trainer.py)
        merged_3d_input = torch.cat([mr_tensor, ct_tensor], dim=1)  # (1, 2, H, W, D)
        
        print(f"📐 Merged 3D input shape: {merged_3d_input.shape}")
        
        # Define model wrapper for SliceInferer (same as trainer.py)
        def model_wrapper(merged_input):
            outputs = self.model(merged_input)
            return outputs.last_hidden_state
        
        # Apply SliceInferer for sliding window inference
        with torch.no_grad():
            output_volume = self.slice_inferer(merged_3d_input, model_wrapper)
        
        print(f"✅ SliceInferer output shape: {output_volume.shape}")
        
        # Convert back to numpy and remove batch dimension
        output_volume = output_volume.squeeze(0).cpu().numpy()  # (1, H, W, D)
        
        # Denormalize CT output from [-1, 1] back to [-1000, 2000] range
        print("🔄 Denormalizing CT output...")
        output_tensor = torch.from_numpy(output_volume)
        denormalized_output = denormalize_ct(output_tensor).numpy()
        
        # Remove channel dimension: (1, H, W, D) -> (H, W, D)
        denormalized_output = denormalized_output[0]
        
        print(f"✅ Denormalized output shape: {denormalized_output.shape}")
        print(f"✅ Denormalized output range: [{denormalized_output.min():.1f}, {denormalized_output.max():.1f}]")
        
        # Create output NII file with original header
        output_nii = nib.Nifti1Image(
            denormalized_output, 
            affine=original_ct_nii.affine, 
            header=original_ct_nii.header
        )
        
        # Save output
        output_path = self._save_output(output_nii, input_path, reference_path, output_dir)
        
        # Copy original files
        self._copy_original_files(input_path, reference_path, output_dir, output_path)
        
        return output_path
    
    def _save_output(self, output_nii, input_path, reference_path, output_dir):
        """Save the output NII file with appropriate naming."""
        # 환자명 추출 (파일명에서 첫 번째 부분 또는 상위 디렉토리명)
        patient_name = self._extract_patient_name(input_path, reference_path)
        
        print(f"🏷️  Detected patient name: {patient_name}")
        
        # 파일명 생성
        output_filename = f"{patient_name}_generated_ct.nii.gz"
        output_path = os.path.join(output_dir, output_filename)
        nib.save(output_nii, output_path)
        
        print(f"💾 Saved generated CT to: {output_path}")
        return output_path
    
    def _extract_patient_name(self, input_path, reference_path):
        """Extract patient name from file paths."""
        # 상위 디렉토리명에서 환자명 추출 시도
        input_parent_dir = os.path.basename(os.path.dirname(input_path))
        reference_parent_dir = os.path.basename(os.path.dirname(reference_path))
        
        # 파일명에서 환자명 추출 시도 (첫 번째 언더스코어 앞부분)
        input_basename = os.path.basename(input_path).replace('.nii.gz', '')
        reference_basename = os.path.basename(reference_path).replace('.nii.gz', '')
        
        # 환자명 결정 (디렉토리명 우선, 그 다음 파일명에서 추출)
        patient_name = None
        
        # 1. 디렉토리명이 숫자나 문자로 시작하면 환자명으로 간주
        if input_parent_dir and (input_parent_dir[0].isalnum() and input_parent_dir not in ['test', 'train', 'val', 'try']):
            patient_name = input_parent_dir
        elif reference_parent_dir and (reference_parent_dir[0].isalnum() and reference_parent_dir not in ['test', 'train', 'val', 'try']):
            patient_name = reference_parent_dir
        
        # 2. 파일명에서 언더스코어로 분리해서 첫 번째 부분 추출
        if not patient_name:
            if '_' in input_basename:
                patient_name = input_basename.split('_')[0]
            elif '_' in reference_basename:
                patient_name = reference_basename.split('_')[0]
        
        # 3. 기본값 설정
        if not patient_name:
            patient_name = "unknown"
            
        return patient_name
    
    def _copy_original_files(self, input_path, reference_path, output_dir, output_path):
        """Copy original input and reference files to output directory."""
        import shutil
        
        patient_name = self._extract_patient_name(input_path, reference_path)
        
        # Copy original MR file
        original_mr_filename = f"{patient_name}_original_mr.nii.gz"
        original_mr_path = os.path.join(output_dir, original_mr_filename)
        shutil.copy2(input_path, original_mr_path)
        print(f"💾 Copied original MR to: {original_mr_path}")
        
        # Copy original CT file
        original_ct_filename = f"{patient_name}_original_ct.nii.gz"
        original_ct_path = os.path.join(output_dir, original_ct_filename)
        shutil.copy2(reference_path, original_ct_path)
        print(f"💾 Copied original CT to: {original_ct_path}") 