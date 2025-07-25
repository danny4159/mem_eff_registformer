"""
Common Metrics Module

This module contains reusable metrics that can be used across different models.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.metric import Metric
from torchmetrics.clustering import NormalizedMutualInfoScore
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# Utility functions
def gray2rgb(x):
    """Convert grayscale to RGB by repeating channels"""
    return torch.cat((x, x, x), dim=1) if x.shape[1] == 1 else x

def flatten_to_1d(x):
    """Flatten tensor to 1D"""
    return x.view(-1)

def norm_to_uint8(x):
    """Normalize from [-1, 1] to [0, 255] and convert to uint8"""
    return ((x + 1) / 2 * 255).to(torch.uint8)


class GradientCorrelationMetric(Metric):
    """
    Gradient Correlation Metric
    
    Computes correlation between gradient magnitudes of two images.
    Useful for measuring structural similarity in medical images.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correlations", default=torch.empty(0, device=self.device), dist_reduce_fx="cat")

    def update(self, imgs1: torch.Tensor, imgs2: torch.Tensor):
        """
        Update metric with new image pairs.
        
        Args:
            imgs1: First set of images [B, C, H, W]
            imgs2: Second set of images [B, C, H, W]
        """
        assert imgs1.ndim == 4 and imgs2.ndim == 4, "Inputs must be 4D tensors"
        assert imgs1.size(0) == imgs2.size(0), "Input tensors must have the same batch size"

        for img1, img2 in zip(imgs1, imgs2):
            img1 = img1.cpu().numpy().squeeze()
            img2 = img2.cpu().numpy().squeeze()
            
            # Canny edge detection
            edges1 = cv2.Canny(img1, 170, 190)
            edges2 = cv2.Canny(img2, 30, 50)  # FIXME: mr-ct: 30, 50  mr-mr: 170, 190
            
            # Sobel gradients
            grad_x1, grad_y1 = cv2.Sobel(edges1, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(edges1, cv2.CV_64F, 0, 1, ksize=3)
            grad_x2, grad_y2 = cv2.Sobel(edges2, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(edges2, cv2.CV_64F, 0, 1, ksize=3)
            
            # Gradient magnitudes
            magnitude1 = np.sqrt(grad_x1**2 + grad_y1**2)
            magnitude2 = np.sqrt(grad_x2**2 + grad_y2**2)
            
            if magnitude1.size == 0 or magnitude2.size == 0 or np.std(magnitude1) == 0 or np.std(magnitude2) == 0:
                continue
            
            # Compute correlation
            correlation = np.corrcoef(magnitude1.flatten(), magnitude2.flatten())[0, 1]
            correlation = torch.tensor(correlation, device=imgs1.device)
            self.correlations = torch.cat([self.correlations, correlation.unsqueeze(0)])

    def compute(self):
        """Compute final metric value"""
        if self.correlations.numel() > 0:
            return self.correlations.mean()
        else:
            raise RuntimeError("GradientCorrelationMetric has no data to compute. Make sure update() was called.")


class SharpnessMetric(Metric):
    """
    Sharpness Metric
    
    Computes image sharpness using Laplacian variance.
    Higher values indicate sharper images.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("scores", default=torch.tensor([]), dist_reduce_fx="cat")

    def update(self, imgs: torch.Tensor):
        """
        Update metric with new images.
        
        Args:
            imgs: Input images [B, C, H, W]
        """
        # Ensure input is a 4D tensor [batch, channel, height, width]
        assert imgs.ndim == 4, "Input must be a 4D tensor"

        # Convert images to grayscale by averaging across the color channels
        if imgs.size(1) == 3:
            imgs = imgs.mean(dim=1, keepdim=True)

        imgs = imgs.squeeze(1)

        for i in range(imgs.size(0)):
            img = imgs[i].cpu().numpy().astype(np.float32)
            blur_map = cv2.Laplacian(img, cv2.CV_32F)
            sharpness_score = np.var(blur_map)
            sharpness_score = torch.tensor(sharpness_score, device=self.device)
            self.scores = torch.cat([self.scores, sharpness_score.unsqueeze(0)])

    def compute(self):
        """Compute final metric value"""
        if self.scores.numel() > 0:
            return self.scores.mean()
        else:
            raise RuntimeError("SharpnessMetric has no data to compute. Make sure update() was called.")


class ImageQualityMetrics:
    """
    Common Image Quality Metrics
    
    A collection of commonly used image quality metrics that can be used
    across different models and tasks.
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self._init_metrics()
        
        # Score storage
        self.psnr_values = []
        self.lpips_values = []
        self.nmi_scores = []
        
        # Update flags
        self.gc_updated = False
        self.fid_updated = False
        self.kid_updated = False
        self.sharpness_updated = False
    
    def _init_metrics(self):
        """Initialize all metrics"""
        self.ssim = StructuralSimilarityIndexMeasure(reduction="none").to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=2.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity().to(self.device)
        self.nmi = NormalizedMutualInfoScore().to(self.device)
        self.fid = FrechetInceptionDistance().to(self.device)
        self.kid = KernelInceptionDistance(subset_size=2).to(self.device)
        self.gc = GradientCorrelationMetric().to(self.device)
        self.sharpness = SharpnessMetric().to(self.device)
    
    def update_align_metrics(self, real, fake):
        """
        Update alignment-based metrics (SSIM, PSNR, LPIPS, Sharpness)
        
        Args:
            real: Real images [B, C, H, W]
            fake: Generated images [B, C, H, W]
        """
        self.ssim.update(real, fake)
        self.psnr.update(real, fake)
        self.psnr_values.append(self.psnr.compute().item())
        self.psnr.reset()
        self.lpips.update(gray2rgb(real), gray2rgb(fake))
        self.lpips_values.append(self.lpips.compute().item())
        self.lpips.reset()
        self.sharpness.update(norm_to_uint8(fake).float())
    
    def update_general_metrics(self, real_a, real_b, fake_b):
        """
        Update general metrics (GC, NMI, FID, KID, Sharpness)
        
        Args:
            real_a: Source images [B, C, H, W]
            real_b: Target images [B, C, H, W]
            fake_b: Generated images [B, C, H, W]
        """
        self.gc.update(norm_to_uint8(real_a), norm_to_uint8(fake_b))
        self.gc_updated = True
        
        nmi_score = self.nmi(flatten_to_1d(norm_to_uint8(real_a)), flatten_to_1d(norm_to_uint8(fake_b)))
        self.nmi_scores.append(nmi_score)
        
        self.fid.update(gray2rgb(norm_to_uint8(real_b)), real=True)
        self.fid.update(gray2rgb(norm_to_uint8(fake_b)), real=False)
        self.fid_updated = True
        
        self.kid.update(gray2rgb(norm_to_uint8(real_b)), real=True)
        self.kid.update(gray2rgb(norm_to_uint8(fake_b)), real=False)
        self.kid_updated = True
        
        self.sharpness.update(norm_to_uint8(fake_b))
        self.sharpness_updated = True
    
    def compute_align_metrics(self):
        """Compute alignment-based metrics"""
        ssim = self.ssim.compute().mean()
        
        if self.psnr_values:
            psnr = torch.mean(torch.tensor(self.psnr_values, device=self.device))
        else:
            psnr = torch.tensor(0.0, device=self.device)
        
        if self.lpips_values:
            lpips = torch.mean(torch.tensor(self.lpips_values, device=self.device))
        else:
            lpips = torch.tensor(0.0, device=self.device)
        
        sharpness = self.sharpness.compute()
        
        return {
            'ssim': ssim.item(),
            'psnr': psnr.item(),
            'lpips': lpips.item(),
            'sharpness': sharpness.item()
        }
    
    def compute_general_metrics(self):
        """Compute general metrics"""
        # Check if metrics have been updated before computing
        if not self.gc_updated:
            raise RuntimeError("GradientCorrelationMetric was not updated before compute() was called")
        if not self.fid_updated:
            raise RuntimeError("FID was not updated before compute() was called")
        if not self.kid_updated:
            raise RuntimeError("KID was not updated before compute() was called")
        if not self.sharpness_updated:
            raise RuntimeError("SharpnessMetric was not updated before compute() was called")
        
        # Compute metrics
        try:
            gc = self.gc.compute()
        except Exception as e:
            raise RuntimeError(f"GradientCorrelationMetric compute failed: {e}")
        
        if self.nmi_scores:
            nmi = torch.mean(torch.stack(self.nmi_scores))
        else:
            raise RuntimeError("NMI scores were not updated before compute() was called")
        
        # FID calculation with error handling
        try:
            fid = self.fid.compute()
        except RuntimeError as e:
            if "More than one sample is required" in str(e):
                print("Warning: FID 계산을 위한 샘플이 부족합니다. 0.0으로 설정합니다.")
                fid = torch.tensor(0.0, device=self.device)
            else:
                raise RuntimeError(f"FID compute failed: {e}")
        
        # KID calculation with error handling
        try:
            kid_mean, _ = self.kid.compute()
        except (RuntimeError, ValueError) as e:
            if ("More than one sample is required" in str(e) or 
                "No samples to concatenate" in str(e) or
                "subset_size should be smaller" in str(e)):
                print("Warning: KID 계산을 위한 샘플이 부족합니다. 0.0으로 설정합니다.")
                kid_mean = torch.tensor(0.0, device=self.device)
            else:
                raise RuntimeError(f"KID compute failed: {e}")
        
        # Sharpness calculation
        try:
            sharpness = self.sharpness.compute()
        except Exception as e:
            raise RuntimeError(f"SharpnessMetric compute failed: {e}")
        
        return {
            'gc': gc.item(),
            'nmi': nmi.item(),
            'fid': fid.item(),
            'kid': kid_mean.item(),
            'sharpness': sharpness.item()
        }
    
    def reset(self):
        """Reset all metrics"""
        self.ssim.reset()
        self.psnr.reset()
        self.lpips.reset()
        self.nmi.reset()
        self.fid.reset()
        self.kid.reset()
        self.gc.reset()
        self.sharpness.reset()
        
        # Reset score lists
        self.psnr_values = []
        self.lpips_values = []
        self.nmi_scores = []
        
        # Reset update flags
        self.gc_updated = False
        self.fid_updated = False
        self.kid_updated = False
        self.sharpness_updated = False 