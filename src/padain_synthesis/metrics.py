"""
PadainSynthesis Metrics Module

This module contains metrics specific to the PadainSynthesis model.
Common metrics are imported from common_metrics.py.
"""

from src.common.metrics import ImageQualityMetrics


class PadainSynthesisMetrics:
    """PadainSynthesis 모델을 위한 메트릭 관리 클래스"""
    
    def __init__(self, eval_on_align=False, device='cuda'):
        self.eval_on_align = eval_on_align
        self.device = device
        
        # 공통 메트릭 초기화
        self.common_metrics = ImageQualityMetrics(device=device)
    
    def update_metrics(self, real_a, real_b, fake_b):
        """
        메트릭 업데이트
        
        Args:
            real_a: Input images [B, C, H, W]
            real_b: Reference images [B, C, H, W]
            fake_b: Generated images [B, C, H, W]
        """
        if self.eval_on_align:
            self._update_align_metrics(real_b, fake_b)
        else:
            self._update_general_metrics(real_a, real_b, fake_b)
    
    def _update_align_metrics(self, real_b, fake_b):
        """정렬 평가용 메트릭 업데이트"""
        self.common_metrics.update_align_metrics(real_b, fake_b)
    
    def _update_general_metrics(self, real_a, real_b, fake_b):
        """일반 평가용 메트릭 업데이트"""
        self.common_metrics.update_general_metrics(real_a, real_b, fake_b)
    
    def compute_metrics(self):
        """메트릭 계산"""
        if self.eval_on_align:
            # 정렬 평가용 메트릭 계산
            metrics = self.common_metrics.compute_align_metrics()
            # PadainSynthesis 전용 접미사 추가
            return {f"{key}_B": value for key, value in metrics.items()}
        else:
            # 일반 평가용 메트릭 계산
            metrics = self.common_metrics.compute_general_metrics()
            # PadainSynthesis 전용 접미사 추가
            return {f"{key}_B": value for key, value in metrics.items()}
    
    def reset_metrics(self):
        """메트릭 리셋"""
        self.common_metrics.reset() 