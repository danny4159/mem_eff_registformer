"""
Base Metrics Module

This module contains a base metrics class that can be used by any network.
Common metrics are imported from common_metrics.py.
"""

from src.common.metrics import ImageQualityMetrics


class BaseMetrics:
    """모든 네트워크에서 사용할 수 있는 기본 메트릭 관리 클래스"""
    
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
            self._update_misalign_metrics(real_a, real_b, fake_b)
    
    def _update_align_metrics(self, real_b, fake_b):
        """정렬 평가용 메트릭 업데이트"""
        self.common_metrics.update_align_metrics(real_b, fake_b)
    
    def _update_misalign_metrics(self, real_a, real_b, fake_b):
        """비정렬 평가용 메트릭 업데이트"""
        self.common_metrics.update_general_metrics(real_a, real_b, fake_b)
    
    def compute_metrics(self, suffix="B"):
        """
        메트릭 계산
        
        Args:
            suffix (str): 메트릭 키에 추가할 접미사 (네트워크별로 다르게 설정 가능)
        """
        if self.eval_on_align:
            # 정렬 평가용 메트릭 계산
            metrics = self.common_metrics.compute_align_metrics()
            return {f"{key}_{suffix}": value for key, value in metrics.items()}
        else:
            # 비정렬 평가용 메트릭 계산
            metrics = self.common_metrics.compute_general_metrics()
            return {f"{key}_{suffix}": value for key, value in metrics.items()}
    
    def reset_metrics(self):
        """메트릭 리셋"""
        self.common_metrics.reset() 