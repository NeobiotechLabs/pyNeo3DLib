"""
성능 측정 유틸리티
단일책임: 처리 시간 측정 및 로깅
"""

import time
from typing import Optional, Dict
from contextlib import contextmanager


class PerformanceTimer:
    """성능 측정을 위한 타이머 클래스"""
    
    def __init__(self, enabled: bool = True, verbose: bool = True):
        """
        초기화
        
        Args:
            enabled: 타이머 활성화 여부
            verbose: 측정 결과 출력 여부
        """
        self.enabled = enabled
        self.verbose = verbose
        self.timings: Dict[str, float] = {}
        self._start_time: Optional[float] = None
        self._current_label: Optional[str] = None
    
    @contextmanager
    def measure(self, label: str):
        """
        컨텍스트 매니저를 사용한 시간 측정
        
        Usage:
            with timer.measure("작업명"):
                # 측정할 코드
                
        Args:
            label: 측정 작업의 이름
        """
        if not self.enabled:
            yield
            return
        
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self.timings[label] = elapsed
            if self.verbose:
                print(f"{label}: {elapsed:.4f}초")
    
    def start(self, label: str):
        """
        시간 측정 시작
        
        Args:
            label: 측정 작업의 이름
        """
        if not self.enabled:
            return
        
        self._start_time = time.time()
        self._current_label = label
    
    def stop(self):
        """시간 측정 종료"""
        if not self.enabled or self._start_time is None:
            return
        
        elapsed = time.time() - self._start_time
        self.timings[self._current_label] = elapsed
        
        if self.verbose:
            print(f"{self._current_label}: {elapsed:.4f}초")
        
        self._start_time = None
        self._current_label = None
    
    def get_timing(self, label: str) -> Optional[float]:
        """
        특정 작업의 측정 시간을 반환
        
        Args:
            label: 측정 작업의 이름
            
        Returns:
            Optional[float]: 측정 시간 (초) 또는 None
        """
        return self.timings.get(label)
    
    def get_all_timings(self) -> Dict[str, float]:
        """
        모든 측정 시간을 반환
        
        Returns:
            Dict[str, float]: {작업명: 시간} 딕셔너리
        """
        return self.timings.copy()
    
    def print_summary(self):
        """측정 결과 요약 출력"""
        if not self.enabled or not self.timings:
            return
        
        print("\n=== 성능 측정 요약 ===")
        total = sum(self.timings.values())
        for label, elapsed in self.timings.items():
            percentage = (elapsed / total * 100) if total > 0 else 0
            print(f"{label}: {elapsed:.4f}초 ({percentage:.1f}%)")
        print(f"총 소요 시간: {total:.4f}초")
        print("=" * 30)
    
    def reset(self):
        """측정 결과 초기화"""
        self.timings.clear()
        self._start_time = None
        self._current_label = None

