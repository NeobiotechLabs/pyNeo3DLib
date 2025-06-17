"""
스마일 아치 랜드마크 감지 라이브러리

이 라이브러리는 STL 파일을 처리하여 스마일 아치 랜드마크를 감지하는 기능을 제공합니다.
"""

__version__ = '0.1.0'

# 필요한 클래스만 라이브러리에서 직접 가져오기
from .landmark.landmark_detector import SmileArchOuterlineDetector

# 외부에 노출할 이름 목록을 제한
__all__ = ['SmileArchOuterlineDetector']