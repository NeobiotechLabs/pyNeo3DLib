"""
페이스 정합(Face Registration) 모듈

이 패키지는 페이스 스캔 데이터의 정합 및 처리를 위한 클래스들을 제공합니다.

주요 모듈:
- faceSmileGuideAligner: 페이스-라미네이트 정합 오케스트레이션
- facesRegistration: 여러 페이스 스캔 정합
- facePhotoAlign: 페이스 사진 정합
"""

# 메인 정합 클래스들
from .faceAlignModule.face_lip_extractor import FaceLipExtractor
from .faceSmileGuideAligner import *
from .facePhotoAlign import *
from .facesRegistration import *

__all__ = [
    # 유틸리티
    "FaceLipExtractor",
    # 메인 정합 클래스
    "faceSmileGuideAligner", 
    "facePhotoAlign", 
    "facesRegistration", 
]
