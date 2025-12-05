"""
Global Fit 모듈
메쉬 변환 유틸리티를 제공합니다.
"""

__version__ = "1.0.0"
__author__ = "3D Comparing Team"

# 주요 클래스들을 최상위 레벨에서 import 가능하도록
from .constants import MeshConversionConfig
from .mesh_converter import MeshConverter

__all__ = [
    # 상수 설정 클래스
    "MeshConversionConfig",
    # 메쉬 변환
    "MeshConverter",
]

