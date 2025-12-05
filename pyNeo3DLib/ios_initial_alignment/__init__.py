"""
ios_initial_alignment - IOS 스캔 데이터 초기 정합 모듈

IOS(Intraoral Scan) 스캔 데이터의 초기 정합을 위한 모듈입니다.

하위 모듈:
- initial_alignment: 초기 정합 알고리즘
- global_fit: 메시 변환 유틸리티
- utils: 메시 I/O 유틸리티
"""

# 주요 클래스 및 함수 export
from .initial_alignment import (
    MeshAligner,
    align_3d_meshes,
    align_meshes_direct,
)

__all__ = [
    'MeshAligner',
    'align_3d_meshes',
    'align_meshes_direct',
]

