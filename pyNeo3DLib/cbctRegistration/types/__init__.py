"""
타입 정의 모듈

이 모듈은 파이프라인에서 사용되는 데이터 타입과 결과 타입을 정의합니다.
"""

from .result_types import (
    PipelineResult,
    CBCTExtractionResult,
    CoordinateTransformResult,
    FaceScanProcessResult,
    AlignmentStepResult,
    ICPAlignmentResult,
    RefinementResult,
)

__all__ = [
    "PipelineResult",
    "CBCTExtractionResult",
    "CoordinateTransformResult",
    "FaceScanProcessResult",
    "AlignmentStepResult",
    "ICPAlignmentResult",
    "RefinementResult",
]


