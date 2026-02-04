"""
CBCT-FaceScan 정합 모듈

리팩토링된 모듈 구조:
- core/: 핵심 파이프라인 로직
  - alignment_pipeline.py: 메인 파이프라인 (Orchestrator)
  - alignment_executor.py: 정합 실행 (초기 정렬, ICP)

- processing/: 데이터 처리
  - cbct_processor.py: CBCT 데이터 처리
  - geometry_processor.py: 메쉬/포인트 클라우드 처리
  - depth_map/: CBCT depth map 추출

- registration/: 정합 및 변환
  - icp_registration.py: ICP 정합
  - coordinate_transformer.py: 좌표계 변환
  - transform_manager.py: 변환 행렬 관리
  - surface_rotation_optimizer.py: 표면 회전 최적화

- visualization/: 시각화
  - alignment_visualizer.py: 정합 결과 시각화

- types/: 타입 정의
  - result_types.py: 결과 데이터 타입

- utils/: 유틸리티
  - common.py: 공통 유틸리티 함수

- config.py: 설정 관리

사용 예제:
```python
from pyNeo3DLib.cbctRegistration import CBCTFaceScanAlignmentPipeline, AlignmentConfig

# 기본 설정으로 파이프라인 실행
pipeline = CBCTFaceScanAlignmentPipeline()
result = pipeline.run(
    dicom_folder="path/to/dicom",
    facescan_path="path/to/facescan.ply",
    smile_arch_path="path/to/smilearch.stl",
    visualize=True,
)

# 결과 확인
print(result.summary())
print(f"Final Transform:\\n{result.final_transform}")
```
"""

# 메인 파이프라인
from .core import CBCTFaceScanAlignmentPipeline, AlignmentExecutor

# 설정
from .config import (
    AlignmentConfig,
    CBCTExtractionConfig,
    NoseEstimationConfig,
    DepthMapConfig,
    MeshSamplingConfig,
    ICPConfig,
    VisualizationConfig,
    CoordinateTransformConfig,
)

# 결과 타입
from .types import (
    PipelineResult,
    CBCTExtractionResult,
    CoordinateTransformResult,
    FaceScanProcessResult,
    AlignmentStepResult,
    ICPAlignmentResult,
    RefinementResult,
)

# 프로세서
from .processing import CBCTProcessor, GeometryProcessor
from .registration import (
    ICPRegistration,
    ICPResult,
    CoordinateTransformer,
    TransformManager,
)
from .visualization import AlignmentVisualizer

# 유틸리티
from .utils import (
    np_to_pcd,
    pcd_to_np,
    apply_transform,
    apply_transform_to_points,
    compute_translation_matrix,
    compute_center_alignment_transform,
    transform_point_homogeneous,
)

__all__ = [
    # 메인 파이프라인
    "CBCTFaceScanAlignmentPipeline",
    "AlignmentExecutor",
    
    # 설정
    "AlignmentConfig",
    "CBCTExtractionConfig",
    "NoseEstimationConfig",
    "DepthMapConfig",
    "MeshSamplingConfig",
    "ICPConfig",
    "VisualizationConfig",
    "CoordinateTransformConfig",
    
    # 결과 타입
    "PipelineResult",
    "CBCTExtractionResult",
    "CoordinateTransformResult",
    "FaceScanProcessResult",
    "AlignmentStepResult",
    "ICPAlignmentResult",
    "RefinementResult",
    
    # 프로세서
    "CBCTProcessor",
    "GeometryProcessor",  
    "AlignmentVisualizer",
    "ICPRegistration",
    "ICPResult",
    "CoordinateTransformer",
    "TransformManager",
    
    # 유틸리티
    "np_to_pcd",
    "pcd_to_np",
    "apply_transform",
    "apply_transform_to_points",
    "compute_translation_matrix",
    "compute_center_alignment_transform",
    "transform_point_homogeneous",
]

__version__ = "2.0.0"
