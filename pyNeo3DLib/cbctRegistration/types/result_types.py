"""
정합 결과 데이터 타입 정의

파이프라인 각 단계의 결과를 담는 데이터 클래스를 정의합니다.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np
import open3d as o3d


@dataclass
class CBCTExtractionResult:
    """CBCT 표면 추출 결과"""
    surface_cropped: o3d.geometry.PointCloud  # Z축 Crop된 표면
    surface_full: o3d.geometry.PointCloud  # 전체 표면
    nose_center: np.ndarray  # 추정된 코 중심 (RAI 좌표계)
    nose_region: o3d.geometry.PointCloud  # 코 주변 영역
    
    def __post_init__(self):
        self.nose_center = np.array(self.nose_center)


@dataclass 
class CoordinateTransformResult:
    """좌표계 변환 결과"""
    pcd_standard: o3d.geometry.PointCloud  # 표준 좌표계 포인트 클라우드
    transform_matrix: np.ndarray  # RAI → 표준 좌표계 변환 행렬
    nose_center_standard: np.ndarray  # 표준 좌표계에서의 코 중심 (원점)
    
    def __post_init__(self):
        self.transform_matrix = np.array(self.transform_matrix)
        self.nose_center_standard = np.array(self.nose_center_standard)


@dataclass
class FaceScanProcessResult:
    """FaceScan 처리 결과"""
    mesh: o3d.geometry.TriangleMesh  # 원본 메쉬
    pcd: o3d.geometry.PointCloud  # 샘플링된 포인트 클라우드
    pcd_filtered: o3d.geometry.PointCloud  # 영역 필터링된 포인트 클라우드
    nose_point: np.ndarray  # 코 끝 포인트 (Y 최상위)
    facescan_transform: np.ndarray  # FaceScan → SmileArch 변환 행렬
    
    def __post_init__(self):
        self.nose_point = np.array(self.nose_point)
        self.facescan_transform = np.array(self.facescan_transform)


@dataclass
class AlignmentStepResult:
    """정렬 단계 결과"""
    aligned_pcd: o3d.geometry.PointCloud  # 정렬된 포인트 클라우드
    transform_matrix: np.ndarray  # 적용된 변환 행렬
    
    def __post_init__(self):
        self.transform_matrix = np.array(self.transform_matrix)


@dataclass
class ICPAlignmentResult:
    """ICP 정합 결과"""
    aligned_pcd: o3d.geometry.PointCloud  # 정렬된 포인트 클라우드
    transform_matrix: np.ndarray  # 최적 변환 행렬 (Z이동 + ICP)
    fitness: float  # 정합 품질 (0~1)
    inlier_rmse: float  # RMSE 값 (mm)
    method: str  # 사용된 방법
    best_z_offset: int  # 최적 Z 오프셋
    
    def __post_init__(self):
        self.transform_matrix = np.array(self.transform_matrix)


@dataclass
class RefinementResult:
    """SDF 기반 회전 정제 결과"""
    transform_matrix: np.ndarray  # 정제 변환 행렬
    best_angle: float  # 최적 회전 각도 (도)
    best_rmse: float  # 최소 RMSE 값
    
    def __post_init__(self):
        self.transform_matrix = np.array(self.transform_matrix)


@dataclass
class PipelineResult:
    """전체 파이프라인 결과"""
    # 최종 결과
    final_transform: np.ndarray  # 최종 변환 행렬 (CBCT RAI → FaceScan 좌표계)
    
    # CBCT 관련
    cbct_extraction: Optional[CBCTExtractionResult] = None
    cbct_coordinate_transform: Optional[CoordinateTransformResult] = None
    cbct_full_final: Optional[o3d.geometry.PointCloud] = None
    
    # FaceScan 관련
    facescan_process: Optional[FaceScanProcessResult] = None
    
    # 정렬 결과
    initial_alignment: Optional[AlignmentStepResult] = None
    icp_alignment: Optional[ICPAlignmentResult] = None
    refinement: Optional[RefinementResult] = None
    
    # 중간 변환 행렬들
    transforms: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.final_transform = np.array(self.final_transform)
    
    def get_accumulated_transform(self) -> np.ndarray:
        """누적 변환 행렬 계산"""
        accumulated = np.eye(4)
        
        if self.cbct_coordinate_transform is not None:
            accumulated = self.cbct_coordinate_transform.transform_matrix @ accumulated
        
        if self.initial_alignment is not None:
            accumulated = self.initial_alignment.transform_matrix @ accumulated
        
        if self.icp_alignment is not None:
            accumulated = self.icp_alignment.transform_matrix @ accumulated
        
        return accumulated
    
    def summary(self) -> str:
        """결과 요약 문자열 생성"""
        lines = [
            "=" * 60,
            "파이프라인 결과 요약",
            "=" * 60,
        ]
        
        if self.cbct_extraction is not None:
            lines.append(f"CBCT 표면 포인트: {len(self.cbct_extraction.surface_full.points):,}")
            lines.append(f"코 중심 (RAI): {self.cbct_extraction.nose_center}")
        
        if self.icp_alignment is not None:
            lines.append(f"ICP Fitness: {self.icp_alignment.fitness:.6f}")
            lines.append(f"ICP RMSE: {self.icp_alignment.inlier_rmse:.6f} mm")
            lines.append(f"ICP Method: {self.icp_alignment.method}")
            lines.append(f"최적 Z 오프셋: {self.icp_alignment.best_z_offset} mm")
        
        lines.append("-" * 60)
        lines.append("최종 변환 행렬:")
        lines.append(str(self.final_transform))
        lines.append("=" * 60)
        
        return "\n".join(lines)


__all__ = [
    "CBCTExtractionResult",
    "CoordinateTransformResult",
    "FaceScanProcessResult",
    "AlignmentStepResult",
    "ICPAlignmentResult",
    "RefinementResult",
    "PipelineResult",
]


