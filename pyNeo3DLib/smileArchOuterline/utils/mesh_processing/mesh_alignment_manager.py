"""
메쉬 정렬 관련 기능을 담당하는 클래스
"""

import numpy as np
import pyvista as pv
from typing import Tuple, Optional, List
from pathlib import Path
from pyNeo3DLib.smileArchOuterline.utils.common.constants import AnalysisConstants
from pyNeo3DLib.smileArchOuterline.utils.ray_casting.ray_caster import RayCaster
from pyNeo3DLib.smileArchOuterline.utils.common.vector_utils import VectorUtils
from .mesh_loader import MeshLoader
from .pca_analyzer import PCAAnalyzer
from .mesh_axis_determiner import MeshAxisDeterminer
from .mesh_transformer import MeshTransformer


class MeshAlignmentManager:
    """메쉬 정렬을 관리하는 클래스"""
    
    # 상수 정의
    EPSILON = 1e-10
    NEAR_PARALLEL_DOT_PRODUCT_THRESHOLD = 0.9
    STANDARD_BASIS = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    
    def __init__(self):
        self.ray_caster = RayCaster()
        self.x_axis_intersection_count = AnalysisConstants.X_AXIS_INTERSECTION_COUNT
        self.y_axis_intersection_count = AnalysisConstants.Y_AXIS_INTERSECTION_COUNT
        self.z_axis_intersection_count = AnalysisConstants.Z_AXIS_INTERSECTION_COUNT
        self.mesh: Optional[pv.PolyData] = None
        self.vertices: Optional[np.ndarray] = None
        self.center: Optional[np.ndarray] = None
        self.diagonal_length: Optional[float] = None
        self.pca_eigenvalues: Optional[np.ndarray] = None
        self.pca_eigenvectors: Optional[np.ndarray] = None
        self.min_variance_axis: Optional[np.ndarray] = None
        self._pca_computed = False
        self.mesh_loader = MeshLoader()
        self.pca_analyzer = PCAAnalyzer()
        self.axis_determiner = MeshAxisDeterminer()
        self.mesh_transformer = MeshTransformer()

    def load_mesh(self, mesh_path: str):
        self.mesh_loader.load_mesh(mesh_path)
        self.mesh = self.mesh_loader.mesh
        self.vertices = self.mesh_loader.vertices
        self.center = self.mesh_loader.center
        self.diagonal_length = self.mesh_loader.diagonal_length

    def compute_principal_axes(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.mesh is None:
            raise ValueError("메쉬가 로드되지 않았습니다. load_mesh를 먼저 호출하세요.")
        self.pca_eigenvalues, self.pca_eigenvectors, self.min_variance_axis = \
            self.pca_analyzer.compute_principal_axes(self.vertices, self.center)
        self._pca_computed = True
        return self.pca_eigenvalues, self.pca_eigenvectors, self.min_variance_axis

    def determine_alignment_axes(
        self,
        input_mesh: pv.PolyData, 
        center: np.ndarray, 
        principal_evecs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.axis_determiner.determine_alignment_axes(input_mesh, center, principal_evecs)

    def align_mesh_to_global_coordinates(
        self,
        input_mesh: pv.PolyData, 
        evec_x: np.ndarray, 
        evec_y: np.ndarray, 
        evec_z: np.ndarray
    ) -> pv.PolyData:
        return self.mesh_transformer.align_mesh_to_global_coordinates(input_mesh, evec_x, evec_y, evec_z)

    def filter_mesh_by_z_threshold(
        self, 
        aligned_mesh: pv.PolyData, 
        filtered_points: np.ndarray
    ) -> pv.PolyData:
        if len(filtered_points) == 0:
            raise ValueError("filtered_points가 비어있습니다.")
        
        if filtered_points.shape[1] != 3:
            raise ValueError(f"filtered_points는 (N, 3) 형태여야 합니다. 현재 shape: {filtered_points.shape}")
        
        z_min_point = np.min(filtered_points[:, 2])
        mask = aligned_mesh.points[:, 2] > z_min_point
        filtered_aligned_mesh = aligned_mesh.extract_points(mask)

        # 가장 큰 덩어리만 추출
        largest_component = filtered_aligned_mesh.extract_largest()

        # 중복된 면이나 점이 있는 경우 제거
        filtered_largest_component = largest_component.clean()
        
        return filtered_largest_component

