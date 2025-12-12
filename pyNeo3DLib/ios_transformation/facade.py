"""
IOS Transformation 파사드 모듈

복잡한 IOS 변환 로직을 단순화된 인터페이스로 제공하는 파사드 클래스입니다.
"""

import numpy as np
from typing import Optional, TYPE_CHECKING

from .mesh_data_preparer import MeshDataPreparer
from .principal_axes_calculator import PrincipalAxesCalculator
from .ray_casting_service import RayCastingService
from .coordinate_system_builder import CoordinateSystemBuilder
from .transformation_calculator import TransformationCalculator

if TYPE_CHECKING:
    from pyNeo3DLib.fileLoader.mesh import Mesh


class IOSTransformationConstants:
    """IOS Transformation 관련 상수"""
    IDENTITY_MATRIX = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    LOWER_JAW_Z_OFFSET = -15


class IOSTransformationFacade:
    """
    IOS Transformation 파사드 클래스
    
    IOS 메시를 Smile Arch로 정렬하는 변환 행렬 계산을 위한
    단순화된 인터페이스를 제공합니다.
    
    내부적으로 다음 서비스들을 사용합니다:
    - MeshDataPreparer: 메시 데이터 준비
    - PrincipalAxesCalculator: 주축 계산
    - RayCastingService: 레이캐스팅
    - CoordinateSystemBuilder: 좌표계 구축
    - TransformationCalculator: 변환 행렬 계산
    
    Example:
        >>> facade = IOSTransformationFacade()
        >>> transformation = facade.compute_transformation(
        ...     ios_mesh=upper_mesh,
        ...     smile_arch_mesh=smile_arch,
        ...     ios_laminate_result=laminate_matrix,
        ...     is_upper=True
        ... )
    """
    
    def __init__(self):
        """파사드 초기화 및 내부 서비스 생성"""
        self._mesh_preparer = MeshDataPreparer()
        self._axes_calculator = PrincipalAxesCalculator()
        self._ray_casting_service = RayCastingService()
        self._coord_builder = CoordinateSystemBuilder()
        self._transform_calculator = TransformationCalculator()
    
    def compute_transformation(
        self,
        ios_mesh: "Mesh",
        smile_arch_mesh: "Mesh",
        ios_laminate_result: np.ndarray,
        is_upper: bool
    ) -> Optional[np.ndarray]:
        """
        IOS 메시를 Smile Arch로 정렬하는 변환 행렬을 계산합니다.
        
        처리 과정:
        1. 메시 데이터 준비
        2. PCA를 통한 주축 계산
        3. Z축 벡터 계산
        4. 레이캐스팅을 통한 단일 교차점 방향 찾기
        5. 좌표계 구축 및 변환 행렬 계산
        
        Args:
            ios_mesh: IOS 메시 객체 (Upper 또는 Lower)
            smile_arch_mesh: Smile Arch 메시 객체
            ios_laminate_result: IOS Laminate 변환 행렬
            is_upper: True면 Upper, False면 Lower
            
        Returns:
            4x4 변환 행렬, 실패 시 None
        """
        try:
            mesh_type = "Upper" if is_upper else "Lower"
            
            # 1. 메시 데이터 준비
            ios_vertices, ios_faces, smile_arch_centroid = self._mesh_preparer.prepare(
                ios_mesh, smile_arch_mesh, ios_laminate_result, mesh_type
            )
            
            # 2. 주축 계산
            principal_axes, closest_axis, closest_axis_vector, centroid = \
                self._axes_calculator.compute(ios_vertices)
            
            # 3. Z축 벡터 계산
            z_axis_vector = self._axes_calculator.compute_z_axis_vector(
                ios_mesh, closest_axis_vector
            )
            
            if z_axis_vector is None:
                return None
            
            # 4. 단일 교차점 방향 찾기
            single_intersection_direction = self._ray_casting_service.find_single_intersection_direction(
                mesh_vertices=ios_vertices,
                mesh_faces=ios_faces,
                principal_axes=principal_axes,
                centroid=centroid,
                closest_axis_idx=closest_axis
            )
            
            if single_intersection_direction is None:
                print("[WARNING] Could not find single intersection direction.")
                return None
            
            # 5. 최종 변환 행렬 계산
            combined_transformation_matrix = self._compute_final_transformation(
                single_intersection_direction=single_intersection_direction,
                z_axis_vector=z_axis_vector,
                centroid=centroid,
                ios_vertices=ios_vertices,
                smile_arch_centroid=smile_arch_centroid,
                is_upper=is_upper
            )
            
            # Lower jaw 오프셋 적용
            if not is_upper:
                lower_translation_matrix = np.eye(4)
                lower_translation_matrix[:3, 3] = np.array([0, 0, IOSTransformationConstants.LOWER_JAW_Z_OFFSET])
                combined_transformation_matrix = np.matmul(lower_translation_matrix, combined_transformation_matrix)
            
            return combined_transformation_matrix
            
        except Exception as e:
            mesh_type = "Upper" if is_upper else "Lower"
            print(f"[ERROR] IOS-SmileArch {mesh_type} transformation calculation error: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def compute_transformation_safe(
        self,
        ios_mesh: "Mesh",
        smile_arch_mesh: "Mesh",
        ios_laminate_result: np.ndarray,
        is_upper: bool
    ) -> np.ndarray:
        """
        안전하게 변환 행렬을 계산합니다.
        
        실패 시 IDENTITY_MATRIX를 반환하여 프로그램이 중단되지 않도록 합니다.
        
        Args:
            ios_mesh: IOS 메시 객체
            smile_arch_mesh: Smile Arch 메시 객체
            ios_laminate_result: IOS Laminate 변환 행렬
            is_upper: True면 Upper, False면 Lower
            
        Returns:
            4x4 변환 행렬 (실패 시 단위 행렬)
        """
        result = self.compute_transformation(
            ios_mesh=ios_mesh,
            smile_arch_mesh=smile_arch_mesh,
            ios_laminate_result=ios_laminate_result,
            is_upper=is_upper
        )
        
        if result is not None:
            return result
        else:
            return np.array(IOSTransformationConstants.IDENTITY_MATRIX)
    
    def apply_transformation(
        self,
        mesh: "Mesh",
        transformation_matrix: np.ndarray
    ) -> "Mesh":
        """
        메시에 변환 행렬을 적용합니다.
        
        Args:
            mesh: 변환할 Mesh 객체
            transformation_matrix: 4x4 동차 변환 행렬
            
        Returns:
            변환된 Mesh 객체
        """
        return self._transform_calculator.apply_transformation_to_mesh(
            mesh, transformation_matrix
        )
    
    def _compute_final_transformation(
        self,
        single_intersection_direction: np.ndarray,
        z_axis_vector: np.ndarray,
        centroid: np.ndarray,
        ios_vertices: np.ndarray,
        smile_arch_centroid: np.ndarray,
        is_upper: bool
    ) -> np.ndarray:
        """최종 변환 행렬을 계산합니다."""
        # 좌표계 구축
        x_axis_vector, y_axis_vector, z_axis_vector = self._coord_builder.build(
            single_intersection_direction=single_intersection_direction,
            closest_axis_vector=z_axis_vector
        )
        
        # 회전 행렬 계산
        rotation_matrix = self._transform_calculator.compute_rotation_matrix_to_standard_jaw(
            x_axis=x_axis_vector,
            y_axis=y_axis_vector,
            z_axis=z_axis_vector,
            is_upper=is_upper
        )
        
        # 회전 + 이동을 결합한 변환 행렬 계산
        combined_transformation_matrix = self._transform_calculator.compute_combined_transformation(
            rotation_matrix=rotation_matrix,
            source_centroid=centroid,
            target_centroid=smile_arch_centroid
        )
        
        return combined_transformation_matrix
