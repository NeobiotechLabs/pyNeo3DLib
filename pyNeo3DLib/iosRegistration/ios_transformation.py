"""
IOS Transformation Calculator Module

IOS 메시를 Smile Arch로 정렬하는 변환 행렬 계산 관련 클래스와 유틸리티 함수들을 제공합니다.
"""

import numpy as np
import pyvista as pv
import trimesh
from scipy.linalg import eigh
from typing import Any, Optional, Tuple

from pyNeo3DLib.smileArchOuterline.utils.ray_caster import RayCaster
from pyNeo3DLib.fileLoader.mesh import Mesh


class IOSTransformationConstants:
    """IOS Transformation에 사용되는 상수들"""
    IDENTITY_MATRIX = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    LOWER_JAW_Z_OFFSET = -15


class IOSTransformationCalculator:
    """
    IOS 메시를 Smile Arch로 정렬하는 변환 행렬을 계산하는 클래스.
    
    주요 기능:
    - PCA를 통한 주축 계산
    - 레이캐스팅을 통한 방향 벡터 찾기
    - 좌표계 구축 (x, y, z 축)
    - 회전 + 이동 변환 행렬 계산
    """
    
    def __init__(self, progress_reporter=None):
        """
        Args:
            progress_reporter: 진행 상황 보고 객체 (선택사항)
        """
        self.progress_reporter = progress_reporter
    
    async def safe_compute_ios_transformation(
        self,
        progress_name: str,
        ios_mesh: "Mesh",
        smile_arch_mesh: "Mesh",
        ios_laminate_result: np.ndarray,
        transformation_name: str,
        is_upper: bool
    ) -> np.ndarray:
        """
        IOS transformation 계산을 안전하게 수행하는 메서드.
        실패 시 IDENTITY_MATRIX를 반환하여 프로그램이 죽지 않도록 합니다.
        
        Args:
            progress_name: 진행 상황 보고용 이름
            ios_mesh: IOS 메시 (upper 또는 lower)
            smile_arch_mesh: smile arch 메시
            ios_laminate_result: IOS laminate registration 결과
            transformation_name: transformation 행렬 이름 (로깅용)
            is_upper: True면 Upper, False면 Lower
            
        Returns:
            계산된 transformation 행렬 또는 IDENTITY_MATRIX (실패 시)
        """
        try:
            if self.progress_reporter:
                await self.progress_reporter.report_progress(progress_name)
            
            transformation_matrix = self.compute_ios_to_smilearch_transformation(
                ios_mesh=ios_mesh,
                smile_arch_mesh=smile_arch_mesh,
                ios_laminate_result=ios_laminate_result,
                is_upper=is_upper
            )
            
            if transformation_matrix is not None:
                print(f'[SUCCESS] {transformation_name}: {transformation_matrix}')
                print(f'[SUCCESS] {progress_name} completed')
                return transformation_matrix
            else:
                print(f'[FAILED] {transformation_name} calculation failed')
                return np.array(IOSTransformationConstants.IDENTITY_MATRIX)
                
        except Exception as e:
            print(f'[FAILED] {progress_name} failed: {str(e)}')
            return np.array(IOSTransformationConstants.IDENTITY_MATRIX)
    
    def compute_ios_to_smilearch_transformation(
        self,
        ios_mesh: "Mesh",
        smile_arch_mesh: "Mesh",
        ios_laminate_result: np.ndarray,
        is_upper: bool
    ) -> Optional[np.ndarray]:
        """
        IOS 메시를 Smile Arch로 정렬하는 변환 행렬을 계산합니다.
        
        처리 과정:
        1. PCA를 통한 주축 계산
        2. 레이캐스팅을 통한 방향 벡터 찾기
        3. 좌표계 구축 (x, y, z 축)
        4. 회전 + 이동 변환 행렬 계산
        
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
            ios_vertices, ios_faces, smile_arch_centroid = self._prepare_mesh_data(
                ios_mesh, smile_arch_mesh, ios_laminate_result, mesh_type
            )
            
            # 2. 주축 계산
            principal_axes, closest_axis, closest_axis_vector, centroid = self._compute_principal_axes(
                ios_vertices
            )
            
            # 3. Z축 벡터 계산
            z_axis_vector = self._compute_z_axis_vector(
                ios_mesh, closest_axis_vector
            )
            
            if z_axis_vector is None:
                return None
            
            # 4. 단일 교차점 방향 찾기
            single_intersection_direction = self._find_single_intersection_direction(
                mesh_vertices=ios_vertices,
                mesh_faces=ios_faces,
                principal_axes=principal_axes,
                centroid=centroid,
                closest_axis_idx=closest_axis
            )
            
            if single_intersection_direction is None:
                print("[WARNING] Failed to find single intersection direction")
                return None
            
            # 5. 좌표계 구축 및 변환 행렬 계산
            combined_transformation_matrix = self._compute_final_transformation(
                single_intersection_direction=single_intersection_direction,
                z_axis_vector=z_axis_vector,
                centroid=centroid,
                ios_vertices=ios_vertices,
                smile_arch_centroid=smile_arch_centroid,
                is_upper=is_upper
            )

            if not is_upper:
                lower_translation_matrix = np.eye(4)
                lower_translation_matrix[:3, 3] = np.array([0, 0, IOSTransformationConstants.LOWER_JAW_Z_OFFSET])
                combined_transformation_matrix = np.matmul(lower_translation_matrix, combined_transformation_matrix)
                
            return combined_transformation_matrix
            
        except Exception as e:
            print(f"[ERROR] IOS-SmileArch {mesh_type} transformation calculation error: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _prepare_mesh_data(
        self,
        ios_mesh: "Mesh",
        smile_arch_mesh: "Mesh",
        ios_laminate_result: np.ndarray,
        mesh_type: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """메시 데이터를 준비합니다."""
        ios_vertices = ios_mesh.vertices
        ios_faces = ios_mesh.faces
        smile_arch_vertices = smile_arch_mesh.vertices
        
        # Smile Arch 변환 적용
        smile_arch_vertices = np.dot(
            smile_arch_vertices,
            ios_laminate_result[:3, :3].T
        ) + ios_laminate_result[:3, 3]
        
        smile_arch_centroid = np.mean(smile_arch_vertices, axis=0)
        
        print(f"[INFO] IOS {mesh_type} mesh: {ios_vertices.shape[0]} vertices")
        print(f"[INFO] Smile Arch mesh: {smile_arch_vertices.shape[0]} vertices")
        
        return ios_vertices, ios_faces, smile_arch_centroid
    
    def _compute_principal_axes(
        self, 
        vertices: np.ndarray
    ) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray]:
        """주축을 계산합니다."""
        # PCA를 통한 주축 계산
        principal_axes, _, centroid = compute_principal_axes_from_vertices(
            vertices, 
            verbose=True
        )
        
        # 분산이 가장 작은 주축 계산
        minimum_variance_axis, _, _ = compute_minimum_variance_axis_from_vertices(
            vertices,
            verbose=True
        )
        print(f"[INFO] Minimum variance axis: {minimum_variance_axis}")
        
        # principal_axes에서 minimum_variance_axis와 가장 가까운 주축 찾기
        closest_axis = np.argmax(np.abs(np.dot(principal_axes, minimum_variance_axis)))
        closest_axis_vector = principal_axes[closest_axis]
        print(f"[INFO] Closest axis index: {closest_axis}")
        print(f"[INFO] Closest axis vector: {closest_axis_vector}")
        
        return principal_axes, closest_axis, closest_axis_vector, centroid
    
    def _compute_z_axis_vector(
        self,
        ios_mesh: "Mesh",
        closest_axis_vector: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Z축 벡터를 계산합니다.
        
        IOS 메시의 평균 법선 벡터와 주축 벡터의 내적을 계산하여
        방향을 결정합니다. 두 벡터가 같은 방향이면 주축 벡터를,
        반대 방향이면 주축 벡터의 반대 방향을 반환합니다.
        
        Args:
            ios_mesh: IOS 메시 객체 (상악 또는 하악)
            closest_axis_vector: PCA로 계산된 주축 벡터
            
        Returns:
            방향이 결정된 Z축 벡터
        """
        if ios_mesh.normals is None:
            ios_mesh._compute_normals()
        
        ios_normals = np.asarray(ios_mesh.normals)
        ios_normals_mean = np.mean(ios_normals, axis=0)
        print(f"[INFO] IOS mesh normals mean: {ios_normals_mean}")
        
        # 내적으로 방향 확인
        inner_product = np.dot(closest_axis_vector, ios_normals_mean)
        if inner_product > 0:
            print("[INFO] Same direction")
            return closest_axis_vector
        else:
            print("[INFO] Opposite direction")
            return -closest_axis_vector

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
        x_axis_vector, y_axis_vector, z_axis_vector = self._build_coordinate_system(
            single_intersection_direction=single_intersection_direction,
            closest_axis_vector=z_axis_vector
        )
        
        # 회전 행렬 계산 (표준 좌표계로 변환)
        rotation_matrix = self._compute_rotation_matrix_to_standard_jaw(
            x_axis=x_axis_vector,
            y_axis=y_axis_vector,
            z_axis=z_axis_vector,
            is_upper=is_upper
        )
        
        # 회전 + 이동을 결합한 변환 행렬 계산
        combined_transformation_matrix = self._compute_combined_transformation(
            rotation_matrix=rotation_matrix,
            centroid=centroid,
            source_vertices=ios_vertices,
            target_centroid=smile_arch_centroid
        )
        
        return combined_transformation_matrix
    
    def _get_ray_casting_vector_to_centroid(
        self,
        mesh_vertices: np.ndarray,
        mesh_faces: np.ndarray,
        centroid: np.ndarray,
        axis_vector: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        도심점에서 axis_vector 방향으로 양방향 레이캐스팅하여 메쉬 표면 포인트를 검출하고,
        검출한 포인트에서 도심점 방향으로 가는 벡터를 반환합니다.
        
        Args:
            mesh_vertices: 메시의 정점 배열
            mesh_faces: 메시의 면 정보
            centroid: 메시의 무게중심
            axis_vector: 레이캐스팅 방향 벡터
            
        Returns:
            교차점에서 도심점으로 가는 벡터, 교차점이 없으면 None
        """
        # pyvista 메시 생성
        pv_mesh = self._create_pyvista_mesh(mesh_vertices, mesh_faces)
        ray_caster = RayCaster()
        
        # 양방향 레이캐스팅으로 표면 포인트 찾기
        surface_point = self._find_surface_point_by_raycasting(
            pv_mesh, ray_caster, centroid, axis_vector
        )
        
        if surface_point is None:
            return None
        
        # 교차점에서 도심점으로 가는 벡터
        return centroid - surface_point
    
    def _create_pyvista_mesh(
        self, 
        vertices: np.ndarray, 
        faces: np.ndarray
    ) -> Any:
        """PyVista 메시를 생성합니다."""
        faces_with_count = np.column_stack([np.full(len(faces), 3), faces])
        return pv.PolyData(vertices, faces_with_count)
    
    def _find_surface_point_by_raycasting(
        self,
        pv_mesh: Any,
        ray_caster: Any,
        origin: np.ndarray,
        direction: np.ndarray
    ) -> Optional[np.ndarray]:
        """레이캐스팅으로 표면 포인트를 찾습니다."""
        # plus 방향 레이캐스팅
        plus_intersections = ray_caster.ray_casting(
            pv_mesh, origin.reshape(1, 3), direction.reshape(1, 3)
        )
        
        if len(plus_intersections) > 0:
            return plus_intersections[0]
        
        # minus 방향 레이캐스팅
        minus_intersections = ray_caster.ray_casting(
            pv_mesh, origin.reshape(1, 3), (-direction).reshape(1, 3)
        )
        
        if len(minus_intersections) > 0:
            return minus_intersections[0]
        
        return None

    def _find_single_intersection_direction(
        self, 
        mesh_vertices: np.ndarray,
        mesh_faces: np.ndarray,
        principal_axes: np.ndarray, 
        centroid: np.ndarray, 
        closest_axis_idx: int
    ) -> Optional[np.ndarray]:
        """
        레이캐스팅을 통해 단일 교차점을 가진 축 방향을 찾습니다.
        
        Args:
            mesh_vertices: 메시의 정점 배열
            mesh_faces: 메시의 면 정보
            principal_axes: 주성분 분석으로 얻은 주축들 (3x3)
            centroid: 메시의 무게중심
            closest_axis_idx: 제외할 축의 인덱스
            
        Returns:
            단일 교차점을 가진 축 방향 벡터, 없으면 None
        """
        # PyVista 메시 및 RayCaster 준비
        pv_mesh = self._create_pyvista_mesh(mesh_vertices, mesh_faces)
        ray_caster = RayCaster()
        
        # 제외할 축을 제외한 나머지 축들
        remaining_axes_indices = [i for i in range(3) if i != closest_axis_idx]
        
        # 나머지 두 축에 대해 레이캐스팅 수행
        for axis_idx in remaining_axes_indices:
            axis_vector = principal_axes[axis_idx]
            
            # 단일 교차점 확인
            unit_vector = self._check_single_intersection(
                pv_mesh, ray_caster, centroid, axis_vector, axis_idx
            )
            
            if unit_vector is not None:
                return unit_vector
        
        print("[WARNING] Could not find axis with single intersection")
        return None
    
    def _check_single_intersection(
        self,
        pv_mesh: Any,
        ray_caster: Any,
        centroid: np.ndarray,
        axis_vector: np.ndarray,
        axis_idx: int
    ) -> Optional[np.ndarray]:
        """특정 축에 대해 단일 교차점 여부를 확인합니다."""
        # +방향 레이캐스팅
        plus_intersections = ray_caster.ray_casting(
            pv_mesh, centroid.reshape(1, 3), axis_vector.reshape(1, 3)
        )
        plus_has_intersection = len(plus_intersections) > 0
        
        # -방향 레이캐스팅
        minus_intersections = ray_caster.ray_casting(
            pv_mesh, centroid.reshape(1, 3), (-axis_vector).reshape(1, 3)
        )
        minus_has_intersection = len(minus_intersections) > 0
        
        # 교차점 개수 계산
        intersection_count = int(plus_has_intersection) + int(minus_has_intersection)
        print(f"  Axis {axis_idx} raycasting result: {intersection_count} initial intersection(s)")
        
        # 최초 교차점이 정확히 1개인 경우
        if intersection_count == 1:
            intersection_point = self._get_closest_intersection_point(
                plus_intersections if plus_has_intersection else minus_intersections,
                centroid
            )
            
            # 도심점에서 교차점 방향으로 나가는 단위벡터 계산
            direction_vector = intersection_point - centroid
            unit_vector = direction_vector / np.linalg.norm(direction_vector)
            
            print(f"[INFO] Found axis with single intersection: axis {axis_idx}")
            print(f"   Intersection point: {intersection_point}")
            print(f"   Unit vector from centroid to intersection: {unit_vector}")
            
            return unit_vector
        
        return None
    
    def _get_closest_intersection_point(
        self,
        intersections: np.ndarray,
        centroid: np.ndarray
    ) -> np.ndarray:
        """교차점들 중 도심점에 가장 가까운 점을 반환합니다."""
        distances = np.linalg.norm(intersections - centroid, axis=1)
        closest_idx = np.argmin(distances)
        return intersections[closest_idx]
    
    def _build_coordinate_system(
        self, 
        single_intersection_direction: np.ndarray, 
        closest_axis_vector: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        좌표계를 구축합니다 (x, y, z 축).
        
        그람-슈미트 직교 정규화를 적용하여 완벽한 직교 정규 기저를 생성합니다.
        
        Args:
            single_intersection_direction: Y축이 될 방향 벡터
            closest_axis_vector: Z축이 될 방향 벡터
            
        Returns:
            정규화된 x_axis, y_axis, z_axis 벡터 (단위 벡터, 서로 직교)
        """
        print(f"[DEBUG] Input vectors:")
        print(f"   y_axis (original): {single_intersection_direction}, norm: {np.linalg.norm(single_intersection_direction):.6f}")
        print(f"   z_axis (original): {closest_axis_vector}, norm: {np.linalg.norm(closest_axis_vector):.6f}")
        print(f"   y dot z: {np.dot(single_intersection_direction, closest_axis_vector):.6f}")
        
        # 그람-슈미트 정규화로 직교 정규 기저 생성
        # y축을 우선으로 유지하고, z축을 조정한 후 x축을 재계산
        
        # 1. y축 정규화
        y_axis_vector = single_intersection_direction / np.linalg.norm(single_intersection_direction)
        
        # 2. z축을 y축에 직교하도록 조정 후 정규화
        z_orthogonal = closest_axis_vector - np.dot(closest_axis_vector, y_axis_vector) * y_axis_vector
        z_axis_vector = z_orthogonal / np.linalg.norm(z_orthogonal)
        
        # 3. x축을 y축과 z축에 직교하도록 외적으로 재계산
        x_axis_vector = np.cross(y_axis_vector, z_axis_vector)
        
        # 정규화된 축 벡터 검증
        print(f"[INFO] Normalized axis vectors:")
        print(f"   x_axis: {x_axis_vector}, norm: {np.linalg.norm(x_axis_vector):.10f}")
        print(f"   y_axis: {y_axis_vector}, norm: {np.linalg.norm(y_axis_vector):.10f}")
        print(f"   z_axis: {z_axis_vector}, norm: {np.linalg.norm(z_axis_vector):.10f}")
        print(f"   x dot y: {np.dot(x_axis_vector, y_axis_vector):.10f} (should be 0)")
        print(f"   y dot z: {np.dot(y_axis_vector, z_axis_vector):.10f} (should be 0)")
        print(f"   z dot x: {np.dot(z_axis_vector, x_axis_vector):.10f} (should be 0)")
        
        return x_axis_vector, y_axis_vector, z_axis_vector

    def _compute_rotation_matrix_to_standard_jaw(
        self, 
        x_axis: np.ndarray, 
        y_axis: np.ndarray, 
        z_axis: np.ndarray,
        is_upper: bool
    ) -> np.ndarray:
        """
        상악/하악에 맞는 회전 행렬을 계산합니다.
        
        Args:
            x_axis: 정규화된 x축 벡터
            y_axis: 정규화된 y축 벡터
            z_axis: 정규화된 z축 벡터
            is_upper: True면 상악(upper), False면 하악(lower)
            
        Returns:
            4x4 동차변환 행렬
        """
        if is_upper:
            target_axes = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).T
        else:
            target_axes = np.eye(3)
        
        return self._compute_rotation_matrix_to_standard(x_axis, y_axis, z_axis, target_axes)

    def _compute_rotation_matrix_to_standard(
        self, 
        x_axis: np.ndarray, 
        y_axis: np.ndarray, 
        z_axis: np.ndarray,
        target_axes: np.ndarray
    ) -> np.ndarray:
        """
        현재 좌표계를 목표 좌표계로 변환하는 회전 행렬을 계산합니다.
        
        입력 축 벡터들은 이미 정규화되고 직교하는 것으로 가정합니다.
        
        Args:
            x_axis, y_axis, z_axis: 현재 좌표계의 정규화된 축 벡터들
            target_axes: 목표 좌표계 (3x3 행렬, 각 열이 목표 축)
            
        Returns:
            4x4 동차변환 행렬
        """
        # 현재 좌표계 행렬 (각 열이 축 벡터)
        current_coordinate_system = np.column_stack([x_axis, y_axis, z_axis])
        
        # 회전 행렬 계산: R = Target @ Current^T
        # 정규 직교 행렬이므로 역행렬 = 전치 행렬
        rotation_matrix_3x3 = target_axes @ current_coordinate_system.T
        
        # 회전 행렬 검증
        det = np.linalg.det(rotation_matrix_3x3)
        is_orthogonal = np.allclose(rotation_matrix_3x3 @ rotation_matrix_3x3.T, np.eye(3), atol=1e-6)
        
        print(f"[INFO] Rotation matrix validation:")
        print(f"   Determinant: {det:.10f} (should be 1)")
        print(f"   Is orthogonal: {is_orthogonal}")
        
        if not is_orthogonal or abs(det - 1.0) > 1e-6:
            print(f"[WARNING] Rotation matrix is not valid!")
            print(f"   R @ R.T:\n{rotation_matrix_3x3 @ rotation_matrix_3x3.T}")
        
        # 4x4 동차변환 행렬로 확장
        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = rotation_matrix_3x3
        
        print(f"[INFO] rotation_matrix (4x4):\n{rotation_matrix}")
        
        return rotation_matrix

    def _compute_combined_transformation(
        self,
        rotation_matrix: np.ndarray,
        centroid: np.ndarray,
        source_vertices: np.ndarray,
        target_centroid: np.ndarray
    ) -> np.ndarray:
        """
        회전 + 이동을 결합한 단일 4x4 동차 변환 행렬을 계산합니다.
        
        변환 순서:
        1. 소스 도심점을 원점으로 이동 (T1)
        2. 회전 변환 적용 (R)
        3. target_centroid로 이동 (T2)
        
        최종 변환: T2 @ R @ T1
        
        Args:
            rotation_matrix: 4x4 동차 변환 행렬
            centroid: 회전 중심점 (메시의 무게중심)
            source_vertices: 변환할 메시의 정점 배열
            target_centroid: 목표 위치의 무게중심
            
        Returns:
            4x4 동차 변환 행렬
        """
        # 소스 도심점 계산
        source_centroid = np.mean(source_vertices, axis=0)
        
        # 1단계: 소스 도심점을 원점으로 이동하는 변환 행렬 (T1)
        T1 = np.eye(4)
        T1[:3, 3] = -source_centroid
        
        # 2단계: 회전 변환 행렬 (R)
        # rotation_matrix가 이미 4x4 형태이므로 그대로 사용
        R = rotation_matrix.copy()
        
        # 3단계: target_centroid로 이동하는 변환 행렬 (T2)
        T2 = np.eye(4)
        T2[:3, 3] = target_centroid
        
        # 4단계: 최종 변환 행렬 결합 (T2 @ R @ T1)
        combined_matrix = T2 @ R @ T1
        
        return combined_matrix

    def apply_transformation_to_mesh(
        self,
        mesh,
        transformation_matrix: np.ndarray
    ):
        """
        메시에 4x4 동차 변환 행렬을 적용합니다 (in-place).
        
        Args:
            mesh: 변환할 Mesh 객체
            transformation_matrix: 4x4 동차 변환 행렬
            
        Returns:
            변환된 Mesh 객체
        """
        # 정점을 동차 좌표로 변환 (N x 3 -> N x 4)
        vertices_homogeneous = np.hstack([
            mesh.vertices,
            np.ones((len(mesh.vertices), 1))
        ])
        
        # 변환 적용
        transformed_vertices_homogeneous = vertices_homogeneous @ transformation_matrix.T
        
        # 3D 좌표로 변환 (N x 4 -> N x 3)
        mesh.vertices = transformed_vertices_homogeneous[:, :3]
        
        return mesh


def compute_principal_axes_from_vertices(
    vertices: np.ndarray, 
    faces: Optional[np.ndarray] = None, 
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    메시 버텍스로부터 회전관성 주축을 계산합니다.
    trimesh 라이브러리를 사용하여 물리적으로 정확한 관성 텐서를 계산합니다.
    
    Args:
        vertices: 메시의 정점 좌표 배열 (N x 3)
        faces: 메시의 면 정보 (선택사항, 제공시 더 정확한 계산)
        verbose: 계산 과정을 출력할지 여부
    
    Returns:
        principal_axes: 회전관성 주축 (3x3 행렬, 각 열이 주축 벡터)
        eigenvalues: 각 주축에 대한 관성 모멘트 값 (작은 순서대로)
        centroid: 메시의 무게중심
    
    Example:
        >>> vertices = mesh.vertices
        >>> axes, moments, center = compute_principal_axes_from_vertices(vertices)
        >>> print(f"First principal axis: {axes[:, 0]}")
        >>> print(f"Second principal axis: {axes[:, 1]}")
        >>> print(f"Third principal axis: {axes[:, 2]}")
    """
    # trimesh 객체 생성
    if faces is not None:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    else:
        # faces가 없으면 convex hull 사용
        mesh = trimesh.convex.convex_hull(vertices)
    
    # trimesh의 내장 함수 사용
    principal_axes = mesh.principal_inertia_vectors
    eigenvalues = mesh.principal_inertia_components
    centroid = mesh.center_mass
    
    if verbose:
        print(f'[INFO] Center of mass: {centroid}')
        print(f'[INFO] Principal moments of inertia (eigenvalues): {eigenvalues}')
        print(f'[INFO] Principal inertia axes (eigenvectors):\n{principal_axes}')
        print(f'   - First axis (min inertia): {principal_axes[:, 0]}')
        print(f'   - Second axis (mid inertia): {principal_axes[:, 1]}')
        print(f'   - Third axis (max inertia): {principal_axes[:, 2]}')
    
    return principal_axes, eigenvalues, centroid


def compute_minimum_variance_axis_from_vertices(
    vertices: np.ndarray, 
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA(주성분 분석)를 사용하여 메시 버텍스의 분산이 가장 작은 주축을 계산합니다.
    scipy를 사용하여 효율적으로 계산합니다.
    
    Args:
        vertices: 메시의 정점 좌표 배열 (N x 3)
        verbose: 계산 과정을 출력할지 여부
    
    Returns:
        minimum_variance_axis: 분산이 가장 작은 주축 벡터 (3,)
        all_axes: 모든 주성분 축 (3x3 행렬, 각 열이 주축 벡터, 분산 작은 순서)
        variances: 각 주축에 대한 분산 값 (작은 순서대로)
    
    Example:
        >>> vertices = mesh.vertices
        >>> min_axis, all_axes, variances = compute_minimum_variance_axis_from_vertices(vertices)
        >>> print(f"Minimum variance axis: {min_axis}")
        >>> print(f"Variance ratio: {variances / np.sum(variances)}")
    
    Note:
        PCA는 공분산 행렬을 사용하여 데이터의 주성분을 찾습니다.
        분산이 가장 작은 축은 데이터가 가장 평평한 방향을 나타냅니다.
    """
    # 1. 데이터의 중심 계산
    centroid = np.mean(vertices, axis=0)
    if verbose:
        print(f'[INFO] PCA center: {centroid}')
    
    # 2. 중심을 원점으로 이동
    centered_vertices = vertices - centroid
    
    # 3. 공분산 행렬 계산 (scipy 사용)
    covariance_matrix = np.cov(centered_vertices.T)
    if verbose:
        print(f'[INFO] Covariance matrix:\n{covariance_matrix}')
    
    # 4. scipy의 eigh로 고유값/고유벡터 계산 (대칭 행렬에 최적화)
    eigenvalues, eigenvectors = eigh(covariance_matrix)
    
    # 5. 고유값(분산)이 작은 순서대로 정렬
    sorted_indices = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # 6. 결과 추출
    all_axes = eigenvectors  # 모든 주성분 축
    variances = eigenvalues  # 각 축의 분산
    minimum_variance_axis = eigenvectors[:, 0]  # 분산이 가장 작은 축
    
    if verbose:
        print(f'[INFO] Principal component variances (eigenvalues): {variances}')
        print(f'[INFO] Variance ratio: {variances / np.sum(variances) * 100}%')
        print(f'[INFO] Principal component axes (eigenvectors):\n{all_axes}')
        print(f'   - First axis (min variance): {all_axes[:, 0]} (variance: {variances[0]:.2f})')
        print(f'   - Second axis (mid variance): {all_axes[:, 1]} (variance: {variances[1]:.2f})')
        print(f'   - Third axis (max variance): {all_axes[:, 2]} (variance: {variances[2]:.2f})')
        print(f'[INFO] Minimum variance axis: {minimum_variance_axis}')
    
    return minimum_variance_axis, all_axes, variances
