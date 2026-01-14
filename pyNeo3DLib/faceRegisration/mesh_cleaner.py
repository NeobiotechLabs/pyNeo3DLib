"""
메쉬 클리닝 모듈

이 모듈은 메쉬의 노이즈 제거 및 정리 기능을 담당합니다.
단일 책임 원칙(SRP)에 따라 메쉬 클리닝 로직만을 캡슐화합니다.
"""
import numpy as np
import open3d as o3d
import pyvista as pv

from pyNeo3DLib.fileLoader.mesh import Mesh
from pyNeo3DLib.faceRegisration.mesh_converter import MeshConverter
from pyNeo3DLib.faceRegisration.constants import (
    MeshCleaningConstants,
    NoiseRemovalConstants
)


class MeshCleaner:
    """
    메쉬 클리닝을 담당하는 클래스.
    
    단일 책임: 메쉬의 노이즈 제거 및 정리
    
    이 클래스는 다음 기능을 제공합니다:
    - 작은 컴포넌트 제거
    - 경계 스무딩
    - 노말 기반 노이즈 제거
    - 중복 정점/삼각형 제거
    """
    
    def clean_mesh(
        self, 
        mesh: Mesh,
        min_cluster_ratio: float = MeshCleaningConstants.MIN_CLUSTER_RATIO,
        smooth_iterations: int = MeshCleaningConstants.SMOOTH_ITERATIONS
    ) -> Mesh:
        """
        메쉬의 지저분한 경계를 정리합니다.
        
        Args:
            mesh: 정리할 Mesh 객체
            min_cluster_ratio: 제거할 작은 클러스터의 최소 비율
            smooth_iterations: Laplacian 스무딩 반복 횟수
            
        Returns:
            Mesh: 정리된 메쉬
        """
        if mesh is None or mesh.vertices is None or len(mesh.vertices) == 0:
            return mesh
        
        try:
            print("메쉬 경계 정리 시작...")
            
            # Open3D로 변환
            mesh_o3d = MeshConverter.mesh_to_open3d(mesh)
            original_triangles = len(mesh_o3d.triangles)
            
            # 1. 작은 컴포넌트 제거
            mesh_o3d = self._remove_small_components(mesh_o3d, min_cluster_ratio)
            
            # 2. 중복 및 참조되지 않는 요소 제거
            mesh_o3d = self._remove_duplicates_and_unreferenced(mesh_o3d)
            
            # 3. 경계 스무딩
            if smooth_iterations > 0:
                mesh_o3d = mesh_o3d.filter_smooth_laplacian(
                    number_of_iterations=smooth_iterations,
                    lambda_filter=MeshCleaningConstants.LAPLACIAN_LAMBDA
                )
            
            final_triangles = len(mesh_o3d.triangles)
            print(f"메쉬 정리 완료: {original_triangles} → {final_triangles} 삼각형")
            
            # Mesh로 변환하여 반환
            return MeshConverter.open3d_to_mesh(mesh_o3d)
            
        except Exception as e:
            print(f"메쉬 정리 중 오류 발생: {e}")
            return mesh
    
    def remove_noise_by_normal_angle(
        self, 
        mesh: Mesh,
        reference_vector: np.ndarray = None,
        angle_threshold_degrees: float = NoiseRemovalConstants.NORMAL_ANGLE_THRESHOLD_DEGREES
    ) -> Mesh:
        """
        노말벡터와 기준벡터 간의 각도를 기반으로 노이즈를 제거합니다.
        
        Args:
            mesh: 처리할 Mesh 객체
            reference_vector: 기준 벡터 (기본값: [0, 1, 0])
            angle_threshold_degrees: 제거할 각도 임계값 (도)
            
        Returns:
            Mesh: 노이즈가 제거된 메쉬
        """
        if reference_vector is None:
            reference_vector = np.array([0, 1, 0])
        
        # PyVista로 변환
        pv_mesh = MeshConverter.mesh_to_pyvista(mesh)
        
        # 노말 기반 필터링
        normal_vectors = np.asarray(pv_mesh.point_normals)
        normal_vectors_dot_product = np.dot(normal_vectors, reference_vector)
        normal_vectors_angle = np.arccos(np.abs(normal_vectors_dot_product))
        normal_vectors_angle_mask = normal_vectors_angle > (angle_threshold_degrees * np.pi / 180)
        indices_to_remove = np.where(normal_vectors_angle_mask)[0]
        
        # 포인트 제거 및 가장 큰 연결 컴포넌트 추출
        cleaned_pv_mesh, _ = pv_mesh.remove_points(indices_to_remove)
        cleaned_pv_mesh = cleaned_pv_mesh.connectivity(extraction_mode='largest')
        
        # Mesh로 변환하여 반환
        return MeshConverter.pyvista_to_mesh(cleaned_pv_mesh)
    
    def clip_mesh_by_axis_range(
        self,
        mesh: Mesh,
        axis: int,
        min_value: float,
        max_value: float,
        extract_largest: bool = True
    ) -> Mesh:
        """
        메쉬를 특정 축 기준으로 지정된 범위만 남기고 클립합니다.
        
        Args:
            mesh: 클립할 Mesh 객체
            axis: 클립할 축 (0=x, 1=y, 2=z)
            min_value: 유지할 최소값
            max_value: 유지할 최대값
            extract_largest: 가장 큰 연결된 덩어리만 반환할지 여부
            
        Returns:
            Mesh: 클립된 Mesh 객체
        """
        # PyVista로 변환
        pv_mesh = MeshConverter.mesh_to_pyvista(mesh)
        
        # 축에 따른 법선 벡터 설정
        normal_positive = [0, 0, 0]
        normal_negative = [0, 0, 0]
        normal_positive[axis] = 1
        normal_negative[axis] = -1
        
        # max_value 평면으로 클립
        origin_max = [0, 0, 0]
        origin_max[axis] = max_value
        clipped = pv_mesh.clip(normal=normal_positive, origin=origin_max, invert=True)
        
        # min_value 평면으로 클립
        origin_min = [0, 0, 0]
        origin_min[axis] = min_value
        clipped = clipped.clip(normal=normal_negative, origin=origin_min, invert=True)
        
        # 가장 큰 연결된 덩어리만 추출
        if extract_largest:
            clipped = clipped.connectivity(extraction_mode='largest')
        
        # Mesh로 변환하여 반환
        return MeshConverter.pyvista_to_mesh(clipped)
    
    def slice_mesh(
        self, 
        mesh: Mesh,
        origin: np.ndarray,
        normal: np.ndarray
    ) -> Mesh:
        """
        평면으로 메쉬를 자르고 법선벡터 방향의 메쉬를 반환합니다.
        
        Args:
            mesh: 자를 Mesh 객체
            origin: 슬라이싱 평면의 시작점
            normal: 슬라이싱 평면의 법선벡터 (유지할 방향)
            
        Returns:
            Mesh: 잘린 메쉬
        """
        # 벡터 정규화
        normal = np.array(normal).flatten()
        normal = normal / np.linalg.norm(normal)
        origin = np.array(origin).flatten()
        
        # PyVista로 변환 및 클립
        pv_mesh = MeshConverter.mesh_to_pyvista(mesh)
        clipped_pv_mesh = pv_mesh.clip(normal=normal, origin=origin, invert=False)
        clipped_pv_mesh = clipped_pv_mesh.connectivity(extraction_mode='largest')
        
        # Mesh로 변환하여 반환
        return MeshConverter.pyvista_to_mesh(clipped_pv_mesh)
    
    def _remove_small_components(
        self, 
        mesh_o3d: o3d.geometry.TriangleMesh,
        min_cluster_ratio: float
    ) -> o3d.geometry.TriangleMesh:
        """작은 컴포넌트를 제거합니다."""
        triangle_clusters, cluster_n_triangles, _ = mesh_o3d.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        
        if len(cluster_n_triangles) > 1:
            largest_cluster_idx = cluster_n_triangles.argmax()
            largest_cluster_size = cluster_n_triangles[largest_cluster_idx]
            min_size = int(largest_cluster_size * min_cluster_ratio)
            
            triangles_to_remove = np.zeros(len(triangle_clusters), dtype=bool)
            for cluster_idx, cluster_size in enumerate(cluster_n_triangles):
                if cluster_size < min_size:
                    triangles_to_remove[triangle_clusters == cluster_idx] = True
            
            removed_count = np.sum(triangles_to_remove)
            if removed_count > 0:
                mesh_o3d.remove_triangles_by_mask(triangles_to_remove)
                print(f"작은 컴포넌트 제거: {removed_count}개 삼각형 제거됨")
        
        return mesh_o3d
    
    def _remove_duplicates_and_unreferenced(
        self, 
        mesh_o3d: o3d.geometry.TriangleMesh
    ) -> o3d.geometry.TriangleMesh:
        """중복 및 참조되지 않는 요소를 제거합니다."""
        mesh_o3d.remove_unreferenced_vertices()
        mesh_o3d.remove_duplicated_vertices()
        mesh_o3d.remove_duplicated_triangles()
        mesh_o3d.remove_degenerate_triangles()
        return mesh_o3d

