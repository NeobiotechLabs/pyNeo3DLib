"""
메쉬 및 포인트 클라우드 처리 모듈
"""
import numpy as np
import open3d as o3d
from typing import Tuple, Optional
import os


class GeometryProcessor:
    """메쉬 및 포인트 클라우드 처리를 담당하는 클래스"""
    
    @staticmethod
    def load_mesh(
        mesh_path: str,
        transform_matrix: Optional[np.ndarray] = None,
        color: Optional[Tuple[float, float, float]] = None,
        verbose: bool = True
    ) -> o3d.geometry.TriangleMesh:
        """
        메쉬를 로드하고 선택적으로 변환 및 색상 적용
        
        Args:
            mesh_path: 메쉬 파일 경로
            transform_matrix: 적용할 4x4 동차변환행렬 (선택사항)
            color: RGB 색상 튜플 (선택사항)
            verbose: 상세 출력 여부
        
        Returns:
            o3d.geometry.TriangleMesh: 로드된 메쉬
        
        Raises:
            FileNotFoundError: 메쉬 파일이 존재하지 않을 경우
        """
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"메쉬 파일을 찾을 수 없습니다: {mesh_path}")
        
        # 메쉬 로드
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        
        if verbose:
            print(f"메쉬 로드 완료: {len(mesh.vertices)} vertices")
        
        # 변환 행렬 적용
        if transform_matrix is not None:
            mesh.transform(transform_matrix)
            if verbose:
                print(f"변환 행렬 적용 완료")
        
        # 색상 적용
        if color is not None:
            mesh.paint_uniform_color(color)
        
        return mesh
    
    @staticmethod
    def extract_top_y_points(
        mesh: o3d.geometry.TriangleMesh,
        num_samples: int = 100000,
        verbose: bool = True
    ) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
        """
        메쉬를 포인트 클라우드로 변환하고 Y값이 가장 큰 포인트를 찾음
        (코 끝 포인트 추정용)
        
        Args:
            mesh: 입력 메쉬
            num_samples: 샘플링할 포인트 개수
            verbose: 상세 출력 여부
        
        Returns:
            tuple: (pcd, top_point)
                - pcd: 샘플링된 포인트 클라우드
                - top_point: Y값이 가장 큰 포인트의 좌표 (3,)
        """
        # 1. 메쉬를 포인트 클라우드로 변환 (균일 샘플링)
        pcd = mesh.sample_points_uniformly(number_of_points=num_samples)
        points = np.asarray(pcd.points)
        
        # 2. Y값이 가장 큰 포인트 찾기
        max_y_idx = np.argmax(points[:, 1])
        top_point = points[max_y_idx]
        
        return pcd, top_point
    
    @staticmethod
    def calculate_bbox_extent(
        pcd: o3d.geometry.PointCloud,
        verbose: bool = True
    ) -> np.ndarray:
        """
        포인트 클라우드의 바운딩 박스 크기 계산
        
        Args:
            pcd: 입력 포인트 클라우드
            verbose: 상세 출력 여부
        
        Returns:
            np.ndarray: [x_width, y_depth, z_height] 형태의 바운딩 박스 크기
        """
        bbox = pcd.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()  # [x_width, y_depth, z_height]
        
        if verbose:
            print(f"바운딩 박스 크기:")
            print(f"  X축 (좌우): {extent[0]:.2f} mm")
            print(f"  Y축 (전후): {extent[1]:.2f} mm")
            print(f"  Z축 (상하): {extent[2]:.2f} mm")
        
        return extent
    
    @staticmethod
    def filter_region_by_bbox(
        pcd: o3d.geometry.PointCloud,
        center_point: np.ndarray,
        bbox_extent: np.ndarray,
        color: Optional[Tuple[float, float, float]] = None,
        verbose: bool = True
    ) -> o3d.geometry.PointCloud:
        """
        중심점을 기준으로 특정 영역의 포인트 클라우드만 필터링
        
        Args:
            pcd: 입력 포인트 클라우드
            center_point: 필터링 영역의 중심점 (3,)
            bbox_extent: 필터링 영역의 크기 [x_width, y_depth, z_height]
            color: RGB 색상 튜플 (선택사항)
            verbose: 상세 출력 여부
        
        Returns:
            o3d.geometry.PointCloud: 필터링된 포인트 클라우드
        """
        x_width, y_depth, z_height = bbox_extent
        
        # 바운딩 박스 정의
        # x: 좌우 (중심 기준 ±width/2)
        # y: 전후 (중심에서 앞쪽으로 y_depth만큼)
        # z: 상하 (중심 기준 ±height/2)
        bbox_min = center_point - np.array([x_width/2, y_depth, z_height/2])
        bbox_max = center_point + np.array([x_width/2, y_depth, z_height/2])
        
        if verbose:
            print(f"필터링 영역:")
            print(f"  중심점: {center_point}")
            print(f"  최소 경계: {bbox_min}")
            print(f"  최대 경계: {bbox_max}")
        
        # 영역 크롭
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=bbox_min, max_bound=bbox_max)
        filtered_pcd = pcd.crop(bbox)
        
        # 색상 적용
        if color is not None:
            filtered_pcd.paint_uniform_color(color)
        
        if verbose:
            print(f"필터링 결과: {len(filtered_pcd.points)} 포인트")
        
        return filtered_pcd
    
    @staticmethod
    def filter_points_near_surface(
        pcd: o3d.geometry.PointCloud,
        mesh: o3d.geometry.TriangleMesh,
        distance_threshold: float = 5.0,
        verbose: bool = True
    ) -> o3d.geometry.PointCloud:
        """
        메쉬 표면에서 가까운 포인트만 필터링 (SDF 기반)
        
        Args:
            pcd: 필터링할 포인트 클라우드
            mesh: 기준 메쉬
            distance_threshold: 거리 임계값 (mm 단위, 이 값보다 가까운 포인트만 선택)
            verbose: 상세 출력 여부
        
        Returns:
            o3d.geometry.PointCloud: 필터링된 포인트 클라우드
        """
        if verbose:
            print(f"\n[SDF 기반 표면 필터링]")
            print(f"  거리 임계값: {distance_threshold} mm")
            print(f"  입력 포인트 수: {len(pcd.points)}")
        
        # 1. 메쉬를 RaycastingScene으로 변환
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh_t)
        
        # 2. 각 포인트에서 메쉬 표면까지의 최단 거리 계산
        points = np.asarray(pcd.points)
        query_points = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
        
        # compute_signed_distance: SDF 계산
        signed_distances = scene.compute_signed_distance(query_points).numpy()
        distances = np.abs(signed_distances)  # 절대값 (표면으로부터의 거리)
        
        # 3. 임계값 이하의 포인트만 선택
        mask = distances < distance_threshold
        filtered_points = points[mask]
        
        # 4. 새로운 포인트 클라우드 생성
        pcd_filtered = o3d.geometry.PointCloud()
        pcd_filtered.points = o3d.utility.Vector3dVector(filtered_points)
        
        # 색상 정보가 있으면 함께 필터링
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)[mask]
            pcd_filtered.colors = o3d.utility.Vector3dVector(colors)
        
        if verbose:
            print(f"  출력 포인트 수: {len(pcd_filtered.points)}")
            print(f"  필터링 비율: {len(pcd_filtered.points) / len(pcd.points) * 100:.2f}%")
            print(f"  거리 통계:")
            print(f"    - 최소: {distances[mask].min():.2f} mm")
            print(f"    - 최대: {distances[mask].max():.2f} mm")
            print(f"    - 평균: {distances[mask].mean():.2f} mm")
        
        return pcd_filtered
    
    @staticmethod
    def create_coordinate_frame(size: float = 50.0) -> o3d.geometry.TriangleMesh:
        """
        좌표축 프레임 생성
        
        Args:
            size: 좌표축 크기
        
        Returns:
            o3d.geometry.TriangleMesh: 좌표축 메쉬
        """
        return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)


