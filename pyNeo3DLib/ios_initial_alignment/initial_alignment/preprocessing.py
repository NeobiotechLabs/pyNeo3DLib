import open3d as o3d
import numpy as np
from typing import List, Tuple, Optional
from pyNeo3DLib.ios_initial_alignment.global_fit.mesh_converter import MeshConverter
from .obb_analyzer import OBBAnalyzer
from .transform_calculator import TransformCalculator
from .distance_calculator import DistanceCalculator
from .constants import (
    SamplingConfig
)


class MeshPreprocessor:
    """3D 메쉬 전처리를 위한 통합 클래스"""
    
    def __init__(self, 
                 default_sample_points: int = SamplingConfig.DEFAULT_SAMPLE_POINTS,
                 random_seed: Optional[int] = SamplingConfig.DEFAULT_RANDOM_SEED):
        """
        Args:
            default_sample_points: 기본 샘플링 포인트 수
            random_seed: 랜덤 시드 (결정론적 결과를 위해 고정값 사용)
                        None으로 설정하면 랜덤 시드를 설정하지 않음
        """
        self.default_sample_points = default_sample_points
        self.random_seed = random_seed
        
        # 랜덤 시드 설정 (결정론적 결과를 위해)
        if random_seed is not None:
            o3d.utility.random.seed(random_seed)
        
        self.mesh_converter = MeshConverter(default_sample_points, random_seed=random_seed)
        self.obb_analyzer = OBBAnalyzer()
        self.transform_calculator = TransformCalculator()
        self.distance_calculator = DistanceCalculator()
        
    # === 메쉬 변환 관련 메서드들 ===
    def downsample_mesh(self, 
                       mesh: o3d.geometry.TriangleMesh, 
                       voxel_size: float = SamplingConfig.DEFAULT_VOXEL_SIZE, 
                       sample_points: Optional[int] = None) -> o3d.geometry.PointCloud:
        """
        STL mesh를 point cloud로 변환 후 voxel 기반 다운샘플링
        
        Args:
            mesh (o3d.geometry.TriangleMesh): 입력 메쉬
            voxel_size (float): 복셀 크기
            sample_points (int, optional): 샘플링할 포인트 수
            
        Returns:
            o3d.geometry.PointCloud: 다운샘플링된 포인트 클라우드
        """
        if sample_points is None:
            sample_points = self.default_sample_points
        
        # 랜덤 시드 재설정 (매번 동일한 샘플링 결과를 얻기 위해)
        if self.random_seed is not None:
            o3d.utility.random.seed(self.random_seed)
            
        pcd = mesh.sample_points_uniformly(number_of_points=sample_points)
        return pcd.voxel_down_sample(voxel_size)
    

    def target_mesh_to_pointcloud(self, 
                                 target_mesh: o3d.geometry.TriangleMesh, 
                                 sample_points: Optional[int] = None) -> o3d.geometry.PointCloud:
        """
        target_mesh를 입력받아 o3d.geometry.PointCloud 타입의 포인트 클라우드를 생성
        
        Args:
            target_mesh (o3d.geometry.TriangleMesh): 타겟 메쉬
            sample_points (int, optional): 샘플링할 포인트 수
            
        Returns:
            o3d.geometry.PointCloud: 생성된 포인트 클라우드
        """
        return self.mesh_converter.mesh_to_pointcloud(target_mesh, sample_points)
    
    # === OBB 분석 관련 메서드들 ===
    def compute_OBB_info(self, pcd: o3d.geometry.PointCloud) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Open3D의 OBB 기반으로 중심 좌표, 축 방향, 각 축 크기를 계산한다.

        Args:
            pcd (o3d.geometry.PointCloud): 입력 포인트 클라우드

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (중심점, 축 방향 행렬, 축 크기)
                - center: (3,) 중심 좌표
                - axes: (3, 3) 각 열이 주축 벡터 (X, Y, Z 축 방향)
                - extent: (3,) 각 축 방향의 길이
        """
        return self.obb_analyzer.compute_obb_info(pcd)
    
    # === 변환 행렬 관련 메서드들 ===
    def generate_permutation_matrices(self) -> List[np.ndarray]:
        """
        Permutation 행렬 생성
        
        Returns:
            List[np.ndarray]: 6개의 permutation 행렬 리스트
        """
        return self.transform_calculator.generate_permutation_matrices()
    
    def generate_sign_matrices(self) -> List[np.ndarray]:
        """
        Sign 행렬 생성 (대각 행렬)
        
        Returns:
            List[np.ndarray]: 8개의 sign 행렬 리스트
        """
        return self.transform_calculator.generate_sign_matrices()
    
    def compute_rotation_candidates(self, B_src: np.ndarray, B_tgt: np.ndarray) -> List[np.ndarray]:
        """
        후보 회전 행렬 생성 (det > 0만 허용)
        
        Args:
            B_src (np.ndarray): 소스 좌표계 기저 행렬
            B_tgt (np.ndarray): 타겟 좌표계 기저 행렬
            
        Returns:
            List[np.ndarray]: 유효한 회전 행렬 리스트
        """
        return self.transform_calculator.compute_rotation_candidates(B_src, B_tgt)
    
    def build_transform(self, R: np.ndarray, c_src: np.ndarray, c_tgt: np.ndarray) -> np.ndarray:
        """
        평행이동을 포함한 4x4 변환행렬 생성
        
        Args:
            R (np.ndarray): 3x3 회전 행렬
            c_src (np.ndarray): 소스 중심점
            c_tgt (np.ndarray): 타겟 중심점
            
        Returns:
            np.ndarray: 4x4 변환 행렬
        """
        return self.transform_calculator.build_transform_matrix(R, c_src, c_tgt)
    
    # === 거리 계산 관련 메서드들 ===
    def compute_rmse_bidirectional(self, 
                                   src_pcd: o3d.geometry.PointCloud, 
                                   tgt_pcd: o3d.geometry.PointCloud, 
                                   step: Optional[int] = None) -> float:
        """
        Chamfer Distance 기반 RMSE (양방향 최근접 거리).
        
        Args:
            src_pcd (o3d.geometry.PointCloud): 소스 포인트 클라우드
            tgt_pcd (o3d.geometry.PointCloud): 타겟 포인트 클라우드
            step (int): 샘플링 간격 (10이면 1/10점만 사용 → 속도 향상)
            
        Returns:
            float: 양방향 RMSE 값
        """
        return self.distance_calculator.compute_rmse_bidirectional(src_pcd, tgt_pcd, step)
    
    # === 편의 메서드들 ===
    def get_coordinate_frame(self, pcd: o3d.geometry.PointCloud) -> Tuple[np.ndarray, np.ndarray]:
        """
        포인트클라우드의 좌표계 정보를 반환
        
        Args:
            pcd (o3d.geometry.PointCloud): 입력 포인트 클라우드
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (중심점, 기저 행렬)
        """
        return self.obb_analyzer.get_coordinate_frame(pcd)
    
    def apply_transform(self, pcd: o3d.geometry.PointCloud, transform_matrix: np.ndarray) -> o3d.geometry.PointCloud:
        """
        포인트클라우드에 변환 행렬 적용
        
        Args:
            pcd (o3d.geometry.PointCloud): 입력 포인트클라우드
            transform_matrix (np.ndarray): 4x4 변환 행렬
            
        Returns:
            o3d.geometry.PointCloud: 변환된 포인트클라우드
        """
        return self.transform_calculator.apply_transform(pcd, transform_matrix)