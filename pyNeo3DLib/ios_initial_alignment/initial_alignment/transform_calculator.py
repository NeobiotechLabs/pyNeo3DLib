import numpy as np
from typing import List, TYPE_CHECKING
from .constants import OBBConfig

if TYPE_CHECKING:
    import open3d as o3d


class TransformCalculator:
    """변환 행렬 계산을 담당하는 클래스"""
    
    def __init__(self, min_determinant_threshold: float = OBBConfig.MIN_DETERMINANT_THRESHOLD):
        """
        Args:
            min_determinant_threshold: 회전 행렬의 최소 determinant 임계값 (reflection 제거용)
        """
        self.min_determinant_threshold = min_determinant_threshold
    
    def generate_permutation_matrices(self) -> List[np.ndarray]:
        """
        Permutation 행렬 생성
        
        Returns:
            List[np.ndarray]: 6개의 permutation 행렬 리스트
        """
        return [
            np.eye(3),
            np.array([[0,1,0],[1,0,0],[0,0,1]]),
            np.array([[0,0,1],[0,1,0],[1,0,0]]),
            np.array([[0,1,0],[0,0,1],[1,0,0]]),
            np.array([[0,0,1],[1,0,0],[0,1,0]]),
            np.array([[1,0,0],[0,0,1],[0,1,0]])
        ]
    
    def generate_sign_matrices(self) -> List[np.ndarray]:
        """
        Sign 행렬 생성 (대각 행렬)
        
        Returns:
            List[np.ndarray]: 8개의 sign 행렬 리스트
        """
        signs = []
        for sx in [1,-1]:
            for sy in [1,-1]:
                for sz in [1,-1]:
                    signs.append(np.diag([sx, sy, sz]))
        return signs
    
    def compute_rotation_candidates(self, B_src: np.ndarray, B_tgt: np.ndarray) -> List[np.ndarray]:
        """
        후보 회전 행렬 생성 (det > threshold만 허용하여 reflection 제거)
        
        Args:
            B_src: 소스 좌표계 기저 행렬
            B_tgt: 타겟 좌표계 기저 행렬
            
        Returns:
            List[np.ndarray]: 유효한 회전 행렬 리스트
        """
        rotations = []
        for P in self.generate_permutation_matrices():
            for S in self.generate_sign_matrices():
                R = B_tgt @ S @ P @ B_src.T
                # reflection 제거 (det > threshold)
                if np.linalg.det(R) > self.min_determinant_threshold:
                    rotations.append(R)
        return rotations
    
    def build_transform_matrix(self, R: np.ndarray, c_src: np.ndarray, c_tgt: np.ndarray) -> np.ndarray:
        """
        평행이동을 포함한 4x4 변환행렬 생성
        
        Args:
            R: 3x3 회전 행렬
            c_src: 소스 중심점
            c_tgt: 타겟 중심점
            
        Returns:
            np.ndarray: 4x4 변환 행렬
        """
        t = c_tgt - R @ c_src
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    
    @staticmethod
    def apply_transform(pcd: 'o3d.geometry.PointCloud', transform_matrix: np.ndarray) -> 'o3d.geometry.PointCloud':
        """
        포인트클라우드에 변환 행렬 적용
        
        Args:
            pcd: 입력 포인트클라우드
            transform_matrix: 4x4 변환 행렬
            
        Returns:
            o3d.geometry.PointCloud: 변환된 포인트클라우드
        """
        transformed_pcd = pcd.copy()
        transformed_pcd.transform(transform_matrix)
        return transformed_pcd
