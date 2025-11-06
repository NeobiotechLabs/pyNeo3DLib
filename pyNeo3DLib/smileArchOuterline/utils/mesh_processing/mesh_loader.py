import numpy as np
import pyvista as pv
from typing import Optional
from pathlib import Path

class MeshLoader:
    """메쉬 로딩 및 기본적인 메쉬 속성(중심화, 대각선 길이) 처리를 담당하는 클래스"""
    
    def __init__(self):
        self.mesh: Optional[pv.PolyData] = None
        self.vertices: Optional[np.ndarray] = None
        self.center: Optional[np.ndarray] = None
        self.diagonal_length: Optional[float] = None

    def load_mesh(self, mesh_path: str):
        """
        STL 파일로부터 메쉬를 로드하고 초기화합니다.

        Args:
            mesh_path: STL 파일 경로

        Raises:
            FileNotFoundError: STL 파일이 존재하지 않을 경우
        """
        stl_path = Path(mesh_path)
        if not stl_path.exists():
            raise FileNotFoundError(f"STL 파일을 찾을 수 없습니다: {mesh_path}")
        
        self.mesh = pv.read(str(stl_path))
        self.vertices = np.asarray(self.mesh.points)
        
        self._center_mesh()
        self.diagonal_length = self._compute_diagonal_length()

    def _center_mesh(self) -> None:
        """메쉬를 중심점이 원점이 되도록 평행이동"""
        center = self.vertices.mean(axis=0)
        self.mesh.points = self.vertices - center
        self.vertices = np.asarray(self.mesh.points)
        self.center = self.vertices.mean(axis=0)  # 원점에 가까움

    def _compute_diagonal_length(self) -> float:
        """
        바운딩 박스의 대각선 길이 계산
        
        Returns:
            대각선 길이
        """
        bounds = self.mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
        min_bound = np.array([bounds[0], bounds[2], bounds[4]])
        max_bound = np.array([bounds[1], bounds[3], bounds[5]])
        return np.linalg.norm(max_bound - min_bound)
