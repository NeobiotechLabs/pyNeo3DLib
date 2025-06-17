import os
import numpy as np
import trimesh
import pyvista as pv
from typing import Optional, Union, List, Dict, Any, Tuple

class MeshLoader:
    """메쉬 파일을 로드하는 클래스"""
    
    file_path: str
    mesh: Optional[trimesh.Trimesh]
    
    def __init__(self, file_path: str) -> None:
        """
        메쉬 로더 초기화
        
        Args:
            file_path (str): 로드할 메쉬 파일 경로
        
        Raises:
            TypeError: 파일 경로가 문자열이 아닌 경우
        """
        if not isinstance(file_path, str):
            raise TypeError("file_path는 문자열(str)이어야 합니다.")
            
        self.file_path = file_path
        self.mesh: Optional[trimesh.Trimesh] = None
        
    def load(self) -> trimesh.Trimesh:
        """
        메쉬 파일을 로드하는 메서드
        
        Returns:
            trimesh.Trimesh: 로드된 메쉬 객체
            
        Raises:
            FileNotFoundError: 파일이 존재하지 않는 경우
            Exception: 메쉬 로드 중 오류가 발생한 경우
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {self.file_path}")
        
        try:
            self.mesh = trimesh.load(self.file_path)
            return self.mesh
        except Exception as e:
            raise Exception(f"메쉬 로드 중 오류 발생: {str(e)}")
            
    def to_pyvista(self) -> Tuple[np.ndarray, pv.PolyData]:
        """
        로드된 trimesh 메쉬를 PyVista 형식으로 변환하는 메서드
        
        Returns:
            Tuple[np.ndarray, pv.PolyData]: 
                - np.ndarray: 메쉬의 정점 좌표 배열 (N x 3 형태)
                - pv.PolyData: PyVista 메쉬 객체
            
        Raises:
            ValueError: 메쉬가 로드되지 않은 경우
        """
        if self.mesh is None:
            raise ValueError("메쉬가 로드되지 않았습니다. 먼저 load() 메서드를 호출하세요.")
        
        # trimesh에서 PyVista로 변환
        vertices = np.array(self.mesh.vertices, dtype=np.float32)
        faces = np.array(self.mesh.faces)
        
        # PyVista 메쉬 생성
        face_count = np.full(len(faces), 3, dtype=np.int32)
        face_data = np.column_stack((face_count, faces))
        pv_mesh = pv.PolyData(vertices, face_data.flatten())
        
        return vertices, pv_mesh
        
    @staticmethod
    def load_mesh(file_path: str) -> Tuple[np.ndarray, pv.PolyData]:
        """
        STL 파일에서 메쉬를 로드하는 함수
        
        Args:
            file_path (str): STL 파일 경로
            
        Returns:
            Tuple[np.ndarray, pv.PolyData]:
                - np.ndarray: 메쉬의 정점 좌표 배열 (N x 3 형태)
                - pv.PolyData: PyVista 메쉬 객체
        """
        loader = MeshLoader(file_path)
        loader.load()
        vertices, pv_mesh = loader.to_pyvista()
        return vertices, pv_mesh