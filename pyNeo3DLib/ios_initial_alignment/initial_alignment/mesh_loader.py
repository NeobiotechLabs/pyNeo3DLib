"""
메시 파일 로딩 및 검증을 담당하는 모듈
"""
import open3d as o3d
import os
from typing import Tuple

from ..utils.mesh_io import load_mesh_safe
from .constants import (
    ValidationConfig,
    ErrorMessages
)


class MeshLoader:
    """메시 파일 로딩 및 검증을 담당하는 클래스"""
    
    def __init__(self, min_vertices: int = ValidationConfig.MIN_MESH_VERTICES):
        """
        Args:
            min_vertices: 유효한 메시로 간주하기 위한 최소 버텍스 개수
        """
        self.min_vertices = min_vertices
    
    def load_mesh(self, file_path: str) -> o3d.geometry.TriangleMesh:
        """
        STL 파일을 로드하고 검증
        
        Args:
            file_path: STL 파일 경로
            
        Returns:
            o3d.geometry.TriangleMesh: 로드된 메시
            
        Raises:
            FileNotFoundError: 파일이 존재하지 않는 경우
            ValueError: 메시가 비어있거나 유효하지 않은 경우
        """
        # 파일 존재 확인
        if not os.path.exists(file_path):
            raise FileNotFoundError(ErrorMessages.FILE_NOT_FOUND.format(path=file_path))
        
        # 메시 로드
        try:
            mesh = load_mesh_safe(file_path)
        except Exception as e:
            raise ValueError(ErrorMessages.MESH_LOAD_FAILED.format(path=file_path)) from e
        
        # 메시 유효성 검증
        self.validate_mesh(mesh, file_path)
        
        return mesh
    
    def validate_mesh(self, mesh: o3d.geometry.TriangleMesh, file_path: str = None):
        """
        메시 유효성 검증
        
        Args:
            mesh: 검증할 메시
            file_path: 파일 경로 (에러 메시지용, 선택적)
            
        Raises:
            ValueError: 메시가 유효하지 않은 경우
        """
        if mesh is None:
            path_info = f": {file_path}" if file_path else ""
            raise ValueError(ErrorMessages.INVALID_MESH.format(path=path_info))
        
        if len(mesh.vertices) == 0:
            path_info = file_path if file_path else "unknown"
            raise ValueError(ErrorMessages.EMPTY_MESH.format(path=path_info))
        
        if len(mesh.vertices) < self.min_vertices:
            path_info = file_path if file_path else "unknown"
            raise ValueError(
                f"메시의 버텍스 개수가 너무 적습니다 ({len(mesh.vertices)} < {self.min_vertices}): {path_info}"
            )
    
    def load_mesh_pair(self, 
                      target_path: str, 
                      control_path: str) -> Tuple[o3d.geometry.TriangleMesh, o3d.geometry.TriangleMesh]:
        """
        타겟과 컨트롤 메시를 한 번에 로드
        
        Args:
            target_path: 타겟 STL 파일 경로
            control_path: 컨트롤 STL 파일 경로
            
        Returns:
            Tuple[o3d.geometry.TriangleMesh, o3d.geometry.TriangleMesh]: (타겟 메시, 컨트롤 메시)
            
        Raises:
            FileNotFoundError: 파일이 존재하지 않는 경우
            ValueError: 메시가 유효하지 않은 경우
        """
        target_mesh = self.load_mesh(target_path)
        control_mesh = self.load_mesh(control_path)
        
        return target_mesh, control_mesh
    
    def get_mesh_info(self, mesh: o3d.geometry.TriangleMesh) -> dict:
        """
        메시 정보 반환
        
        Args:
            mesh: 메시 객체
            
        Returns:
            dict: 메시 정보 딕셔너리
        """
        return {
            'num_vertices': len(mesh.vertices),
            'num_triangles': len(mesh.triangles),
            'has_vertex_normals': mesh.has_vertex_normals(),
            'has_vertex_colors': mesh.has_vertex_colors(),
            'is_watertight': mesh.is_watertight(),
            'is_orientable': mesh.is_orientable()
        }
