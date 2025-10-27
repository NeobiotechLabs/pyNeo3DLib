"""
Face Normal 필터링을 위한 유틸리티 모듈
수직 방향과 유사한 법선벡터를 가진 면들을 찾는 기능을 제공합니다.
"""

import numpy as np
import pyvista as pv
from typing import Tuple, List, Optional
from .visualizer import VisualizeForTest


class FaceNormalFilter:
    """Face Normal 필터링을 위한 클래스"""
    
    def __init__(self, tolerance: float = 0.1):
        """
        FaceNormalFilter 초기화
        
        Args:
            tolerance: 수직 방향과의 허용 오차 (0~1, 1에 가까울수록 엄격)
        """
        self.tolerance = tolerance
    
    def find_vertical_faces(self, mesh: pv.PolyData, 
                          vertical_direction: np.ndarray = np.array([0, 0, 1])) -> Tuple[pv.PolyData, np.ndarray]:
        """
        수직 방향과 유사한 법선벡터를 가진 면들을 찾습니다.
        
        Args:
            mesh: 입력 메시
            vertical_direction: 수직 방향 벡터 (기본값: Z축)
            
        Returns:
            Tuple[vertical_mesh, face_indices]:
                - vertical_mesh: 수직 방향 면들만 포함한 메시
                - face_indices: 수직 방향 면들의 인덱스
        """
        # 면 법선벡터 계산
        mesh_with_normals = mesh.compute_normals()
        
        # 각 면의 법선벡터와 수직 방향 벡터의 내적 계산
        dot_products = np.abs(np.dot(mesh_with_normals.face_normals, vertical_direction))
        
        # 수직 방향과 거의 같은 방향인 면들 찾기
        vertical_face_indices = np.where(dot_products > (1 - self.tolerance))[0]
        
        if len(vertical_face_indices) > 0:
            vertical_mesh = mesh_with_normals.extract_cells(vertical_face_indices)
            return vertical_mesh, vertical_face_indices
        else:
            return pv.PolyData(), np.array([])
    
    def find_horizontal_faces(self, mesh: pv.PolyData, 
                            horizontal_direction: np.ndarray = np.array([0, 1, 0])) -> Tuple[pv.PolyData, np.ndarray]:
        """
        수평 방향과 유사한 법선벡터를 가진 면들을 찾습니다.
        
        Args:
            mesh: 입력 메시
            horizontal_direction: 수평 방향 벡터 (기본값: Y축)
            
        Returns:
            Tuple[horizontal_mesh, face_indices]:
                - horizontal_mesh: 수평 방향 면들만 포함한 메시
                - face_indices: 수평 방향 면들의 인덱스
        """
        # 면 법선벡터 계산
        mesh_with_normals = mesh.compute_normals()
        
        # 각 면의 법선벡터와 수평 방향 벡터의 내적 계산
        dot_products = np.abs(np.dot(mesh_with_normals.face_normals, horizontal_direction))
        
        # 수평 방향과 거의 같은 방향인 면들 찾기
        horizontal_face_indices = np.where(dot_products > (1 - self.tolerance))[0]
        
        if len(horizontal_face_indices) > 0:
            horizontal_mesh = mesh_with_normals.extract_cells(horizontal_face_indices)
            return horizontal_mesh, horizontal_face_indices
        else:
            return pv.PolyData(), np.array([])
    
    def find_faces_by_angle(self, mesh: pv.PolyData, 
                           target_direction: np.ndarray, 
                           max_angle_degrees: float = 10.0) -> Tuple[pv.PolyData, np.ndarray]:
        """
        특정 방향과 일정 각도 이내의 법선벡터를 가진 면들을 찾습니다.
        
        Args:
            mesh: 입력 메시
            target_direction: 목표 방향 벡터
            max_angle_degrees: 최대 허용 각도 (도 단위)
            
        Returns:
            Tuple[filtered_mesh, face_indices]:
                - filtered_mesh: 필터링된 면들만 포함한 메시
                - face_indices: 필터링된 면들의 인덱스
        """
        # 면 법선벡터 계산
        mesh_with_normals = mesh.compute_normals()
        
        # 각 면의 법선벡터와 목표 방향 벡터의 내적 계산
        dot_products = np.dot(mesh_with_normals.face_normals, target_direction)
        
        # 각도 계산 (라디안을 도로 변환)
        angles_degrees = np.degrees(np.arccos(np.abs(dot_products)))
        
        # 허용 각도 이내의 면들 찾기
        valid_face_indices = np.where(angles_degrees <= max_angle_degrees)[0]
        
        if len(valid_face_indices) > 0:
            filtered_mesh = mesh_with_normals.extract_cells(valid_face_indices)
            return filtered_mesh, valid_face_indices
        else:
            return pv.PolyData(), np.array([])
    
    def analyze_face_normals(self, mesh: pv.PolyData) -> dict:
        """
        메시의 면 법선벡터들을 분석합니다.
        
        Args:
            mesh: 입력 메시
            
        Returns:
            dict: 법선벡터 분석 결과
        """
        mesh_with_normals = mesh.compute_normals()
        normals = mesh_with_normals.face_normals
        
        # 각 축과의 내적 계산
        x_dot = np.abs(np.dot(normals, np.array([1, 0, 0])))
        y_dot = np.abs(np.dot(normals, np.array([0, 1, 0])))
        z_dot = np.abs(np.dot(normals, np.array([0, 0, 1])))
        
        # 각 축과 유사한 방향의 면 개수 계산
        tolerance = 0.1
        x_aligned = np.sum(x_dot > (1 - tolerance))
        y_aligned = np.sum(y_dot > (1 - tolerance))
        z_aligned = np.sum(z_dot > (1 - tolerance))
        
        return {
            'total_faces': len(normals),
            'x_aligned_faces': x_aligned,
            'y_aligned_faces': y_aligned,
            'z_aligned_faces': z_aligned,
            'x_aligned_ratio': x_aligned / len(normals),
            'y_aligned_ratio': y_aligned / len(normals),
            'z_aligned_ratio': z_aligned / len(normals)
        }
    
    def visualize_face_normals(self, mesh: pv.PolyData, 
                              vertical_mesh: Optional[pv.PolyData] = None,
                              horizontal_mesh: Optional[pv.PolyData] = None,
                              show_original: bool = True) -> None:
        """
        면 법선벡터 분석 결과를 시각화합니다.
        
        Args:
            mesh: 원본 메시
            vertical_mesh: 수직 방향 면들 (선택사항)
            horizontal_mesh: 수평 방향 면들 (선택사항)
            show_original: 원본 메시 표시 여부
        """
        visualizer = VisualizeForTest()
        
        if show_original:
            visualizer.visualize_mesh(mesh, color='lightblue', opacity=0.3, title="원본 메시")
        
        if vertical_mesh is not None and len(vertical_mesh.points) > 0:
            visualizer.visualize_mesh(vertical_mesh, color='red', opacity=0.7, title="수직 방향 면들")
        
        if horizontal_mesh is not None and len(horizontal_mesh.points) > 0:
            visualizer.visualize_mesh(horizontal_mesh, color='green', opacity=0.7, title="수평 방향 면들")
        
        visualizer.show()
