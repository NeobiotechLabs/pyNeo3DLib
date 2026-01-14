"""
메쉬 변환 모듈

이 모듈은 메쉬에 대한 좌표 변환 기능을 담당합니다.
단일 책임 원칙(SRP)에 따라 변환 로직만을 캡슐화합니다.
"""
import numpy as np
import copy
from typing import Tuple

from pyNeo3DLib.fileLoader.mesh import Mesh


class MeshTransformer:
    """
    메쉬 변환을 담당하는 클래스.
    
    단일 책임: 메쉬의 좌표 변환 (회전, 이동, 스케일)
    
    이 클래스는 다음 기능을 제공합니다:
    - 변환 행렬 적용
    - 회전 변환
    - 이동 변환
    - 좌표계 변환
    """
    
    @staticmethod
    def apply_transformation(mesh: Mesh, transformation_matrix: np.ndarray) -> Mesh:
        """
        메쉬에 4x4 변환 행렬을 적용합니다.
        
        Args:
            mesh: 변환할 Mesh 객체
            transformation_matrix: 4x4 변환 행렬
            
        Returns:
            Mesh: 변환된 메쉬 (새 객체)
        """
        transformed_mesh = copy.deepcopy(mesh)
        
        # 동차 좌표로 변환
        vertices = transformed_mesh.vertices
        vertices_homogeneous = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
        
        # 변환 적용
        transformed_vertices = np.dot(vertices_homogeneous, transformation_matrix.T)
        transformed_mesh.vertices = transformed_vertices[:, :3]
        
        # 노말 벡터에는 회전만 적용 (이동 제외)
        if transformed_mesh.normals is not None:
            rotation_matrix = transformation_matrix[:3, :3]
            transformed_mesh.normals = np.dot(transformed_mesh.normals, rotation_matrix.T)
        
        return transformed_mesh
    
    @staticmethod
    def apply_transformation_inplace(mesh: Mesh, transformation_matrix: np.ndarray) -> None:
        """
        메쉬에 4x4 변환 행렬을 직접 적용합니다 (in-place).
        
        Args:
            mesh: 변환할 Mesh 객체
            transformation_matrix: 4x4 변환 행렬
        """
        # 동차 좌표로 변환
        vertices = mesh.vertices
        vertices_homogeneous = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
        
        # 변환 적용
        transformed_vertices = np.dot(vertices_homogeneous, transformation_matrix.T)
        mesh.vertices = transformed_vertices[:, :3]
        
        # 노말 벡터에는 회전만 적용 (이동 제외)
        if mesh.normals is not None:
            rotation_matrix = transformation_matrix[:3, :3]
            mesh.normals = np.dot(mesh.normals, rotation_matrix.T)
    
    @staticmethod
    def create_rotation_matrix_z(angle_radians: float) -> np.ndarray:
        """
        Z축 중심 회전 행렬을 생성합니다.
        
        Args:
            angle_radians: 회전 각도 (라디안)
            
        Returns:
            np.ndarray: 4x4 회전 행렬
        """
        cos_a = np.cos(angle_radians)
        sin_a = np.sin(angle_radians)
        
        return np.array([
            [cos_a, -sin_a, 0, 0],
            [sin_a, cos_a, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    
    @staticmethod
    def create_translation_matrix(translation: np.ndarray) -> np.ndarray:
        """
        이동 변환 행렬을 생성합니다.
        
        Args:
            translation: 이동 벡터 (shape: (3,))
            
        Returns:
            np.ndarray: 4x4 이동 행렬
        """
        matrix = np.eye(4)
        matrix[:3, 3] = translation
        return matrix
    
    @staticmethod
    def transform_to_global_coordinate_system(
        mesh: Mesh, 
        local_coordinate_system: np.ndarray
    ) -> Tuple[Mesh, np.ndarray]:
        """
        로컬 좌표계에서 글로벌 좌표계로 메쉬를 변환합니다.
        
        Args:
            mesh: 변환할 Mesh 객체
            local_coordinate_system: 3x3 로컬 좌표계 행렬
            
        Returns:
            tuple: (변환된 메쉬, 4x4 변환 행렬)
        """
        # 로컬 -> 글로벌 변환 행렬 계산
        rotation_matrix_3x3 = np.linalg.inv(local_coordinate_system)
        global_transform = np.eye(4)
        global_transform[:3, :3] = rotation_matrix_3x3.T
        
        # 변환 적용
        transformed_mesh = copy.deepcopy(mesh)
        transformed_mesh.vertices = np.dot(
            transformed_mesh.vertices, 
            global_transform[:3, :3].T
        ) + global_transform[:3, 3]
        
        return transformed_mesh, global_transform
    
    @staticmethod
    def translate_mesh(mesh: Mesh, translation: np.ndarray) -> Mesh:
        """
        메쉬를 이동시킵니다.
        
        Args:
            mesh: 이동할 Mesh 객체
            translation: 이동 벡터 (shape: (3,))
            
        Returns:
            Mesh: 이동된 메쉬 (새 객체)
        """
        translated_mesh = copy.deepcopy(mesh)
        translated_mesh.vertices = translated_mesh.vertices + translation
        return translated_mesh
    
    @staticmethod
    def translate_mesh_inplace(mesh: Mesh, translation: np.ndarray) -> None:
        """
        메쉬를 이동시킵니다 (in-place).
        
        Args:
            mesh: 이동할 Mesh 객체
            translation: 이동 벡터 (shape: (3,))
        """
        mesh.vertices = mesh.vertices + translation
    
    @staticmethod
    def apply_rotation_and_translation(
        mesh: Mesh, 
        rotation: np.ndarray, 
        translation: np.ndarray
    ) -> Mesh:
        """
        메쉬에 회전과 이동을 동시에 적용합니다.
        
        Args:
            mesh: 변환할 Mesh 객체
            rotation: 3x3 회전 행렬
            translation: 이동 벡터 (shape: (3,))
            
        Returns:
            Mesh: 변환된 메쉬 (새 객체)
        """
        transformed_mesh = copy.deepcopy(mesh)
        transformed_mesh.vertices = np.dot(
            transformed_mesh.vertices, 
            rotation.T
        ) + translation
        
        if transformed_mesh.normals is not None:
            transformed_mesh.normals = np.dot(transformed_mesh.normals, rotation.T)
        
        return transformed_mesh
    
    @staticmethod
    def apply_rotation_and_translation_inplace(
        mesh: Mesh, 
        rotation: np.ndarray, 
        translation: np.ndarray
    ) -> None:
        """
        메쉬에 회전과 이동을 동시에 적용합니다 (in-place).
        
        Args:
            mesh: 변환할 Mesh 객체
            rotation: 3x3 회전 행렬
            translation: 이동 벡터 (shape: (3,))
        """
        mesh.vertices = np.dot(mesh.vertices, rotation.T) + translation
        
        if mesh.normals is not None:
            mesh.normals = np.dot(mesh.normals, rotation.T)

