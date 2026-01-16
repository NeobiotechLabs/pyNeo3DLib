"""
중절치 정렬 모듈

이 모듈은 상악 전치부의 중절치 위치 기반 정렬을 담당합니다.
단일 책임 원칙(SRP)에 따라 중절치 정렬 로직만을 캡슐화합니다.
"""
import numpy as np
from typing import Tuple
from dataclasses import dataclass

from pyNeo3DLib.fileLoader.mesh import Mesh
from pyNeo3DLib.faceRegisration.mesh_cleaner import MeshCleaner
from pyNeo3DLib.faceRegisration.constants import IncisorAlignmentConstants


@dataclass
class IncisorAlignmentResult:
    """중절치 정렬 결과를 담는 데이터 클래스"""
    translation_vector: np.ndarray
    translation_matrix: np.ndarray


class IncisorAligner:
    """
    중절치 정렬을 담당하는 클래스.
    
    단일 책임: 상악 전치부의 중절치 위치 기반 정렬
    
    이 클래스는 다음 기능을 제공합니다:
    - 중절치 중심점 찾기
    - 정렬을 위한 이동 벡터 계산
    """
    
    def __init__(self):
        self._mesh_cleaner = MeshCleaner()
    
    def calculate_alignment_translation(
        self,
        target_mesh: Mesh,
        source_mesh: Mesh
    ) -> IncisorAlignmentResult:
        """
        소스 메쉬를 타겟 메쉬의 중절치 위치로 정렬하기 위한 이동 벡터를 계산합니다.
        
        Args:
            target_mesh: 타겟 메쉬 (라미네이트)
            source_mesh: 소스 메쉬 (변환할 메쉬)
            
        Returns:
            IncisorAlignmentResult: 정렬 결과 (이동 벡터 및 변환 행렬)
        """
        # 각 메쉬의 중절치 중심점 찾기
        target_center = self.find_central_incisor_center_point_for_second_icp(target_mesh)
        source_center = self.find_central_incisor_center_point_for_second_icp(source_mesh)
        
        # 이동 벡터 계산
        translation_vector = target_center - source_center
        
        # 4x4 이동 행렬 생성
        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = translation_vector
        
        return IncisorAlignmentResult(
            translation_vector=translation_vector,
            translation_matrix=translation_matrix
        )
    
    def find_central_incisor_center_point_for_second_icp(self, mesh: Mesh) -> np.ndarray:
        """
        상악 전치부 중절치 중심점을 추정합니다.
        
        메쉬를 x축 기준으로 클립한 후 z값이 가장 작은 정점을 찾고,
        x값을 0으로 설정하여 중절치 중심점을 추정합니다.
        
        Args:
            mesh: 중심점을 찾을 Mesh 객체
            
        Returns:
            np.ndarray: 추정된 중절치 중심점 좌표 (shape: (3,))
        """
        # x축 기준으로 메쉬 클립
        clipped_mesh = self._mesh_cleaner.clip_mesh_by_axis_range(
            mesh=mesh,
            axis=0,  # x축
            min_value=IncisorAlignmentConstants.X_AXIS_CLIP_MIN,
            max_value=IncisorAlignmentConstants.X_AXIS_CLIP_MAX,
            extract_largest=True
        )
        
        vertices = clipped_mesh.vertices
        
        # z값이 가장 작은 정점 찾기 (전치부 가장 앞쪽)
        min_z_idx = np.argmin(vertices[:, 2])
        center_point = vertices[min_z_idx].copy()
        
        # x값을 0으로 설정 (중앙 정렬, 중절치 중심점 x축 중앙 정렬을 위함)
        center_point[0] = 0
        
        return center_point
    
    def align_meshes_by_centroid(
        self,
        source_mesh: Mesh,
        target_mesh: Mesh
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        소스 메쉬의 중심점을 타겟 메쉬의 중심점으로 이동시키기 위한 변환을 계산합니다.
        
        Args:
            source_mesh: 소스 메쉬
            target_mesh: 타겟 메쉬
            
        Returns:
            tuple: (이동 벡터, 4x4 이동 행렬)
        """
        source_center = np.mean(source_mesh.vertices, axis=0)
        target_center = np.mean(target_mesh.vertices, axis=0)
        
        translation_vector = target_center - source_center
        
        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = translation_vector
        
        return translation_vector, translation_matrix

