"""
메시 데이터 준비 모듈

IOS 메시와 Smile Arch 메시의 데이터를 준비하는 클래스입니다.
"""

import numpy as np
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from pyNeo3DLib.fileLoader.mesh import Mesh


class MeshDataPreparer:
    """
    메시 데이터를 준비하는 클래스
    
    IOS 메시와 Smile Arch 메시의 정점, 면 정보를 추출하고
    필요한 변환을 적용합니다.
    """
    
    def prepare(
        self,
        ios_mesh: "Mesh",
        smile_arch_mesh: "Mesh",
        ios_laminate_result: np.ndarray,
        mesh_type: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        메시 데이터를 준비합니다.
        
        Args:
            ios_mesh: IOS 메시 객체
            smile_arch_mesh: Smile Arch 메시 객체
            ios_laminate_result: IOS Laminate 변환 행렬
            mesh_type: 메시 타입 ("Upper" 또는 "Lower")
            
        Returns:
            Tuple[ios_vertices, ios_faces, smile_arch_centroid]
        """
        ios_vertices = ios_mesh.vertices
        ios_faces = ios_mesh.faces
        smile_arch_vertices = smile_arch_mesh.vertices
        
        # Smile Arch 변환 적용
        smile_arch_vertices = np.dot(
            smile_arch_vertices,
            ios_laminate_result[:3, :3].T
        ) + ios_laminate_result[:3, 3]
        
        smile_arch_centroid = np.mean(smile_arch_vertices, axis=0)
        
        print(f"[INFO] IOS {mesh_type} mesh: {ios_vertices.shape[0]} vertices")
        print(f"[INFO] Smile Arch mesh: {smile_arch_vertices.shape[0]} vertices")
        
        return ios_vertices, ios_faces, smile_arch_centroid
