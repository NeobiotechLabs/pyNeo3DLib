"""
텍스처 기반 메쉬 추출 모듈

이 모듈은 텍스처 정보를 기반으로 메쉬의 특정 영역을 추출하는 기능을 담당합니다.
단일 책임 원칙(SRP)에 따라 텍스처 기반 추출 로직만을 캡슐화합니다.
"""
import numpy as np
import cv2
import os

from pyNeo3DLib.fileLoader.mesh import Mesh
from pyNeo3DLib.faceRegisration.constants import TextureConstants


class TextureMeshExtractor:
    """
    텍스처 기반 메쉬 추출을 담당하는 클래스.
    
    단일 책임: 텍스처의 특정 영역(투명, 특정 색상 등)에 해당하는 메쉬 추출
    
    이 클래스는 다음 기능을 제공합니다:
    - 투명 영역 메쉬 추출
    - UV 좌표 기반 메쉬 분리
    """
    
    def extract_transparent_region(
        self, 
        mesh: Mesh, 
        texture_image: np.ndarray = None,
        texture_path: str = None
    ) -> Mesh:
        """
        텍스처에서 투명한 영역(알파 채널이 0인 곳)에 해당하는 메쉬를 추출합니다.
        
        Args:
            mesh: 처리할 Mesh 객체
            texture_image: 텍스처 이미지 (numpy.ndarray)
            texture_path: 텍스처 파일 경로 (texture_image가 None인 경우 사용)
            
        Returns:
            Mesh: 투명한 영역의 메쉬, 없으면 None
        """
        # 텍스처 이미지 로드
        if texture_image is None:
            texture_image = self._load_texture(texture_path)
            if texture_image is None:
                return None
        
        img_height, img_width = texture_image.shape[:2]
        
        # 알파 채널 추출
        alpha_channel = self._extract_alpha_channel(texture_image)
        
        # UV 좌표 검증
        if not self._validate_uvs(mesh):
            return None
        
        # 투명 정점 식별
        vertex_is_transparent = self._identify_transparent_vertices(
            mesh, alpha_channel, img_width, img_height
        )
        
        print(f"투명한 정점 수: {np.sum(vertex_is_transparent)} / {len(mesh.vertices)}")
        
        # 투명 영역 메쉬 생성
        transparent_mesh = self._create_transparent_mesh(mesh, vertex_is_transparent)
        
        return transparent_mesh
    
    def _load_texture(self, texture_path: str) -> np.ndarray:
        """텍스처 이미지를 로드합니다."""
        if texture_path is None:
            print("텍스처 경로가 제공되지 않았습니다.")
            return None
        
        if not os.path.exists(texture_path):
            # .png 또는 .jpg 확장자로 시도
            base_path = os.path.splitext(texture_path)[0]
            for ext in ['.png', '.jpg', '.jpeg']:
                alt_path = base_path + ext
                if os.path.exists(alt_path):
                    texture_path = alt_path
                    break
            else:
                print(f"텍스처 파일을 찾을 수 없습니다: {texture_path}")
                return None
        
        texture_image = cv2.imread(texture_path, cv2.IMREAD_UNCHANGED)
        if texture_image is None:
            print(f"텍스처 이미지를 로드할 수 없습니다: {texture_path}")
            return None
        
        return texture_image
    
    def _extract_alpha_channel(self, texture_image: np.ndarray) -> np.ndarray:
        """텍스처에서 알파 채널을 추출합니다."""
        if len(texture_image.shape) >= 3 and texture_image.shape[2] == 4:
            return texture_image[:, :, 3]
        else:
            print("텍스처에 알파 채널이 없습니다. RGB 기준으로 검정색을 투명으로 처리합니다.")
            if len(texture_image.shape) == 2:
                gray = texture_image
            else:
                gray = cv2.cvtColor(texture_image, cv2.COLOR_BGR2GRAY)
            return np.where(gray < TextureConstants.BLACK_THRESHOLD, 0, 255).astype(np.uint8)
    
    def _validate_uvs(self, mesh: Mesh) -> bool:
        """메쉬의 UV 좌표 유효성을 검증합니다."""
        if not hasattr(mesh, 'uvs') or mesh.uvs is None or len(mesh.uvs) == 0:
            print("메시에 UV 좌표가 없어서 투명 영역을 분리할 수 없습니다.")
            return False
        return True
    
    def _identify_transparent_vertices(
        self, 
        mesh: Mesh, 
        alpha_channel: np.ndarray,
        img_width: int,
        img_height: int
    ) -> np.ndarray:
        """투명한 영역에 있는 정점들을 식별합니다."""
        uvs = np.asarray(mesh.uvs, dtype=np.float32)
        vertex_is_transparent = np.zeros(len(mesh.vertices), dtype=bool)
        
        for i, uv in enumerate(uvs):
            u, v = uv
            u = np.clip(u, 0, 1)
            v = np.clip(v, 0, 1)
            
            px = int(u * (img_width - 1))
            py = int((1.0 - v) * (img_height - 1))
            
            if alpha_channel[py, px] < TextureConstants.ALPHA_THRESHOLD:
                vertex_is_transparent[i] = True
        
        return vertex_is_transparent
    
    def _create_transparent_mesh(
        self, 
        mesh: Mesh, 
        vertex_is_transparent: np.ndarray
    ) -> Mesh:
        """투명한 정점들로 새로운 메쉬를 생성합니다."""
        uvs = np.asarray(mesh.uvs, dtype=np.float32)
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        
        # 모든 정점이 투명한 face만 선택
        transparent_face_indices = []
        for i, face in enumerate(faces):
            if all(vertex_is_transparent[v_idx] for v_idx in face):
                transparent_face_indices.append(i)
        
        if len(transparent_face_indices) == 0:
            print("투명한 영역에 해당하는 face가 없습니다.")
            return None
        
        print(f"투명한 face 수: {len(transparent_face_indices)} / {len(faces)}")
        
        # 사용되는 정점만 추출
        transparent_faces = faces[transparent_face_indices]
        used_vertex_indices = np.unique(transparent_faces.flatten())
        
        # 인덱스 리맵핑
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertex_indices)}
        
        # 새로운 메쉬 생성
        transparent_mesh = Mesh()
        transparent_mesh.vertices = vertices[used_vertex_indices]
        transparent_mesh.faces = np.array(
            [[index_map[v_idx] for v_idx in face] for face in transparent_faces],
            dtype=np.int32
        )
        
        # UV 좌표 복사
        if hasattr(mesh, 'uvs') and mesh.uvs is not None:
            transparent_mesh.uvs = uvs[used_vertex_indices]
        
        # 노멀 복사
        if hasattr(mesh, 'normals') and mesh.normals is not None:
            transparent_mesh.normals = mesh.normals[used_vertex_indices]
        
        print(f"투명 영역 메시 생성 완료: 정점 {len(transparent_mesh.vertices)}개, face {len(transparent_mesh.faces)}개")
        
        return transparent_mesh

