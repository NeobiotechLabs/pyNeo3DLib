"""
Face Lip Extractor - MediaPipe 기반 3D 입술 영역 추출

PLY/OBJ 3D 모델에서 텍스처 이미지를 이용해 MediaPipe Face Mesh로 
입술 랜드마크를 감지하고, UV 매핑을 통해 3D 메시 정점을 추출합니다.
"""

import struct
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# MediaPipe 0.10.30+ 새로운 API 사용
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision


class FaceLipExtractor:
    """MediaPipe를 사용하여 3D 얼굴 스캔에서 입술 영역을 추출하는 클래스"""
    
    # MediaPipe Face Mesh 내부 입술 랜드마크 인덱스 (20개)
    LIPS_INNER = [
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
        308, 324, 318, 402, 317, 14, 87, 178, 88, 95
    ]
    
    # 외부 입술 랜드마크 (참고용)
    LIPS_OUTER = [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        291, 409, 270, 269, 267, 0, 37, 39, 40, 185
    ]
    
    # MediaPipe model path (다운로드 필요)
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    
    def __init__(self, model_path: Optional[str] = None):
        """MediaPipe Face Landmarker 초기화
        
        Args:
            model_path: face_landmarker.task 모델 파일 경로. None이면 자동 다운로드 시도.
        """
        self.model_path = model_path
        self._detector = None
        self._ensure_model()
    
    def _ensure_model(self):
        """모델 파일 확인 및 다운로드, detector 초기화"""
        import urllib.request
        import os
        
        # 기본 모델 경로 설정
        if self.model_path is None:
            model_dir = Path(__file__).parent / 'models'
            model_dir.mkdir(exist_ok=True)
            self.model_path = str(model_dir / 'face_landmarker.task')
        
        # 모델 파일이 없으면 다운로드
        if not Path(self.model_path).exists():
            print(f"Downloading face_landmarker model to {self.model_path}...")
            urllib.request.urlretrieve(self.MODEL_URL, self.model_path)
            print("Download complete.")
        
        # FaceLandmarker 초기화
        # Windows 경로 문제 방지를 위해 파일을 직접 읽어서 버퍼로 전달
        try:
            with open(self.model_path, 'rb') as f:
                model_content = f.read()
            base_options = mp_tasks.BaseOptions(model_asset_buffer=model_content)
        except Exception as e:
            print(f"Failed to read model file: {e}")
            # fallback
            base_options = mp_tasks.BaseOptions(model_asset_path=self.model_path)

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self._detector = vision.FaceLandmarker.create_from_options(options)
    
    def parse_ply(self, path: str) -> Dict:
        """
        Binary PLY 파일 파싱
        
        Args:
            path: PLY 파일 경로
            
        Returns:
            dict: {
                'vertices': np.ndarray (N, 3),
                'normals': np.ndarray (N, 3),
                'uvs': np.ndarray (N, 2),
                'faces': np.ndarray (M, 3) - 있는 경우
            }
        """
        with open(path, 'rb') as f:
            # 헤더 파싱
            header_lines = []
            while True:
                line = f.readline().decode('ascii').strip()
                header_lines.append(line)
                if line == 'end_header':
                    break
            
            # 헤더 분석
            num_vertices = 0
            num_faces = 0
            vertex_properties = []
            
            for line in header_lines:
                if line.startswith('element vertex'):
                    num_vertices = int(line.split()[-1])
                elif line.startswith('element face'):
                    num_faces = int(line.split()[-1])
                elif line.startswith('property'):
                    parts = line.split()
                    if len(parts) >= 3:
                        vertex_properties.append((parts[1], parts[2]))
            
            # 정점 데이터 읽기
            # 예상 형식: x, y, z, nx, ny, nz, texture_u, texture_v
            vertex_format = ''
            for prop_type, prop_name in vertex_properties:
                if prop_type == 'float':
                    vertex_format += 'f'
                elif prop_type == 'double':
                    vertex_format += 'd'
                elif prop_type in ['uchar', 'uint8']:
                    vertex_format += 'B'
            
            vertex_size = struct.calcsize(vertex_format)
            
            vertices = np.zeros((num_vertices, 3), dtype=np.float32)
            normals = np.zeros((num_vertices, 3), dtype=np.float32)
            uvs = np.zeros((num_vertices, 2), dtype=np.float32)
            
            for i in range(num_vertices):
                data = struct.unpack(vertex_format, f.read(vertex_size))
                vertices[i] = data[0:3]
                if len(data) >= 6:
                    normals[i] = data[3:6]
                if len(data) >= 8:
                    uvs[i] = data[6:8]
            
            # 면 데이터 읽기 (있는 경우)
            faces = []
            for _ in range(num_faces):
                # 면 정점 수 읽기
                n_verts = struct.unpack('B', f.read(1))[0]
                face_indices = struct.unpack(f'{n_verts}i', f.read(4 * n_verts))
                if n_verts == 3:
                    faces.append(face_indices)
            
            result = {
                'vertices': vertices,
                'normals': normals,
                'uvs': uvs,
            }
            
            if faces:
                result['faces'] = np.array(faces, dtype=np.int32)
            
            return result
    
    def parse_obj(self, path: str) -> Dict:
        """
        OBJ 파일 파싱
        
        Args:
            path: OBJ 파일 경로
            
        Returns:
            dict: {
                'vertices': np.ndarray (N, 3),
                'uvs': np.ndarray (M, 2),
                'faces': list of tuples (vertex_indices, uv_indices),
                'vertex_uvs': np.ndarray (N, 2) - 정점별 UV 매핑
            }
        """
        vertices = []
        uvs = []
        faces = []
        
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if not parts:
                    continue
                
                if parts[0] == 'v':
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif parts[0] == 'vt':
                    uvs.append([float(parts[1]), float(parts[2])])
                elif parts[0] == 'f':
                    face_v = []
                    face_vt = []
                    for vert in parts[1:]:
                        indices = vert.split('/')
                        face_v.append(int(indices[0]) - 1)  # OBJ는 1-indexed
                        if len(indices) > 1 and indices[1]:
                            face_vt.append(int(indices[1]) - 1)
                    faces.append((face_v, face_vt))
        
        vertices_arr = np.array(vertices, dtype=np.float32)
        uvs_arr = np.array(uvs, dtype=np.float32) if uvs else np.zeros((0, 2), dtype=np.float32)
        
        # 정점별 UV 매핑 생성
        vertex_uvs = np.zeros((len(vertices), 2), dtype=np.float32)
        for face_v, face_vt in faces:
            for v_idx, vt_idx in zip(face_v, face_vt):
                if vt_idx < len(uvs_arr):
                    vertex_uvs[v_idx] = uvs_arr[vt_idx]
        
        return {
            'vertices': vertices_arr,
            'uvs': uvs_arr,
            'faces': faces,
            'vertex_uvs': vertex_uvs,
        }
    
    def get_texture_image(self, folder_path: str) -> Optional[np.ndarray]:
        """
        폴더에서 텍스처 PNG 이미지 로드
        
        Args:
            folder_path: 스캔 데이터 폴더 경로
            
        Returns:
            np.ndarray: BGR 이미지 또는 None
        """
        folder = Path(folder_path)
        
        # PNG 파일 찾기
        png_files = list(folder.glob('*.png'))
        if not png_files:
            return None
        
        # 첫 번째 PNG 파일 로드
        image = cv2.imread(str(png_files[0]))
        return image
    
    def detect_face_landmarks(self, image: np.ndarray) -> Optional[List]:
        """
        이미지에서 얼굴 랜드마크 감지
        
        Args:
            image: BGR 이미지
            
        Returns:
            list: 정규화된 랜드마크 좌표 [(x, y, z), ...] 또는 None
        """
        # BGR -> RGB 변환
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # MediaPipe Image로 변환
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # 얼굴 랜드마크 감지
        result = self._detector.detect(mp_image)
        
        if not result.face_landmarks:
            return None
        
        # 첫 번째 얼굴의 랜드마크 반환 (정규화된 좌표)
        landmarks = result.face_landmarks[0]
        return [(lm.x, lm.y, lm.z) for lm in landmarks]
    
    def get_lip_landmarks(self, landmarks: List, use_outer: bool = False) -> np.ndarray:
        """
        입술 랜드마크의 정규화된 UV 좌표 추출
        
        Args:
            landmarks: 전체 얼굴 랜드마크 리스트
            use_outer: True면 외부 입술 사용, False면 내부 입술
            
        Returns:
            np.ndarray: 입술 랜드마크 UV 좌표 (N, 2)
        """
        indices = self.LIPS_OUTER if use_outer else self.LIPS_INNER
        lip_coords = []
        
        for idx in indices:
            if idx < len(landmarks):
                x, y, _ = landmarks[idx]
                # MediaPipe는 이미지 좌표(Y 아래로 증가), 
                # 3D 모델 UV는 텍스처 좌표(V 위로 증가)이므로 V = 1 - y
                lip_coords.append([x, 1.0 - y])
        
        return np.array(lip_coords, dtype=np.float32)
    
    def find_matching_vertices(
        self, 
        lip_uv_coords: np.ndarray, 
        mesh_uvs: np.ndarray, 
        threshold: float = 0.02
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        입술 랜드마크 다각형 안쪽의 메시 정점 인덱스 찾기
        
        Args:
            lip_uv_coords: 입술 랜드마크 UV 좌표 (N, 2) - 폐곡선 형성
            mesh_uvs: 메시 UV 좌표 (M, 2)
            threshold: 외곽선 정점 매칭용 임계값
            
        Returns:
            tuple: (다각형 내부에 있는 정점 인덱스 배열, 각 입술 랜드마크에 대한 최근접 정점 인덱스)
        """
        from matplotlib.path import Path
        
        # 1. 입술 랜드마크로 다각형 경로 생성
        lip_polygon = Path(lip_uv_coords)
        
        # 2. 다각형 내부에 있는 모든 메시 정점 찾기
        inside_mask = lip_polygon.contains_points(mesh_uvs)
        inside_indices = np.where(inside_mask)[0]
        
        # 3. 각 랜드마크에 대한 최근접 정점 찾기 (외곽선용)
        nearest_indices = []
        for lip_uv in lip_uv_coords:
            distances = np.linalg.norm(mesh_uvs - lip_uv, axis=1)
            nearest_idx = np.argmin(distances)
            nearest_indices.append(nearest_idx)
        
        return inside_indices.astype(np.int32), np.array(nearest_indices, dtype=np.int32)
    
    def extract_lip_boundary(
        self, 
        vertices: np.ndarray, 
        vertex_indices: np.ndarray
    ) -> np.ndarray:
        """
        입술 영역의 외곽선 정점 좌표 추출
        
        입술 랜드마크 순서대로 정점을 반환하여 폐곡선을 형성
        
        Args:
            vertices: 전체 메시 정점 (N, 3)
            vertex_indices: 입술 랜드마크에 대응하는 정점 인덱스 (순서대로)
            
        Returns:
            np.ndarray: 외곽선 정점 좌표 (M, 3)
        """
        return vertices[vertex_indices]
    
    def visualize_result(
        self,
        texture_image: np.ndarray,
        lip_uv_coords: np.ndarray,
        boundary_vertices: np.ndarray,
        all_lip_vertices: np.ndarray,
        title: str = "Lip Extraction Result"
    ):
        """
        추출 결과 시각화
        
        Args:
            texture_image: 텍스처 이미지 (BGR)
            lip_uv_coords: 입술 랜드마크 UV 좌표
            boundary_vertices: 외곽선 정점 좌표
            all_lip_vertices: 입술 영역 전체 정점
            title: 그래프 제목
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(14, 6))
        
        # 1. 텍스처 이미지 + 입술 랜드마크
        ax1 = fig.add_subplot(1, 2, 1)
        rgb_image = cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB)
        ax1.imshow(rgb_image)
        
        # 랜드마크 포인트 그리기 (이미지 좌표로 변환)
        # UV 좌표에서 다시 이미지 좌표로: V를 다시 뒤집음 (y = 1 - v)
        h, w = texture_image.shape[:2]
        lip_points_px = np.column_stack([
            lip_uv_coords[:, 0] * w,  # U -> X
            (1.0 - lip_uv_coords[:, 1]) * h  # V -> Y (flip back)
        ])
        
        # 입술 외곽선 연결
        lip_points_closed = np.vstack([lip_points_px, lip_points_px[0]])
        ax1.plot(lip_points_closed[:, 0], lip_points_closed[:, 1], 'r-', linewidth=2, label='Lip Boundary')
        ax1.scatter(lip_points_px[:, 0], lip_points_px[:, 1], c='yellow', s=30, zorder=5, edgecolors='red')
        
        ax1.set_title('Texture Image - Lip Landmarks')
        ax1.axis('off')
        ax1.legend()
        
        # 2. 3D 입술 메시
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        
        # 전체 입술 영역 점
        ax2.scatter(
            all_lip_vertices[:, 0],
            all_lip_vertices[:, 1],
            all_lip_vertices[:, 2],
            c='lightblue', s=1, alpha=0.3, label='Lip Region'
        )
        
        # 외곽선 정점
        ax2.scatter(
            boundary_vertices[:, 0],
            boundary_vertices[:, 1],
            boundary_vertices[:, 2],
            c='red', s=50, label='Boundary Vertices'
        )
        
        # 외곽선 연결
        boundary_closed = np.vstack([boundary_vertices, boundary_vertices[0]])
        ax2.plot(
            boundary_closed[:, 0],
            boundary_closed[:, 1],
            boundary_closed[:, 2],
            'r-', linewidth=2
        )
        
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title('3D Lip Mesh')
        ax2.legend()
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def extract_lip_region(self, folder_path: str, use_outer: bool = False, visualize: bool = False) -> Optional[Dict]:
        """
        폴더에서 3D 모델을 로드하고 입술 영역 추출
        
        Args:
            folder_path: 스캔 데이터 폴더 경로
            use_outer: True면 외부 입술 사용, False면 내부 입술
            visualize: True면 추출 결과 시각화
            
        Returns:
            dict: {
                'boundary_vertices': np.ndarray (N, 3) - 외곽선 정점 좌표 (순서대로),
                'all_lip_vertices': np.ndarray (M, 3) - 입술 영역 전체 정점,
                'vertex_indices': np.ndarray - 외곽선 정점의 메시 인덱스,
                'all_vertex_indices': np.ndarray - 입술 영역 전체 정점 인덱스
            }
        """
        folder = Path(folder_path)
        
        # 1. 3D 모델 파일 찾기 및 파싱
        ply_files = list(folder.glob('*.ply'))
        obj_files = list(folder.glob('*.obj'))
        
        mesh_data = None
        if ply_files:
            mesh_data = self.parse_ply(str(ply_files[0]))
            mesh_uvs = mesh_data['uvs']
        elif obj_files:
            mesh_data = self.parse_obj(str(obj_files[0]))
            mesh_uvs = mesh_data['vertex_uvs']
        else:
            print(f"No PLY or OBJ files found in {folder_path}")
            return None
        
        vertices = mesh_data['vertices']
        
        # 2. 텍스처 이미지 로드
        texture_image = self.get_texture_image(folder_path)
        if texture_image is None:
            print(f"No texture image found in {folder_path}")
            return None
        
        # 3. MediaPipe로 얼굴 랜드마크 감지
        landmarks = self.detect_face_landmarks(texture_image)
        if landmarks is None:
            print("No face detected in texture image")
            return None
        
        # 4. 입술 UV 좌표 추출
        lip_uv_coords = self.get_lip_landmarks(landmarks, use_outer)
        
        # 5. UV 좌표로 3D 정점 매칭
        all_matched_indices, boundary_indices = self.find_matching_vertices(
            lip_uv_coords, mesh_uvs, threshold=0.015
        )
        
        # 6. 결과 생성
        result = {
            'boundary_vertices': self.extract_lip_boundary(vertices, boundary_indices),
            'all_lip_vertices': vertices[all_matched_indices],
            'vertex_indices': boundary_indices,
            'all_vertex_indices': all_matched_indices,
        }
        
        # 7. 시각화 (옵션)
        if visualize:
            self.visualize_result(
                texture_image=texture_image,
                lip_uv_coords=lip_uv_coords,
                boundary_vertices=result['boundary_vertices'],
                all_lip_vertices=result['all_lip_vertices'],
                title=f"Lip Extraction - {Path(folder_path).name}"
            )
        
        return result
    
    def close(self):
        """MediaPipe 리소스 해제"""
        if self._detector:
            self._detector.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# 편의 함수
def extract_lip_from_folder(folder_path: str, use_outer: bool = False) -> Optional[Dict]:
    """
    폴더에서 입술 영역 추출 (편의 함수)
    
    Args:
        folder_path: 스캔 데이터 폴더 경로
        use_outer: True면 외부 입술 사용
        
    Returns:
        dict: 입술 영역 정보
    """
    with FaceLipExtractor() as extractor:
        return extractor.extract_lip_region(folder_path, use_outer)


if __name__ == '__main__':
    # 테스트 실행
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Face Lip Extractor Test')
    parser.add_argument('--visualize', '-v', action='store_true', help='시각화 활성화')
    parser.add_argument('--sample', '-s', type=int, default=None, help='특정 샘플만 테스트 (1, 2, 3)')
    args = parser.parse_args()
    
    samples = [
        './3dmodel/facescan/sample_1',
        './3dmodel/facescan/sample_2',
        './3dmodel/facescan/sample_3',
    ]
    
    # 특정 샘플만 테스트
    if args.sample:
        samples = [f'./3dmodel/facescan/sample_{args.sample}']
    
    print("Face Lip Extractor Test")
    print("=" * 50)
    
    with FaceLipExtractor() as extractor:
        for sample in samples:
            print(f"\nProcessing: {sample}")
            result = extractor.extract_lip_region(sample, visualize=args.visualize)
            
            if result:
                print(f"  외곽선 정점 수: {len(result['boundary_vertices'])}")
                print(f"  입술 영역 전체 정점 수: {len(result['all_lip_vertices'])}")
                print(f"  정점 인덱스 범위: {result['vertex_indices'].min()} ~ {result['vertex_indices'].max()}")
                
                # 외곽선 정점 좌표 출력 (처음 5개)
                print("  외곽선 정점 좌표 (처음 5개):")
                for i, v in enumerate(result['boundary_vertices'][:5]):
                    print(f"    [{i}] ({v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f})")
            else:
                print("  추출 실패")

