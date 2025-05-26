import mediapipe as mp
from pyNeo3DLib.fileLoader.mesh import Mesh
from pyNeo3DLib.visualization.neovis import visualize_meshes
import os
import cv2
import open3d as o3d
import numpy as np
from scipy.spatial import KDTree
import time
import copy

class FaceMeshAnalyzer:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        
        # 왼쪽 : 234, 227 
        # 오른쪽 : 447, 454
        self.landmark_indices = [ 454, 227] #447, 234,

class CondyleFinder:
    def __init__(self, face_mesh_path, visualization=False):
        self.face_mesh_path = face_mesh_path
        self.visualization = visualization
        
        image_path = self.face_mesh_path.replace('.obj', '.png')
        if not os.path.exists(image_path):
            image_path = self.face_mesh_path.replace('.obj', '.jpg')
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: image version of {self.face_mesh_path}")
        self.face_image_path = image_path
        
        self.__load_model()
      
    def __load_model(self):
        self.face_mesh = Mesh.from_file(self.face_mesh_path)
        
    def _normalize_and_flip_coordinates(self, points, image_size):
        """
        Convert image coordinates to UV coordinates.
        
        Args:
            points: List of image coordinates [[x1, y1], [x2, y2], ...]
            image_size: Image size (width, height)
            
        Returns:
            List of UV coordinates [[u1, v1], [u2, v2], ...]
        """
        w, h = image_size
        uv_points = []
        
        for x, y in points:
            # Normalize (to 0~1 range)
            u = x / w
            v = y / h
            
            # Flip V coordinate (image coordinate system and UV coordinate system have opposite Y-axis direction)
            v = 1.0 - v
            
            uv_points.append([u, v])
        
        return uv_points

    def find_condyles_via_analyze_face_landmark(self):
        analyzer = FaceMeshAnalyzer()
        
        # Extract image file path (assuming image file has same name as obj file)
        image_path = self.face_mesh_path.replace('.obj', '.png')
        if not os.path.exists(image_path):
            image_path = self.face_mesh_path.replace('.obj', '.jpg')
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: image version of {self.face_mesh_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # Get image size
        h, w = image.shape[:2]
        
        # Convert image to RGB (MediaPipe requires RGB format)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect face landmarks
        results = analyzer.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            raise ValueError("Cannot detect face in the image.")
        
        # Get first face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract condyle landmarks
        self.condyle_landmarks = []
        for idx in analyzer.landmark_indices:
            landmark = face_landmarks.landmark[idx]
            # Convert image coordinates to pixel coordinates (0~1 range to pixel coordinates)
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            self.condyle_landmarks.append([x, y])
        
        # Convert to UV coordinates (normalize and flip coordinates)
        self.condyle_uv = self._normalize_and_flip_coordinates(self.condyle_landmarks, (w, h))
        
        self.condyle_uv[0][0] += 0.06
        self.condyle_uv[0][1] += 0.01
        self.condyle_uv[1][0] -= 0.06
        self.condyle_uv[1][1] += 0.01

        print(self.condyle_uv)
        
    def find_condyles_from_uv(self):
        """
        UV 좌표에서 콘딜 정점을 찾는 함수 (개선된 버전)
        - KDTree를 사용하여 검색 속도 개선
        - 각 입력 UV에 대해 정확히 하나의 정점 출력
        - condyle_mesh 객체 생성
        """
        start_time = time.time()
        print(f"콘딜 UV 좌표 {len(self.condyle_uv)}개 처리 중...")
        
        # 메시 데이터 가져오기
        vertices = self.face_mesh.vertices
        faces = self.face_mesh.faces
        uvs = self.face_mesh.uvs
        
        # UV 공간에서 KDTree 구축 (삼각형 중심점 기준)
        triangle_centers = []
        for i, face in enumerate(faces):
            face_uvs = uvs[face]
            center_uv = np.mean(face_uvs, axis=0)
            triangle_centers.append((center_uv, i))
        
        # KDTree 생성 (UV 좌표계 기준)
        tree_data = np.array([center[0] for center in triangle_centers])
        tree = KDTree(tree_data)
        
        # 결과 저장용 리스트
        condyle_vertices_indices = []
        condyle_vertices_3d = []
        
        # 각 콘딜 UV에 대해 가장 가까운 삼각형 및 정점 찾기
        for i, condyle_point in enumerate(self.condyle_uv):
            # KDTree로 가장 가까운 삼각형 찾기
            dist, idx = tree.query(condyle_point, k=1)
            triangle_idx = triangle_centers[idx][1]
            
            # 해당 삼각형의 정점들
            triangle_vertices = faces[triangle_idx]
            
            # 삼각형 내의 UV 좌표들
            triangle_uvs = uvs[triangle_vertices]
            
            # UV 공간에서 가중치 계산 (바리센트릭 좌표)
            weights = self._calculate_barycentric_weights(condyle_point, triangle_uvs)
            
            # 가중치에 따라 3D 좌표 계산
            interpolated_vertex = np.zeros(3)
            for w, v_idx in zip(weights, triangle_vertices):
                interpolated_vertex += w * vertices[v_idx]
            
            # 결과 저장
            closest_vertex_idx = triangle_vertices[np.argmax(weights)]  # 가중치가 가장 큰 정점 선택
            condyle_vertices_indices.append(closest_vertex_idx)
            condyle_vertices_3d.append(vertices[closest_vertex_idx])
        
        print(f'condyle_vertices_indices: {condyle_vertices_indices}')
        print(f'condyle_vertices_3d: {condyle_vertices_3d}')
        # 결과 저장
        self.condyle_vertices = copy.deepcopy(condyle_vertices_3d)
        
        self.condyle_vertices[0][0] -= 5
        self.condyle_vertices[1][0] += 5
        
        print(f'self.condyle_vertices: {self.condyle_vertices}')
        # condyle_mesh 객체 생성
        self.condyle_mesh = Mesh()
        self.condyle_mesh.vertices = np.array(self.condyle_vertices)
        
        # 구 또는 점으로 표현 (시각화용)
        radius = 3  # 적절한 크기 설정
        sphere_vertices, sphere_faces = self._create_sphere_representation(self.condyle_vertices, radius)
        self.condyle_mesh.vertices = sphere_vertices
        self.condyle_mesh.faces = sphere_faces
        
        elapsed_time = time.time() - start_time
        print(f"콘딜 정점 {len(self.condyle_vertices)}개 찾기 완료 (소요시간: {elapsed_time:.3f}초)")
        
    def _calculate_barycentric_weights(self, point, triangle_uvs):
        """삼각형에 대한 바리센트릭 좌표 계산"""
        v0, v1, v2 = triangle_uvs
        
        # 삼각형의 각 변을 계산
        v0v1 = v1 - v0
        v0v2 = v2 - v0
        point_v0 = np.array(point) - v0
        
        # 내적 계산
        d00 = np.dot(v0v1, v0v1)
        d01 = np.dot(v0v1, v0v2)
        d11 = np.dot(v0v2, v0v2)
        d20 = np.dot(point_v0, v0v1)
        d21 = np.dot(point_v0, v0v2)
        
        # 크래머 공식을 사용하여 가중치 계산
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-10:
            return [1/3, 1/3, 1/3]  # 삼각형이 너무 작으면 균등 가중치 반환
        
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        
        return [u, v, w]

    def _create_sphere_representation(self, points, radius):
        """점을 작은 구로 표현하기 위한 함수 (Open3D 사용)"""
        all_vertices = []
        all_faces = []
        vertex_offset = 0
        
        for point in points:
            # Open3D 구 메시 생성
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=10)
            
            # 중심점으로 이동
            sphere.translate(point)
            
            # 메시 데이터 추출
            sphere_vertices = np.asarray(sphere.vertices)
            sphere_faces = np.asarray(sphere.triangles)
            
            # 정점 오프셋 적용
            sphere_faces = sphere_faces + vertex_offset
            
            # 결과 누적
            all_vertices.extend(sphere_vertices)
            all_faces.extend(sphere_faces)
            vertex_offset += len(sphere_vertices)
        
        return np.array(all_vertices), np.array(all_faces)
        
    def run_analysis(self):
        self.find_condyles_via_analyze_face_landmark()
        self.find_condyles_from_uv()
        
        if self.visualization:
            visualize_meshes([self.face_mesh, self.condyle_mesh], ["Face", "Condyle"], title="Condyle Detection")
            
        return self.condyle_vertices
            

if __name__ == "__main__":
    condyle_finder = CondyleFinder(face_mesh_path="../../example/data/FaceScan/Smile/Smile.obj", visualization=True)
    condyle_finder.run_analysis()
