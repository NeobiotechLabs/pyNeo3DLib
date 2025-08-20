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

class GoldenProportionFinder:
    def __init__(self, face_mesh_path, visualization=False):
        self.face_mesh_path = face_mesh_path
        self.visualization = visualization
        
        # 파일 확장자에 따라 이미지 파일 경로 생성
        base_path = self.face_mesh_path.rsplit('.', 1)[0]  # 확장자 제거
        
        # PNG 파일 먼저 확인
        image_path = base_path + '.png'
        if not os.path.exists(image_path):
            # JPG 파일 확인
            image_path = base_path + '.jpg'
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: image version of {self.face_mesh_path}")
        self.face_image_path = image_path
        
        # MediaPipe 초기화
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh_detector = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        
        # 찾을 4개의 랜드마크 인덱스 정의 (MediaPipe 얼굴 메시는 0~467 인덱스)
        # A: 두 눈의 중심 (33번, 263번의 중심점 - 눈 안쪽 모서리)
        # B: 코 높은점 (4번)
        # C: 입꼴리 중점 (61번, 291번의 중심점 - 입꼴리)
        # D: 턱 (18번 - 턱 아래)
        self.landmark_indices = {
            'left_eye_inner': 33,   # 왼쪽 눈 안쪽 모서리
            'right_eye_inner': 263, # 오른쪽 눈 안쪽 모서리  
            'nose_tip': 19,          # 코 끝
            'left_mouth_corner': 61,  # 왼쪽 입꼴리
            'right_mouth_corner': 291, # 오른쪽 입꼴리
            'chin': 199              # 턱 아래
        }
        
        self.__load_model()
        
    def __load_model(self):
        self.face_mesh = Mesh.from_file(self.face_mesh_path)
        
    def _normalize_and_flip_coordinates(self, points, image_size):
        """
        이미지 좌표를 UV 좌표로 변환.
        
        Args:
            points: 이미지 좌표 리스트 [[x1, y1], [x2, y2], ...]
            image_size: 이미지 크기 (width, height)
            
        Returns:
            UV 좌표 리스트 [[u1, v1], [u2, v2], ...]
        """
        w, h = image_size
        uv_points = []
        
        for x, y in points:
            # 정규화 (0~1 범위로)
            u = x / w
            v = y / h
            
            # V 좌표 뒤집기 (이미지 좌표계와 UV 좌표계는 Y축 방향이 반대)
            v = 1.0 - v
            
            uv_points.append([u, v])
        
        return uv_points
    
    def find_golden_proportion_landmarks(self):
        """
        얼굴 랜드마크를 분석하여 4개의 황금비율 점을 찾는 함수
        """
        # 이미지 로드
        image = cv2.imread(self.face_image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {self.face_image_path}")
        
        # 이미지 크기 가져오기
        h, w = image.shape[:2]
        
        # RGB로 변환 (MediaPipe는 RGB 형식 필요)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 얼굴 랜드마크 감지
        results = self.face_mesh_detector.process(image_rgb)
        
        if not results.multi_face_landmarks:
            raise ValueError("Cannot detect face in the image.")
        
        # 첫 번째 얼굴의 랜드마크 가져오기
        face_landmarks = results.multi_face_landmarks[0]
        
        # 4개의 특정 점 추출
        landmark_points = []
        
        # A: 두 눈의 중심점 (33번, 263번의 중심)
        left_eye = face_landmarks.landmark[self.landmark_indices['left_eye_inner']]
        right_eye = face_landmarks.landmark[self.landmark_indices['right_eye_inner']]
        eye_center_x = int((left_eye.x + right_eye.x) * w / 2)
        eye_center_y = int((left_eye.y + right_eye.y) * h / 2)
        landmark_points.append([eye_center_x, eye_center_y])
        
        # B: 코 높은점 (4번)
        nose_tip = face_landmarks.landmark[self.landmark_indices['nose_tip']]
        nose_x = int(nose_tip.x * w)
        nose_y = int(nose_tip.y * h)
        landmark_points.append([nose_x, nose_y])
        
        # C: 입꼬리 중점 (61번, 291번의 중심)
        left_mouth = face_landmarks.landmark[self.landmark_indices['left_mouth_corner']]
        right_mouth = face_landmarks.landmark[self.landmark_indices['right_mouth_corner']]
        mouth_center_x = int((left_mouth.x + right_mouth.x) * w / 2)
        mouth_center_y = int((left_mouth.y + right_mouth.y) * h / 2)
        landmark_points.append([mouth_center_x, mouth_center_y])
        
        # D: 턱 (18번)
        chin = face_landmarks.landmark[self.landmark_indices['chin']]
        chin_x = int(chin.x * w)
        chin_y = int(chin.y * h)
        landmark_points.append([chin_x, chin_y])
        
        # 이미지 좌표를 저장
        self.landmark_image_points = landmark_points
        
        # UV 좌표로 변환
        self.landmark_uv = self._normalize_and_flip_coordinates(landmark_points, (w, h))
        
        print(f"Golden Proportion Landmarks found:")
        print(f"  A (Eye Center): {landmark_points[0]} -> UV: {self.landmark_uv[0]}")
        print(f"  B (Nose Tip): {landmark_points[1]} -> UV: {self.landmark_uv[1]}")
        print(f"  C (Mouth Center): {landmark_points[2]} -> UV: {self.landmark_uv[2]}")
        print(f"  D (Chin): {landmark_points[3]} -> UV: {self.landmark_uv[3]}")
        
        return self.landmark_uv
        
    def find_3d_points_from_uv(self):
        """
        UV 좌표에서 3D 점을 찾는 함수 (개선된 버전)
        - KDTree를 사용하여 검색 속도 개선
        - 각 입력 UV에 대해 정확히 하나의 정점 출력
        """
        start_time = time.time()
        print(f"황금비율 UV 좌표 {len(self.landmark_uv)}개 처리 중...")
        
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
        landmark_vertices_3d = []
        
        # 각 랜드마크 UV에 대해 가장 가까운 삼각형 및 정점 찾기
        for i, landmark_point in enumerate(self.landmark_uv):
            # KDTree로 가장 가까운 삼각형 찾기
            dist, idx = tree.query(landmark_point, k=1)
            triangle_idx = triangle_centers[idx][1]
            
            # 해당 삼각형의 정점들
            triangle_vertices = faces[triangle_idx]
            
            # 삼각형 내의 UV 좌표들
            triangle_uvs = uvs[triangle_vertices]
            
            # UV 공간에서 가중치 계산 (바리센트릭 좌표)
            weights = self._calculate_barycentric_weights(landmark_point, triangle_uvs)
            
            # 가중치에 따라 3D 좌표 계산
            interpolated_vertex = np.zeros(3)
            for w, v_idx in zip(weights, triangle_vertices):
                interpolated_vertex += w * vertices[v_idx]
            
            # 결과 저장
            landmark_vertices_3d.append(interpolated_vertex)
        
        print(f'황금비율 3D 점들: {landmark_vertices_3d}')
        
        # 결과 저장
        self.original_landmarks_3d = copy.deepcopy(landmark_vertices_3d)
        
        elapsed_time = time.time() - start_time
        print(f"황금비율 정점 {len(landmark_vertices_3d)}개 찾기 완료 (소요시간: {elapsed_time:.3f}초)")
        
        return landmark_vertices_3d
        
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
        
    def apply_golden_proportion_transformation(self):
        """
        황금비율 변환을 적용하는 함수:
        1. 4개 점의 X 좌표를 평균값으로 동일하게 만들기
        2. 4개 점의 Y 좌표를 가장 작은 Y 좌표로 동일하게 만들기
        3. 모든 점의 Y 좌표에서 5mm 빼기
        """
        if not hasattr(self, 'original_landmarks_3d'):
            raise ValueError("3D 랜드마크가 먼저 계산되어야 합니다. find_3d_points_from_uv()를 먼저 호출하세요.")
        
        # 원본 좌표 복사
        transformed_landmarks = copy.deepcopy(self.original_landmarks_3d)
        
        # 1. X 좌표의 평균값 계산
        avg_x = sum(point[0] for point in transformed_landmarks) / len(transformed_landmarks)
        print(f"X 좌표 평균값: {avg_x:.3f}")
        
        # 2. 가장 작은 Y 좌표 찾기
        min_y = min(point[1] for point in transformed_landmarks)
        print(f"가장 작은 Y 좌표: {min_y:.3f}")
        
        # 3. 모든 점의 X 좌표를 평균값으로, Y 좌표를 최소값으로 설정
        for i, point in enumerate(transformed_landmarks):
            original_x = point[0]
            original_y = point[1]
            transformed_landmarks[i][0] = avg_x
            transformed_landmarks[i][1] = min_y
            print(f"점 {i}: X {original_x:.3f} -> {avg_x:.3f}, Y {original_y:.3f} -> {min_y:.3f}")
        
        # 4. 모든 점의 Y 좌표에서 5mm 빼기
        for i, point in enumerate(transformed_landmarks):
            transformed_landmarks[i][1] -= 5.0  # 5mm 빼기
            print(f"점 {i}: Y {min_y:.3f} -> {transformed_landmarks[i][1]:.3f} (-5mm)")
        
        # 결과 저장
        self.transformed_landmarks_3d = transformed_landmarks
        
        print(f"\n황금비율 변환 완료:")
        print(f"원본 좌표:")
        for i, point in enumerate(self.original_landmarks_3d):
            print(f"  점 {i}: [{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}]")
        print(f"변환된 좌표:")
        for i, point in enumerate(self.transformed_landmarks_3d):
            print(f"  점 {i}: [{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}]")
        
        return self.transformed_landmarks_3d
        
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
        
    def create_visualization_meshes(self):
        """시각화를 위한 메시 객체들을 생성하는 함수"""
        if not hasattr(self, 'original_landmarks_3d') or not hasattr(self, 'transformed_landmarks_3d'):
            raise ValueError("3D 랜드마크와 변환된 좌표가 먼저 계산되어야 합니다.")
        
        # 원본 황금비율 점들을 위한 메시 (빨간색 구)
        self.original_landmarks_mesh = Mesh()
        radius = 2  # 적절한 크기 설정
        sphere_vertices, sphere_faces = self._create_sphere_representation(self.original_landmarks_3d, radius)
        self.original_landmarks_mesh.vertices = sphere_vertices
        self.original_landmarks_mesh.faces = sphere_faces
        
        # 변환된 황금비율 점들을 위한 메시 (파란색 구)
        self.transformed_landmarks_mesh = Mesh()
        sphere_vertices, sphere_faces = self._create_sphere_representation(self.transformed_landmarks_3d, radius)
        self.transformed_landmarks_mesh.vertices = sphere_vertices
        self.transformed_landmarks_mesh.faces = sphere_faces
        
        print(f"시각화 메시 생성 완료:")
        print(f"  - 원본 랜드마크 메시: {len(self.original_landmarks_mesh.vertices)} 정점, {len(self.original_landmarks_mesh.faces)} 면")
        print(f"  - 변환된 랜드마크 메시: {len(self.transformed_landmarks_mesh.vertices)} 정점, {len(self.transformed_landmarks_mesh.faces)} 면")
        
    def run_analysis(self):
        """
        황금비율 분석을 실행하는 메인 함수
        """
        print("=== 황금비율 분석 시작 ===")
        
        # 1. 얼굴 랜드마크 감지
        print("\n1. 얼굴 랜드마크 감지 중...")
        self.find_golden_proportion_landmarks()
        
        # 2. 3D 점 추출
        print("\n2. 3D 좌표 추출 중...")
        self.find_3d_points_from_uv()
        
        # 3. 황금비율 변환 적용
        print("\n3. 황금비율 변환 적용 중...")
        self.apply_golden_proportion_transformation()
        
        # 4. 시각화 메시 생성
        print("\n4. 시각화 메시 생성 중...")
        self.create_visualization_meshes()
        
        # 5. 시각화 (옵션)
        if self.visualization:
            print("\n5. 결과 시각화 중...")
            visualize_meshes(
                [self.face_mesh, self.original_landmarks_mesh, self.transformed_landmarks_mesh], 
                ["Face", "Original Points", "Transformed Points"], 
                title="Golden Proportion Analysis"
            )
        
        print("\n=== 황금비율 분석 완료 ===")
        
        # transformed_landmarks_3d를 a, b, c, d 형태의 JSON으로 변환
        golden_proportion_points = {
            'a': self.transformed_landmarks_3d[0].tolist(),  # 두 눈의 중심
            'b': self.transformed_landmarks_3d[1].tolist(),  # 코 높은점
            'c': self.transformed_landmarks_3d[2].tolist(),  # 입꼬리 중점
            'd': self.transformed_landmarks_3d[3].tolist()   # 턱
        }
        
        # 결과 반환
        return golden_proportion_points


if __name__ == "__main__":
    # 테스트 실행
    g_finder = GoldenProportionFinder(face_mesh_path="../../example/data/FaceScan/Smile/Smile.obj", visualization=True)
    # g_finder = GoldenProportionFinder(face_mesh_path="../../example/data/choi/Smile.obj", visualization=True)
    # g_finder = GoldenProportionFinder(face_mesh_path="../../example/data/sim/Smile.obj", visualization=True)
    # g_finder = GoldenProportionFinder(face_mesh_path="../../example/data/park1/Smile.obj", visualization=True)
    # g_finder = GoldenProportionFinder(face_mesh_path="../../example/data/park2/Smile.obj", visualization=True)
    # g_finder = GoldenProportionFinder(face_mesh_path="../../example/data/oh/Smile.obj", visualization=True)
    # g_finder = GoldenProportionFinder(face_mesh_path="../../example/data/kim/Smile.obj", visualization=True)
    result = g_finder.run_analysis()
    print(f"\n최종 결과: {result}")
        