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
            'nose_tip': 2,          # 코 끝
            'left_mouth_corner': 61,  # 왼쪽 입꼴리
            'right_mouth_corner': 291, # 오른쪽 입꼴리
            'chin': 152              # 턱 아래
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
        
        # 결과를 인스턴스 변수에 캐싱 (성능 개선)
        self._cached_face_landmarks = results.multi_face_landmarks[0]
        self._cached_image_size = (w, h)
        
        # 4개의 특정 점 추출
        landmark_points = []
        
        # A: 두 눈의 중심점 (33번, 263번의 중심)
        left_eye = self._cached_face_landmarks.landmark[self.landmark_indices['left_eye_inner']]
        right_eye = self._cached_face_landmarks.landmark[self.landmark_indices['right_eye_inner']]
        eye_center_x = int((left_eye.x + right_eye.x) * w / 2)
        eye_center_y = int((left_eye.y + right_eye.y) * h / 2)
        landmark_points.append([eye_center_x, eye_center_y])
        
        # B: 코 높은점 (4번)
        nose_tip = self._cached_face_landmarks.landmark[self.landmark_indices['nose_tip']]
        nose_x = int(nose_tip.x * w)
        nose_y = int(nose_tip.y * h)
        landmark_points.append([nose_x, nose_y])
        
        # C: 입꼬리 중점 (61번, 291번의 중심)
        left_mouth = self._cached_face_landmarks.landmark[self.landmark_indices['left_mouth_corner']]
        right_mouth = self._cached_face_landmarks.landmark[self.landmark_indices['right_mouth_corner']]
        mouth_center_x = int((left_mouth.x + right_mouth.x) * w / 2)
        mouth_center_y = int((left_mouth.y + right_mouth.y) * h / 2)
        landmark_points.append([mouth_center_x, mouth_center_y])
        
        # D: 턱 (18번)
        chin = self._cached_face_landmarks.landmark[self.landmark_indices['chin']]
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
        
        # 캐싱된 KDTree 사용 (성능 개선)
        tree, triangle_centers = self._get_or_create_kdtree()
        
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
    
    def _get_or_create_kdtree(self):
        """KDTree를 생성하고 캐싱하는 함수 (성능 최적화)"""
        if hasattr(self, '_cached_kdtree'):
            return self._cached_kdtree, self._cached_triangle_centers
        
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
        
        # 캐싱
        self._cached_kdtree = tree
        self._cached_triangle_centers = triangle_centers
        
        return tree, triangle_centers
    
    
    def get_single_landmark_3d(self, landmark_key):
        """개별 랜드마크의 3D 좌표를 추출하는 함수 (최적화됨)"""
        # 캐싱된 데이터 사용 (성능 개선)
        if not hasattr(self, '_cached_face_landmarks'):
            raise ValueError("얼굴 랜드마크가 먼저 감지되어야 합니다. find_golden_proportion_landmarks()를 먼저 호출하세요.")
        
        face_landmarks = self._cached_face_landmarks
        w, h = self._cached_image_size
        
        # 해당 랜드마크 추출
        landmark = face_landmarks.landmark[self.landmark_indices[landmark_key]]
        
        # 2D 이미지 좌표
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        
        # UV 좌표로 변환
        uv = self._normalize_and_flip_coordinates([[x, y]], (w, h))[0]
        
        # 캐싱된 KDTree 사용
        tree, triangle_centers = self._get_or_create_kdtree()
        
        # 메시 데이터
        vertices = self.face_mesh.vertices
        faces = self.face_mesh.faces
        uvs = self.face_mesh.uvs
        
        # 3D 좌표 계산
        dist, idx = tree.query(uv, k=1)
        triangle_idx = triangle_centers[idx][1]
        triangle_vertices = faces[triangle_idx]
        triangle_uvs = uvs[triangle_vertices]
        
        weights = self._calculate_barycentric_weights(uv, triangle_uvs)
        interpolated_vertex = np.zeros(3)
        for w, v_idx in zip(weights, triangle_vertices):
            interpolated_vertex += w * vertices[v_idx]
        
        return interpolated_vertex


    def calculate_face_coordinate_system(self):
        """얼굴의 로컬 좌표계를 계산하는 함수"""
        # 왼쪽눈, 오른쪽눈 3D 좌표 추출
        left_eye_3d = self.get_single_landmark_3d('left_eye_inner')
        right_eye_3d = self.get_single_landmark_3d('right_eye_inner')
        
        # 4개 랜드마크 포인트
        points = np.array(self.original_landmarks_3d)
        
        # 1. 좌우축: 왼쪽눈 -> 오른쪽눈 방향
        eye_vector = right_eye_3d - left_eye_3d
        right_axis = eye_vector / np.linalg.norm(eye_vector)
        
        # 2. 위아래축: 4개 점 중 편차가 가장 큰 축 방향 찾기
        x_variance = np.var(points[:, 0])
        y_variance = np.var(points[:, 1])
        z_variance = np.var(points[:, 2])
        
        print(f"축별 편차 - X: {x_variance:.6f}, Y: {y_variance:.6f}, Z: {z_variance:.6f}")
        
        # 턱(마지막 점)에서 눈 중심으로의 방향을 기본으로 사용
        eye_center = (left_eye_3d + right_eye_3d) / 2
        chin_to_eye = eye_center - points[3]  # 턱에서 눈으로
        
        # right_axis와 직교하도록 투영 제거
        up_axis = chin_to_eye - np.dot(chin_to_eye, right_axis) * right_axis
        up_axis = up_axis / np.linalg.norm(up_axis)
        
        # 3. 앞뒤축: 위아래 × 좌우의 외적
        forward_axis = np.cross(up_axis, right_axis)
        forward_axis = forward_axis / np.linalg.norm(forward_axis)
        
        # 얼굴 중심점
        face_center = np.mean(points, axis=0)
        
        print(f"얼굴 로컬 좌표계:")
        print(f"  위아래 축: {up_axis}")
        print(f"  좌우 축: {right_axis}") 
        print(f"  앞뒤 축: {forward_axis}")
        print(f"  얼굴 중심: {face_center}")
        
        return up_axis, right_axis, forward_axis, face_center



    def apply_golden_proportion_transformation(self):
        """
        얼굴 방향을 고려한 황금비율 변환:
        1. 얼굴의 로컬 좌표계 계산
        2. 로컬 좌표계 기준으로 정렬
        3. 황금비율 적용
        """
        if not hasattr(self, 'original_landmarks_3d'):
            raise ValueError("3D 랜드마크가 먼저 계산되어야 합니다. find_3d_points_from_uv()를 먼저 호출하세요.")
        
        # 1. 얼굴 로컬 좌표계 계산
        up_axis, right_axis, forward_axis, face_center = self.calculate_face_coordinate_system()
        
        # 2. 로컬 좌표계로 변환
        points = np.array(self.original_landmarks_3d)
        local_coords = []
        
        print("원본 -> 로컬 좌표계 변환:")
        for i, point in enumerate(points):
            # 얼굴 중심 기준으로 이동
            relative_point = point - face_center
            
            # 로컬 좌표계로 투영
            local_right = np.dot(relative_point, right_axis)    # 좌우 (-왼쪽 +오른쪽)
            local_up = np.dot(relative_point, up_axis)          # 위아래 (-아래 +위)  
            local_forward = np.dot(relative_point, forward_axis) # 앞뒤 (-뒤 +앞)
            
            local_coords.append([local_right, local_up, local_forward])
            print(f"  점 {i}: 로컬좌표 [우:{local_right:.3f}, 상:{local_up:.3f}, 전:{local_forward:.3f}]")
        
        local_coords = np.array(local_coords)
        
        # 3. 로컬 좌표계에서 황금비율 적용
        transformed_local = local_coords.copy()
        
        # 모든 점을 좌우 중앙으로 정렬
        avg_right = np.mean(transformed_local[:, 0])
        transformed_local[:, 0] = avg_right  # 또는 0으로 중앙 정렬
        
        # 모든 점을 같은 앞뒤 위치로 정렬 (최상단 점 기준)
        max_forward = np.min(transformed_local[:, 2])
        transformed_local[:, 2] = max_forward - 5.0  # 5mm 아래로
        
        # 앞뒤 방향은 원본 그대로 유지 (정렬하지 않음)
        
        print(f"\n로컬 좌표계에서 변환:")
        print(f"  좌우 평균값: {avg_right:.3f} -> 0으로 정렬")
        # print(f"  최상단 위치: {max_up:.3f}")
        # print(f"  변환 후 위치: {max_up - 5.0:.3f} (-5mm)")
        print(f"  앞뒤 방향: 원본 그대로 유지")
        
        # 4. 다시 전역 좌표계로 변환
        final_landmarks = []
        for i, local_point in enumerate(transformed_local):
            # 로컬 좌표를 전역 좌표로 변환
            global_point = face_center + \
                        local_point[0] * right_axis + \
                        local_point[1] * up_axis + \
                        local_point[2] * forward_axis
            final_landmarks.append(global_point)
            print(f"  점 {i}: 최종좌표 [{global_point[0]:.3f}, {global_point[1]:.3f}, {global_point[2]:.3f}]")
        
        self.transformed_landmarks_3d = np.array(final_landmarks)
        
        print(f"\n얼굴 방향 고려한 황금비율 변환 완료!")
        
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
            # 색상 고정 시각화
            import pyvista as pv
            
            # 플롯터 생성
            plotter = pv.Plotter()
            
            # 얼굴 메시 (회색, 반투명)
            face_pv = self._mesh_to_pyvista(self.face_mesh)
            plotter.add_mesh(face_pv, color='lightgray', opacity=0.3, label="Face")
            
            # 원본 랜드마크 (빨간색)
            original_pv = self._mesh_to_pyvista(self.original_landmarks_mesh)
            plotter.add_mesh(original_pv, color='red', opacity=0.8, label="Original Points")
            
            # 변환된 랜드마크 (파란색)
            transformed_pv = self._mesh_to_pyvista(self.transformed_landmarks_mesh)
            plotter.add_mesh(transformed_pv, color='blue', opacity=0.8, label="Transformed Points")
            
            # 축과 범례 추가
            plotter.add_axes()
            plotter.add_legend()
            plotter.add_title("Golden Proportion Analysis", font_size=16)
            
            # 시각화
            plotter.show()
        
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

    def _mesh_to_pyvista(self, mesh):
        """Mesh 객체를 PyVista 메쉬로 변환"""
        import pyvista as pv
        
        vertices = mesh.vertices
        faces = mesh.faces
        
        pv_mesh = pv.PolyData()
        pv_mesh.points = vertices
        
        face_list = []
        for face in faces:
            face_list.append(len(face))
            face_list.extend(face)
        
        pv_mesh.faces = face_list
        return pv_mesh


if __name__ == "__main__":
    # 테스트 실행
    # g_finder = GoldenProportionFinder(face_mesh_path="../../example/data/FaceScan/Smile/Smile.obj", visualization=True)
    # g_finder = GoldenProportionFinder(face_mesh_path="../../example/data/ahn/Smile/Smile_Scan.ply", visualization=True)
    # g_finder = GoldenProportionFinder(face_mesh_path="../../example/data/choi/Smile.obj", visualization=True)
    # g_finder = GoldenProportionFinder(face_mesh_path="../../example/data/sim/Smile.obj", visualization=True)
    # g_finder = GoldenProportionFinder(face_mesh_path="../../example/data/park1/Smile.obj", visualization=True)
    g_finder = GoldenProportionFinder(face_mesh_path="../../example/data/park2/Smile.obj", visualization=True)
    # g_finder = GoldenProportionFinder(face_mesh_path="../../example/data/oh/Smile.obj", visualization=True)
    # g_finder = GoldenProportionFinder(face_mesh_path="../../example/data/kim/Smile.obj", visualization=True)
    result = g_finder.run_analysis()
    print(f"\n최종 결과: {result}")
        