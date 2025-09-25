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
from retinaface import RetinaFace

class GoldenProportionFinder:
    def __init__(self, face_mesh_path=None, face_mesh=None, face_image_path=None, visualization=False):
        """
        Args:
            face_mesh_path: 3D 메시 파일 경로 (기존 방식 - 호환성 유지)
            face_mesh: Mesh 객체 (FaceAlignment3D 결과 또는 기존 3D 메시)
            face_image_path: 이미지 파일 경로 (face_mesh가 None일 때 사용)
            visualization: 시각화 여부
        """
        self.visualization = visualization
        
        # MediaPipe 초기화
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh_detector = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            refine_landmarks=True
        )
        
        # 찾을 4개의 랜드마크 인덱스 정의 (MediaPipe 얼굴 메시는 0~467 인덱스)
        self.landmark_indices = {
            'left_eye_inner': 468,   # 왼쪽 눈 안쪽 모서리
            'right_eye_inner': 473, # 오른쪽 눈 안쪽 모서리  
            'nose_tip': 2,          # 코 끝
            'left_mouth_corner': 61,  # 왼쪽 입꼴리
            'right_mouth_corner': 291, # 오른쪽 입꼴리
            'chin': 152              # 턱 아래
        }
        
        # 기존 방식 호환성 유지: face_mesh_path가 제공된 경우
        if face_mesh_path is not None:
            # 파일 확장자에 따라 이미지 파일 경로 생성 (기존 방식)
            base_path = face_mesh_path.rsplit('.', 1)[0]  # 확장자 제거
            
            # PNG 파일 먼저 확인
            image_path = base_path + '.png'
            if not os.path.exists(image_path):
                # JPG 파일 확인
                image_path = base_path + '.jpg'
                if not os.path.exists(image_path):
                    print(f"경고: 이미지 파일을 찾을 수 없습니다: {base_path}")
                    image_path = None
            
            self.face_image_path = image_path
            self.face_mesh = Mesh.from_file(face_mesh_path)
            
        elif face_mesh is not None:
            # Mesh 객체가 직접 전달된 경우 (새로운 방식)
            self.face_mesh = face_mesh
            self.face_image_path = face_image_path  # 이미지 경로가 별도로 제공될 수 있음
            
        elif face_image_path is not None:
            # 이미지 경로만 제공된 경우 - plane mesh 생성
            self.face_image_path = face_image_path
            self.face_mesh = self._create_plane_mesh_from_image(face_image_path)
            
        else:
            raise ValueError("face_mesh_path, face_mesh 또는 face_image_path 중 하나는 반드시 제공되어야 합니다.")
    
    def _create_plane_mesh_from_image(self, image_path):
        """
        이미지에서 FaceAlignment3D 결과와 유사한 plane mesh를 생성
        """
        print(f"이미지에서 plane mesh 생성: {image_path}")
        
        # 이미지 로드 - RGBA 이미지 지원
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # 알파 채널도 읽기
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
        
        # RGBA를 RGB로 변환 (알파 채널이 있는 경우)
        if len(image.shape) == 3 and image.shape[2] == 4:
            # RGBA 이미지인 경우 RGB로 변환
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            print("RGBA 이미지를 RGB로 변환했습니다.")
        
        h, w = image.shape[:2]
        
        # 얼굴 랜드마크 감지하여 스케일 계산
        landmarks = self._detect_face_landmarks_for_scale(image_path)
        
        if landmarks:
            # 입 너비 기반 스케일 계산 (FaceAlignment3D와 동일한 방식)
            mouth_left = np.array(landmarks['mouth_left'])
            mouth_right = np.array(landmarks['mouth_right'])
            mouth_width_pixels = np.linalg.norm(mouth_right - mouth_left)
            target_mouth_width = 50.0  # FaceAlignment3D의 기본값
            
            if mouth_width_pixels > 0:
                scale_factor = target_mouth_width / mouth_width_pixels
            else:
                scale_factor = 1.0
        else:
            # 랜드마크를 찾을 수 없으면 기본 스케일 사용
            scale_factor = 0.1  # 기본 스케일
        
        plane_width = w * scale_factor
        plane_height = h * scale_factor
        
        half_width = plane_width / 2
        half_height = plane_height / 2
        
        # XZ 평면에 정점 생성 (Y=0)
        vertices = np.array([
            [-half_width, 0, -half_height],  # 0: bottom-left
            [half_width, 0, -half_height],   # 1: bottom-right  
            [half_width, 0, half_height],    # 2: top-right
            [-half_width, 0, half_height]    # 3: top-left
        ])
        
        # 삼각형 면 생성
        faces = np.array([[0, 2, 1], [0, 3, 2]])
        
        # UV 좌표 생성 (이미지와 매핑)
        uvs = np.array([
            [0, 0],  # bottom-left
            [1, 0],  # bottom-right
            [1, 1],  # top-right
            [0, 1]   # top-left
        ])
        
        # Mesh 객체 생성
        mesh = Mesh()
        mesh.vertices = vertices
        mesh.faces = faces
        mesh.uvs = uvs
        mesh.face_uvs = faces  # UV 인덱스는 정점 인덱스와 동일
        
        print(f"Plane mesh 생성 완료:")
        print(f"  - 크기: {plane_width:.1f} x {plane_height:.1f}")
        print(f"  - 정점 수: {len(vertices)}")
        print(f"  - 면 수: {len(faces)}")
        
        return mesh
    
    def _detect_face_landmarks_for_scale(self, image_path):
        """
        스케일 계산을 위한 얼굴 랜드마크 감지 (RetinaFace 사용)
        """
        try:
            faces = RetinaFace.detect_faces(image_path)
            if not faces:
                print("얼굴을 감지할 수 없습니다.")
                return None
            
            # 첫 번째 얼굴의 랜드마크 사용
            face_key = list(faces.keys())[0]
            landmarks = faces[face_key]['landmarks']
            
            # RetinaFace 랜드마크를 적절한 형식으로 변환
            return {
                'left_eye': landmarks['left_eye'],
                'right_eye': landmarks['right_eye'],
                'mouth_left': landmarks['mouth_left'],
                'mouth_right': landmarks['mouth_right']
            }
        except Exception as e:
            print(f"랜드마크 감지 실패: {e}")
            return None

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
        # 이미지 소스 확인 (파일 경로 또는 메모리 배열)
        image = None
        image_source = None
        
        if self.face_image_path is not None:
            # 파일에서 이미지 로드 - RGBA 지원
            image = cv2.imread(self.face_image_path, cv2.IMREAD_UNCHANGED)
            image_source = "파일"
            
            # RGBA를 BGR로 먼저 변환 (알파 채널 제거)
            if image is not None and len(image.shape) == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                print("파일에서 RGBA 이미지를 BGR로 변환했습니다.")
                
        elif hasattr(self, '_has_image_array') and self._has_image_array:
            # 메모리 배열에서 이미지 사용
            image = self._face_image_array.copy()
            image_source = "메모리"
            
            # RGBA를 RGB로 변환 (메모리 배열은 보통 RGB 순서)
            if len(image.shape) == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                print("메모리에서 RGBA 이미지를 RGB로 변환했습니다.")
        
        if image is None:
            # 이미지가 없는 경우 기본 UV 좌표 사용
            print("이미지 소스가 없습니다. 기본 UV 좌표를 사용합니다.")
            return self._create_default_golden_proportion_uv()
        
        print(f"이미지 소스: {image_source}, 크기: {image.shape}")
        
        # 이미지 크기 가져오기
        h, w = image.shape[:2]
        
        # RGB로 변환 (MediaPipe는 RGB 형식 필요)
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                # RGBA 이미지인 경우 RGB로 변환
                image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                print("RGBA 이미지를 RGB로 변환했습니다.")
            elif image.shape[2] == 3:
                # BGR 이미지인 경우 RGB로 변환
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # 예상치 못한 채널 수
                print(f"예상치 못한 이미지 채널 수: {image.shape[2]}")
                image_rgb = image
        elif len(image.shape) == 2:
            # 그레이스케일 이미지인 경우 RGB로 변환
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            # 기타 경우 그대로 사용
            image_rgb = image
        
        # 얼굴 랜드마크 감지
        results = self.face_mesh_detector.process(image_rgb)
        
        if not results.multi_face_landmarks:
            print("얼굴을 감지할 수 없습니다. 기본 UV 좌표를 사용합니다.")
            # 이미지 크기 정보는 저장해둠
            self._cached_image_size = (w, h)
            return self._create_default_golden_proportion_uv()
        
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
    
    def _create_default_golden_proportion_uv(self):
        """
        이미지가 없을 때 사용할 기본 황금비율 UV 좌표 생성
        평면의 중앙 부분에 황금비율에 따라 점들을 배치
        """
        # 황금비율 (1.618)을 기반으로 얼굴 비율 계산
        golden_ratio = 1.618
        
        # 평면의 중앙을 기준으로 황금비율 점들 배치
        # A: 두 눈의 중심 (상단 1/3 지점)
        eye_center_uv = [0.5, 0.7]
        
        # B: 코 높은점 (중앙 약간 위)
        nose_uv = [0.5, 0.55]
        
        # C: 입꼬리 중점 (중앙 약간 아래)
        mouth_uv = [0.5, 0.4]
        
        # D: 턱 (하단 1/4 지점)
        chin_uv = [0.5, 0.2]
        
        self.landmark_uv = [eye_center_uv, nose_uv, mouth_uv, chin_uv]
        
        # 더미 이미지 좌표도 생성 (1000x1000 가상 이미지 기준)
        virtual_size = 1000
        self.landmark_image_points = [
            [int(uv[0] * virtual_size), int((1.0 - uv[1]) * virtual_size)]
            for uv in self.landmark_uv
        ]
        
        # 캐시 정보도 설정
        self._cached_image_size = (virtual_size, virtual_size)
        
        print(f"기본 황금비율 UV 좌표 생성:")
        print(f"  A (Eye Center): UV: {self.landmark_uv[0]}")
        print(f"  B (Nose Tip): UV: {self.landmark_uv[1]}")
        print(f"  C (Mouth Center): UV: {self.landmark_uv[2]}")
        print(f"  D (Chin): UV: {self.landmark_uv[3]}")
        
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
        # 캐싱된 데이터가 있는지 확인
        if not hasattr(self, '_cached_image_size'):
            raise ValueError("랜드마크가 먼저 감지되어야 합니다. find_golden_proportion_landmarks()를 먼저 호출하세요.")
        
        w, h = self._cached_image_size
        
        # 이미지 기반 랜드마크가 있는 경우
        if hasattr(self, '_cached_face_landmarks') and self._cached_face_landmarks is not None:
            face_landmarks = self._cached_face_landmarks
            
            # 해당 랜드마크 추출
            landmark = face_landmarks.landmark[self.landmark_indices[landmark_key]]
            
            # 2D 이미지 좌표
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            
            # UV 좌표로 변환
            uv = self._normalize_and_flip_coordinates([[x, y]], (w, h))[0]
        else:
            # 기본 UV 좌표 사용 (이미지가 없는 경우)
            landmark_map = {
                'left_eye_inner': [0.45, 0.7],   # 왼쪽 눈
                'right_eye_inner': [0.55, 0.7],  # 오른쪽 눈
                'nose_tip': [0.5, 0.55],         # 코
                'left_mouth_corner': [0.45, 0.4], # 왼쪽 입꼴리
                'right_mouth_corner': [0.55, 0.4], # 오른쪽 입꼴리
                'chin': [0.5, 0.2]               # 턱
            }
            uv = landmark_map.get(landmark_key, [0.5, 0.5])
        
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
        
        # 좌우 중앙 정렬은 건너뛰고 원본 X좌표 유지
        # (각 점의 고유한 위치를 보존하기 위해)
        avg_right = np.mean(transformed_local[:, 0])
        print(f"좌우 평균값: {avg_right:.3f} (정렬하지 않고 원본 유지)")
        
        # 평면에 수직인 방향으로 5mm 띄우기
        # 평면 메시인지 확인
        is_plane = self._is_plane_mesh(self.face_mesh)
        
        if is_plane:
            # 평면의 경우: 평면의 법선 방향으로 5mm 띄움
            plane_normal = self._calculate_plane_normal(self.face_mesh)
            print(f"평면 법선 벡터: {plane_normal}")
            
            # 평면에서 5mm 떨어진 위치로 모든 점 이동
            # Y축 방향으로 -5.0mm 오프셋 추가 (기존 Y좌표 + 오프셋)
            transformed_local[:, 1] -= 0.0  # Y축 방향으로 5mm 아래로 추가
            print(f"평면 메시: Y축 방향으로 -5.0mm 오프셋 추가 적용")
            
        else:
            # 3D 메시의 경우: 기존 방식 (forward 방향으로 5mm)
            max_forward = np.min(transformed_local[:, 2])
            transformed_local[:, 2] = max_forward - 5.0  # 5mm 아래로
            print(f"3D 메시: Z축 방향으로 -5.0mm 오프셋 적용")
        
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
            
            # 평면인 경우 법선 방향으로 추가 오프셋 적용
            if is_plane:
                offset_distance = 5.0  # 5mm 아래로
                global_point += offset_distance * plane_normal
                print(f"  점 {i}: 평면 법선 방향으로 {offset_distance}mm 오프셋 적용")
            
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
            import pyvista as pv
            
            # 플롯터 생성
            plotter = pv.Plotter()
            
            # 얼굴 메시 처리 - 텍스처 소스 확인
            has_texture_source = False
            texture_source_type = None
            
            # 1. 파일 경로 확인
            if self.face_image_path and os.path.exists(self.face_image_path):
                has_texture_source = True
                texture_source_type = "파일"
            # 2. 메모리 배열 확인
            elif hasattr(self, '_has_image_array') and self._has_image_array:
                has_texture_source = True
                texture_source_type = "메모리"
            
            if has_texture_source:
                # 텍스처 소스가 있는 경우
                face_pv = self._mesh_to_pyvista_with_texture(self.face_mesh, self.face_image_path)
                if hasattr(face_pv, '_has_texture') and face_pv._has_texture:
                    # 텍스처가 성공적으로 적용된 경우
                    plotter.add_mesh(face_pv, texture=face_pv._texture, opacity=0.8, 
                                   label=f"Face with Texture ({texture_source_type})")
                    print(f"시각화: {texture_source_type} 텍스처 적용됨")
                else:
                    # 텍스처 적용 실패 시 기본 색상
                    plotter.add_mesh(face_pv, color='lightgray', opacity=0.5, label="Face")
                    print(f"시각화: {texture_source_type} 텍스처 적용 실패, 기본 색상 사용")
            else:
                # 텍스처 소스가 없는 경우 기본 색상으로 시각화
                face_pv = self._mesh_to_pyvista(self.face_mesh)
                plotter.add_mesh(face_pv, color='lightgray', opacity=0.3, label="Face")
                print("시각화: 텍스처 소스 없음, 기본 색상 사용")
            
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

    def _mesh_to_pyvista_with_texture(self, mesh, texture_path=None):
        """Mesh 객체를 PyVista 메쉬로 변환하고 텍스처 적용"""
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
        
        # 텍스처 적용 시도
        texture_applied = False
        texture_source = None
        
        # 1. 이미지 배열이 있는 경우 (메모리에서 직접)
        if hasattr(self, '_has_image_array') and self._has_image_array:
            texture_source = self._face_image_array
            print(f"메모리의 이미지 배열 사용 (형태: {self._face_image_array.shape})")
        # 2. 이미지 파일 경로가 있는 경우
        elif texture_path is not None and os.path.exists(texture_path):
            texture_source = texture_path
            print(f"파일에서 텍스처 로드: {texture_path}")
        
        if (hasattr(mesh, 'uvs') and mesh.uvs is not None and texture_source is not None):
            try:
                # UV 좌표를 텍스처 좌표로 설정 (PyVista는 2D만 지원)
                tex_coords = mesh.uvs.copy()
                # 2D 형태로 유지 (u, v만 사용)
                if tex_coords.shape[1] > 2:
                    tex_coords = tex_coords[:, :2]
                
                # UV 좌표 개수와 정점 개수가 일치하는지 확인
                num_points = len(pv_mesh.points)
                num_uvs = len(tex_coords)
                
                print(f"디버그: 정점 수={num_points}, UV 수={num_uvs}")
                print(f"UV 좌표: {tex_coords}")
                
                if num_points != num_uvs:
                    print(f"경고: UV 좌표 개수({num_uvs})와 정점 개수({num_points})가 일치하지 않습니다. 텍스처 적용을 건너뜁니다.")
                    texture_applied = False
                else:
                    # UV 좌표 적용
                    pv_mesh.active_t_coords = tex_coords
                    
                    # 텍스처 로드 (배열 또는 파일)
                    if isinstance(texture_source, str):
                        # 파일에서 로드
                        texture = pv.read_texture(texture_source)
                        print("파일에서 텍스처 로드 완료")
                    else:
                        # 메모리 배열에서 직접 생성 (색상 변환 없이)
                        texture = pv.numpy_to_texture(texture_source)
                        print("메모리 배열에서 텍스처 생성 완료 (원본 색상 순서 유지)")
                    
                    # 텍스처를 mesh에 저장 (사용자 정의 속성으로)
                    pv_mesh._texture = texture
                    texture_applied = True
                    print("텍스처 적용 완료")
                
            except Exception as e:
                print(f"텍스처 로드 실패: {e}")
                import traceback
                traceback.print_exc()
                texture_applied = False
        else:
            print(f"텍스처 적용 조건 확인:")
            print(f"  - mesh.uvs 존재: {hasattr(mesh, 'uvs') and mesh.uvs is not None}")
            print(f"  - texture_source 존재: {texture_source is not None}")
        
        # 텍스처 적용 여부를 mesh에 저장
        pv_mesh._has_texture = texture_applied
        
        return pv_mesh
    
    def _is_plane_mesh(self, mesh):
        """메시가 평면 메시인지 확인 (X, Y, Z축 중 가장 작은 변화량으로 평면 판단)"""
        if mesh.vertices is None or len(mesh.vertices) < 3:
            return False
        
        # 각 축의 좌표 변화량 확인
        x_coords = mesh.vertices[:, 0]
        y_coords = mesh.vertices[:, 1]
        z_coords = mesh.vertices[:, 2]
        
        x_range = np.max(x_coords) - np.min(x_coords)
        y_range = np.max(y_coords) - np.min(y_coords)
        z_range = np.max(z_coords) - np.min(z_coords)
        
        # 가장 작은 변화량 찾기
        min_range = min(x_range, y_range, z_range)
        
        # 가장 작은 축의 변화가 매우 작으면 평면으로 간주 (임계값: 1.0)
        is_plane = min_range < 1.0
        
        print(f"메시 좌표 범위 - X: {x_range:.3f}, Y: {y_range:.3f}, Z: {z_range:.3f}")
        print(f"최소 범위: {min_range:.3f}, 평면 여부: {is_plane}")
        
        return is_plane
    
    def _calculate_plane_normal(self, mesh):
        """평면의 법선 벡터 계산"""
        if mesh.vertices is None or len(mesh.vertices) < 3:
            return np.array([0, 1, 0])  # 기본값: Y축 방향
        
        # 첫 번째 삼각형의 법선 벡터 계산
        if mesh.faces is not None and len(mesh.faces) > 0:
            face = mesh.faces[0]
            v1 = mesh.vertices[face[0]]
            v2 = mesh.vertices[face[1]]
            v3 = mesh.vertices[face[2]]
            
            # 두 벡터의 외적으로 법선 벡터 계산
            edge1 = v2 - v1
            edge2 = v3 - v1
            normal = np.cross(edge1, edge2)
            
            # 정규화
            if np.linalg.norm(normal) > 0:
                normal = normal / np.linalg.norm(normal)
            else:
                normal = np.array([0, 1, 0])  # 기본값
            
            return normal
        else:
            # 면 정보가 없으면 기본값 반환
            return np.array([0, 1, 0])

    def _mesh_to_pyvista(self, mesh):
        """Mesh 객체를 PyVista 메쉬로 변환 (기존 메서드 유지)"""
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

    @classmethod
    def from_image(cls, image_path, visualization=False):
        """
        이미지 경로에서 GoldenProportionFinder 인스턴스를 생성하는 클래스 메서드
        """
        return cls(face_mesh=None, face_image_path=image_path, visualization=visualization)
    
    @classmethod  
    def from_mesh(cls, face_mesh, face_image_path=None, visualization=False):
        """
        Mesh 객체에서 GoldenProportionFinder 인스턴스를 생성하는 클래스 메서드
        """
        return cls(face_mesh=face_mesh, face_image_path=face_image_path, visualization=visualization)
    
    @classmethod
    def from_face_alignment_result(cls, face_alignment_result, visualization=False):
        """
        FaceAlignment3D 결과에서 GoldenProportionFinder 인스턴스를 생성하는 클래스 메서드
        
        Args:
            face_alignment_result: FaceAlignmentResult 객체 (front_plane, front_texture 등 포함)
            visualization: 시각화 여부
        """
        # FaceAlignmentResult에서 front_plane을 Mesh 객체로 변환
        face_mesh = cls._convert_o3d_to_mesh(face_alignment_result.front_plane)
        
        # 텍스처 이미지가 있으면 numpy 배열로 직접 전달
        face_image_array = None
        
        # FaceAlignment3D 결과에서 텍스처 추출 시도
        texture_source = None
        
        # 1. front_texture 속성 확인
        if hasattr(face_alignment_result, 'front_texture') and face_alignment_result.front_texture is not None:
            texture_source = face_alignment_result.front_texture
            print("front_texture에서 텍스처 발견")
        
        # 2. front_plane의 textures 확인
        elif (hasattr(face_alignment_result, 'front_plane') and 
              hasattr(face_alignment_result.front_plane, 'textures') and 
              len(face_alignment_result.front_plane.textures) > 0):
            texture_source = face_alignment_result.front_plane.textures[0]
            print("front_plane.textures에서 텍스처 발견")
        
        if texture_source is not None:
            import numpy as np
            
            try:
                # Open3D Image를 numpy 배열로 변환
                face_image_array = np.asarray(texture_source)
                print(f"Open3D 텍스처 변환 성공: {face_image_array.shape}")
            except Exception as e:
                print(f"Open3D 텍스처 변환 실패: {e}")
                face_image_array = None
        else:
            print("FaceAlignment3D 결과에서 텍스처를 찾을 수 없습니다.")
        
        instance = cls(face_mesh=face_mesh, face_image_path=None, visualization=visualization)
        
        # 이미지 배열을 직접 저장
        if face_image_array is not None:
            instance._face_image_array = face_image_array
            instance._has_image_array = True
            print(f"FaceAlignment3D 이미지 배열 저장 완료: {face_image_array.shape}")
        else:
            instance._has_image_array = False
            print("FaceAlignment3D에서 텍스처 이미지가 없습니다.")
            # 텍스처가 없어도 기본 UV 좌표로 동작하도록 함
        
        return instance
    
    @staticmethod
    def _convert_o3d_to_mesh(o3d_mesh):
        """
        Open3D TriangleMesh를 pyNeo3DLib Mesh 객체로 변환
        """
        mesh = Mesh()
        
        # 정점 변환
        mesh.vertices = np.asarray(o3d_mesh.vertices)
        
        # 면 변환
        mesh.faces = np.asarray(o3d_mesh.triangles)
        
        # UV 좌표 생성 (정점 수에 맞춰서)
        num_vertices = len(mesh.vertices)
        
        if num_vertices == 4:  # 사각형 평면
            mesh.uvs = np.array([
                [0, 0],  # bottom-left
                [1, 0],  # bottom-right
                [1, 1],  # top-right
                [0, 1]   # top-left
            ])
            mesh.face_uvs = mesh.faces
        else:
            # 일반적인 경우 기본 UV 좌표 생성
            mesh._try_create_default_uvs()
        
        return mesh


if __name__ == "__main__":
    # 테스트 실행
    print("=== 황금비율 분석 테스트 ===\n")
    
    # 1. 기존 방식 - 3D 메시 파일 경로 테스트 (호환성 유지)
    print("1. 기존 방식 - 3D 메시 파일 경로 테스트:")
    mesh_path = "../../example/data/ahn/Smile/Smile_Scan.ply"
    g_finder_legacy = GoldenProportionFinder(face_mesh_path=mesh_path, visualization=True)
    result_legacy = g_finder_legacy.run_analysis()
    print(f"기존 방식 결과: {result_legacy}\n")
    
    # 2. 새로운 방식 - Mesh 객체 직접 전달 테스트
    print("2. 새로운 방식 - Mesh 객체 직접 전달 테스트:")
    import os
    base_path = os.path.splitext(mesh_path)[0]
    image_path = base_path + '.png' if os.path.exists(base_path + '.png') else base_path + '.jpg'
    if not os.path.exists(image_path):
        image_path = None
    
    face_mesh = Mesh.from_file(mesh_path)
    g_finder_mesh = GoldenProportionFinder.from_mesh(face_mesh, face_image_path=image_path, visualization=True)
    result_mesh = g_finder_mesh.run_analysis()
    print(f"Mesh 객체 결과: {result_mesh}\n")
    
    # 3. FaceAlignment3D 시뮬레이션 테스트 (이미지에서 3D plane 생성)
    print("3. FaceAlignment3D 시뮬레이션 테스트:")
    from pyNeo3DLib.faceRegisration.faceAlign import FaceAlignment3D
    
    # 테스트용 이미지 경로들
    test_images = [
        "../../example/data/photo/hk1.jpg",
        "../../example/data/photo/su1.png"
    ]
    
    for test_image in test_images:
        if os.path.exists(test_image):
            print(f"\n테스트 이미지: {test_image}")
            
            try:
                # FaceAlignment3D로 3D plane 생성
                face_alignment = FaceAlignment3D(front_image_path=test_image)
                rotation_matrix, alignment_result = face_alignment.run_registration(visualize=False)
                
                if alignment_result is not None:
                    # FaceAlignment3D 결과로 GoldenProportionFinder 생성
                    g_finder_alignment = GoldenProportionFinder.from_face_alignment_result(
                        alignment_result, visualization=True)
                    result_alignment = g_finder_alignment.run_analysis()
                    print(f"FaceAlignment3D 결과: {result_alignment}")
                else:
                    print("FaceAlignment3D 처리 실패")
                    
            except Exception as e:
                print(f"FaceAlignment3D 테스트 실패: {e}")
                
                # 대체 방법: 직접 plane mesh 생성
                print("대체 방법: 직접 plane mesh 생성")
                g_finder_image = GoldenProportionFinder.from_image(test_image, visualization=True)
                result_image = g_finder_image.run_analysis()
                print(f"직접 plane 생성 결과: {result_image}")
            
            # break  # 첫 번째 유효한 이미지만 테스트
        