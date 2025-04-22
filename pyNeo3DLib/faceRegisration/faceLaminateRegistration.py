import numpy as np
from pyNeo3DLib.fileLoader.mesh import Mesh
import copy
from scipy.spatial import ConvexHull
import mediapipe as mp
from pyNeo3DLib.visualization.neovis import visualize_meshes
import cv2
import open3d as o3d
import time
import os

class FaceMeshAnalyzer:
    def __init__(self):
        """MediaPipe Face Mesh 초기화"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        
        # 입술 랜드마크 인덱스
        self.outer_lips = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
                          291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
        self.inner_lips = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
                          308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
  


class FaceLaminateRegistration:
    def __init__(self, face_path, laminate_path, visualization=False):
        self.face_smile_path = face_path
        self.laminate_path = laminate_path
        self.visualization = visualization
        # 변환 행렬 초기화 (4x4 행렬로 설정)
        self.transform_matrix = np.eye(4)

        self.__load_models()

    def __load_models(self):
        self.face_smile_mesh = Mesh.from_file(self.face_smile_path)
        self.laminate_mesh = Mesh.from_file(self.laminate_path)

        return self.face_smile_mesh, self.laminate_mesh
    
    def apply_transformation(self, transformation_matrix):
        """
        face_smile_mesh에 변환을 적용하고 transform_matrix에 누적합니다.
        
        Args:
            transformation_matrix (np.ndarray): 4x4 변환 행렬
        """
        # 변환 행렬 누적
        self.transform_matrix = np.dot(transformation_matrix, self.transform_matrix)
        
        # 메시 정점에 변환 적용
        vertices = self.face_smile_mesh.vertices
        # 동차 좌표로 변환 (4xN 행렬)
        vertices_homogeneous = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
        # 변환 적용
        transformed_vertices = np.dot(vertices_homogeneous, self.transform_matrix.T)
        # 동차 좌표에서 3D 좌표로 변환
        self.face_smile_mesh.vertices = transformed_vertices[:, :3]
        
        # 법선 벡터도 변환 (회전만 적용)
        if self.face_smile_mesh.normals is not None:
            normals = self.face_smile_mesh.normals
            # 법선 벡터는 이동 성분 없이 회전만 적용
            rotation_matrix = self.transform_matrix[:3, :3]
            self.face_smile_mesh.normals = np.dot(normals, rotation_matrix.T)
    
    def align_y_axis(self):
        """
        Z축 기준으로 180도 회전하는 변환을 적용합니다.
        """
        # Z축 기준으로 180도 회전하는 변환 행렬 생성
        angle = np.pi  # 180도
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0, 0],
            [np.sin(angle), np.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # 변환 적용
        self.apply_transformation(rotation_matrix)

    def find_lip_via_analyze_face_landmarks(self):
        """
        얼굴 이미지에서 입술 랜드마크를 추출하고 UV 좌표로 변환합니다.
        """
        analyzer = FaceMeshAnalyzer()
        
        # 이미지 파일 경로 추출 (obj 파일과 같은 이름의 이미지 파일 가정)
        image_path = self.face_smile_path.replace('.obj', '.png')
        if not os.path.exists(image_path):
            image_path = self.face_smile_path.replace('.obj', '.jpg')
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {self.face_smile_path}의 이미지 버전")
        
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
        
        # 이미지 크기 가져오기
        h, w = image.shape[:2]
        
        # 이미지를 RGB로 변환 (MediaPipe는 RGB 형식 필요)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 얼굴 랜드마크 감지
        results = analyzer.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            raise ValueError("이미지에서 얼굴을 감지할 수 없습니다.")
        
        # 첫 번째 얼굴의 랜드마크 가져오기
        face_landmarks = results.multi_face_landmarks[0]
        
        # 입술 랜드마크 추출
        inner_points = []
        outer_points = []
        
        # 내부 입술 랜드마크 추출
        for idx in analyzer.inner_lips:
            landmark = face_landmarks.landmark[idx]
            # 이미지 좌표로 변환 (0~1 범위를 픽셀 좌표로)
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            inner_points.append([x, y])
        
        # 외부 입술 랜드마크 추출
        for idx in analyzer.outer_lips:
            landmark = face_landmarks.landmark[idx]
            # 이미지 좌표로 변환 (0~1 범위를 픽셀 좌표로)
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            outer_points.append([x, y])
        
        # UV 좌표로 변환 (정규화 및 좌표 뒤집기)
        inner_uv_points = self._normalize_and_flip_coordinates(inner_points, (w, h))
        outer_uv_points = self._normalize_and_flip_coordinates(outer_points, (w, h))
        
        # 결과 저장
        self.inner_lip_points = inner_points
        self.outer_lip_points = outer_points
        self.inner_lip_uv = inner_uv_points
        self.outer_lip_uv = outer_uv_points
        
        return inner_uv_points, outer_uv_points
    
    def _normalize_and_flip_coordinates(self, points, image_size):
        """
        이미지 좌표를 UV 좌표로 변환합니다.
        
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
    
    def is_point_in_polygon(self, point, polygon, epsilon=1e-10):
        """점이 다각형 내부에 있는지 확인 (Ray Casting Algorithm)"""
        x, y = point
        n = len(polygon)
        inside = False
        
        # 경계선 상의 점 처리
        def on_segment(p, q, r):
            if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
                d = (r[1] - p[1]) * (q[0] - p[0]) - (q[1] - p[1]) * (r[0] - p[0])
                if abs(d) < epsilon:
                    return True
            return False
        
        # 경계선 상의 점 확인
        j = n - 1
        for i in range(n):
            if on_segment(polygon[i], [x, y], polygon[j]):
                return True
            j = i
        
        # Ray casting
        j = n - 1
        for i in range(n):
            if ((polygon[i, 1] > y) != (polygon[j, 1] > y) and
                (x < (polygon[j, 0] - polygon[i, 0]) * (y - polygon[i, 1]) /
                (polygon[j, 1] - polygon[i, 1]) + polygon[i, 0])):
                inside = not inside
            j = i
        
        return inside


    def find_lip_regions(self, mesh, inner_uv_points, margin=0.0005):
        """내부와 외부 입술 영역 검출"""
        start_time = time.time()
        print(f"[시간 측정] 입술 영역 검출 시작")
        
        # 메시 데이터 준비
        prep_start = time.time()
        faces = np.asarray(mesh.faces)
        uvs = np.asarray(mesh.uvs)
        face_uvs = np.asarray(mesh.face_uvs)
        print(f"[시간 측정] 메시 데이터 준비: {time.time() - prep_start:.4f}초")
        
        inner_triangles = set()
        
        # 삼각형 처리
        triangle_start = time.time()
        total_triangles = len(faces)
        print(f"[시간 측정] 총 {total_triangles}개의 삼각형 처리 시작")
        
        for i in range(len(faces)):
            # 삼각형의 UV 좌표 가져오기
            triangle_uvs_group = uvs[face_uvs[i]]  # face_uvs[i]를 사용하여 UV 인덱스 참조
            # UV 중심점 계산
            center = np.mean(triangle_uvs_group, axis=0)
            
            # 내부 입술 영역 체크
            if self.is_point_in_polygon(center, inner_uv_points):
                inner_triangles.add(i)
        
        print(f"[시간 측정] 삼각형 처리 완료: {time.time() - triangle_start:.4f}초")
        print(f"[시간 측정] 선택된 삼각형 수: {len(inner_triangles)}")
        
        # 버텍스 수집
        vertex_start = time.time()
        inner_vertices = set()
        for triangle_idx in inner_triangles:
            inner_vertices.update(faces[triangle_idx])
        
        print(f"[시간 측정] 버텍스 수집 완료: {time.time() - vertex_start:.4f}초")
        print(f"[시간 측정] 선택된 버텍스 수: {len(inner_vertices)}")
        
        total_time = time.time() - start_time
        print(f"[시간 측정] 입술 영역 검출 총 소요 시간: {total_time:.4f}초")
        
        return list(inner_vertices)

    def find_lip_via_convex_hull(self, inner_uv_points):
        """
        내부 입술 UV 좌표를 이용하여 입술 내부의 메시를 선택합니다.
        
        Args:
            inner_uv_points: 내부 입술 UV 좌표 리스트 [[u1, v1], [u2, v2], ...]
            
        Returns:
            선택된 정점들로 구성된 부분 메시
        """
        start_time = time.time()
        print(f"[시간 측정] 입술 메시 생성 시작")
        
        # UV 좌표를 numpy 배열로 변환
        uv_start = time.time()
        inner_uv_points = np.array(inner_uv_points)
        print(f"[시간 측정] UV 좌표 변환: {time.time() - uv_start:.4f}초")
        
        # 입술 영역 정점 찾기
        region_start = time.time()
        selected_vertices = self.find_lip_regions(self.face_smile_mesh, inner_uv_points)
        print(f"[시간 측정] 입술 영역 정점 찾기: {time.time() - region_start:.4f}초")
        
        print(f"선택된 정점 수: {len(selected_vertices)}")
        
        if len(selected_vertices) == 0:
            print("경고: 선택된 정점이 없습니다!")
            return None
        
        # 선택된 정점 저장
        self.lip_vertices = selected_vertices
        
        # 선택된 정점으로 부분 메시 생성
        mesh_start = time.time()
        lip_mesh = self.face_smile_mesh.extract_mesh_from_vertices(selected_vertices)
        print(f"[시간 측정] 부분 메시 생성: {time.time() - mesh_start:.4f}초")
        
        total_time = time.time() - start_time
        print(f"[시간 측정] 입술 메시 생성 총 소요 시간: {total_time:.4f}초")
        
        return lip_mesh
    
    def align_lip_to_laminate(self, lip_mesh):
        """입술 메시를 라미네이트 메시 위치로 이동시킵니다."""
        # 각 메시의 중심점 계산
        lip_center = np.mean(lip_mesh.vertices, axis=0)
        laminate_center = np.mean(self.laminate_mesh.vertices, axis=0)
        
        # 이동 벡터 계산
        translation = laminate_center - lip_center
        
        # 이동 변환 행렬 생성
        transform_matrix = np.eye(4)
        transform_matrix[:3, 3] = translation
        
        # 메시 정점에 변환 적용
        vertices = lip_mesh.vertices
        # 동차 좌표로 변환 (4xN 행렬)
        vertices_homogeneous = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
        # 변환 적용
        transformed_vertices = np.dot(vertices_homogeneous, transform_matrix.T)
        # 동차 좌표에서 3D 좌표로 변환
        lip_mesh.vertices = transformed_vertices[:, :3]
        
        # 법선 벡터도 변환 (회전만 적용)
        if lip_mesh.normals is not None:
            normals = lip_mesh.normals
            # 법선 벡터는 이동 성분 없이 회전만 적용
            rotation_matrix = transform_matrix[:3, :3]
            lip_mesh.normals = np.dot(normals, rotation_matrix.T)
        
        # 변환 행렬 누적
        self.transform_matrix = np.dot(transform_matrix, self.transform_matrix)
        
        return lip_mesh


    def fast_registration_with_vis(self, source_mesh, target_mesh, vis=None):
        """
        ICP 정합을 3단계로 수행하고 과정을 시각화합니다.
        Returns:
            transformed_source_mesh: 변환된 소스 메시
            transform_matrix: 적용된 변환 행렬
        """
        import open3d as o3d
        import copy
        import time
        import numpy as np
        
        # Mesh를 Open3D PointCloud로 변환
        def mesh_to_pointcloud(mesh):
            # 1. 포인트 클라우드 생성
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(mesh.vertices)
            
            # 2. 법선 벡터 처리
            if mesh.normals is not None:
                pcd.normals = o3d.utility.Vector3dVector(mesh.normals)
            else:
                temp_mesh = o3d.geometry.TriangleMesh()
                temp_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
                temp_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
                temp_mesh.compute_vertex_normals()
                pcd.normals = temp_mesh.vertex_normals
            
            # 3. 법선 방향 추정 및 일관성 확인
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            pcd.orient_normals_consistent_tangent_plane(k=100)
            
            pcd.uniform_down_sample(every_k_points=2)
            
            return pcd
        
        # 시각화 창 생성
        if vis is None and self.visualization:
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name='Registration', width=1920, height=1080)
            opt = vis.get_render_option()
            opt.background_color = np.asarray([0.9, 0.9, 0.9])
            opt.point_size = 2.0
            
            # 카메라 설정 (+y 방향에서 -y 방향을 보도록)
            ctr = vis.get_view_control()
            ctr.set_zoom(0.8)
            ctr.set_front([0, -1, 0])  # -y 방향을 바라봄
            ctr.set_up([0, 0, 1])      # z축이 위쪽
            
            # 카메라 설정 강제 적용
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.1)
        
        # Mesh를 PointCloud로 변환
        print("\nMesh를 PointCloud로 변환 중...")
        source = mesh_to_pointcloud(source_mesh)
        target = mesh_to_pointcloud(target_mesh)
        
        # 소스는 빨간색, 타겟은 파란색으로 설정
        source.paint_uniform_color([1, 0, 0])
        target.paint_uniform_color([0, 0, 1])
        
        if self.visualization:
            # 초기 상태 시각화
            vis.clear_geometries()
            vis.add_geometry(source)
            vis.add_geometry(target)
            
            # 카메라 뷰 리셋
            ctr = vis.get_view_control()
            ctr.set_zoom(0.8)
            ctr.set_front([0, -1, 0])
            ctr.set_up([0, 0, 1])
            
            vis.poll_events()
            vis.update_renderer()
            time.sleep(1)
        
        # ICP 실행
        print("\n1번째 ICP 정합 시작...")
        current_transform = np.eye(4)
        
        for iteration in range(1000):
            result = o3d.pipelines.registration.registration_icp(
                source, target,
                1.0,  # 거리 임계값
                current_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6,
                    relative_rmse=1e-6,
                    max_iteration=1
                )
            )
            
            if iteration % 20 == 0:  # 매 반복마다 시각화
                print(f"  - ICP 반복 {iteration}: fitness = {result.fitness:.6f}")
                
                # 시각화 업데이트
                source_temp = copy.deepcopy(source)
                source_temp.transform(result.transformation)
                if self.visualization:
                    vis.clear_geometries()
                    vis.add_geometry(source_temp)
                    vis.add_geometry(target)
                
                # 매 반복마다 카메라 뷰 리셋
                    ctr = vis.get_view_control()
                    ctr.set_zoom(0.8)
                    ctr.set_front([0, -1, 0])
                    ctr.set_up([0, 0, 1])
                    
                    vis.poll_events()
                    vis.update_renderer()
                    time.sleep(0.05)  # 애니메이션 속도 조절
            
            if np.allclose(result.transformation, current_transform, atol=1e-6):
                print(f"  - ICP 수렴 (반복 {iteration})")
                break
                
            current_transform = result.transformation
        
        print("2번째 ICP 정합 시작...")
        for iteration in range(1000):
            result = o3d.pipelines.registration.registration_icp(
                source, target,
                0.3,  # 거리 임계값
                current_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6,
                    relative_rmse=1e-6,
                    max_iteration=1
                )
            )
            
            if iteration % 20 == 0:  # 매 반복마다 시각화
                print(f"  - ICP 반복 {iteration}: fitness = {result.fitness:.6f}")
                
                # 시각화 업데이트
                source_temp = copy.deepcopy(source)
                source_temp.transform(result.transformation)
                if self.visualization:
                    vis.clear_geometries()
                    vis.add_geometry(source_temp)
                    vis.add_geometry(target)
                
                    # 매 반복마다 카메라 뷰 리셋
                    ctr = vis.get_view_control()
                    ctr.set_zoom(0.8)
                    ctr.set_front([0, -1, 0])
                    ctr.set_up([0, 0, 1])
                    
                    vis.poll_events()
                    vis.update_renderer()
                    time.sleep(0.05)  # 애니메이션 속도 조절
            
            if np.allclose(result.transformation, current_transform, atol=1e-6):
                print(f"  - ICP 수렴 (반복 {iteration})")
                break
                
            current_transform = result.transformation
        
        print("3번째 ICP 정합 시작...")
        for iteration in range(1000):
            result = o3d.pipelines.registration.registration_icp(
                source, target,
                0.05,  # 거리 임계값
                current_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6,
                    relative_rmse=1e-6,
                    max_iteration=1
                )
            )
            
            if iteration % 20 == 0:  # 매 반복마다 시각화
                print(f"  - ICP 반복 {iteration}: fitness = {result.fitness:.6f}")
                
                # 시각화 업데이트
                source_temp = copy.deepcopy(source)
                source_temp.transform(result.transformation)
                if self.visualization:
                    vis.clear_geometries()
                    vis.add_geometry(source_temp)
                    vis.add_geometry(target)
                    
                    # 매 반복마다 카메라 뷰 리셋
                    ctr = vis.get_view_control()
                    ctr.set_zoom(0.8)
                    ctr.set_front([0, -1, 0])
                    ctr.set_up([0, 0, 1])
                    
                    vis.poll_events()
                    vis.update_renderer()
                    time.sleep(0.05)  # 애니메이션 속도 조절
            
            if np.allclose(result.transformation, current_transform, atol=1e-6):
                print(f"  - ICP 수렴 (반복 {iteration})")
                break
                
            current_transform = result.transformation
        
        print("\n=== 정합 완료 ===")
        print(f"최종 fitness: {result.fitness:.6f}")
        
        if self.visualization:
            # 시각화 창을 계속 열어두고 마우스 인터렉션 허용
            while True:
                if not vis.poll_events():
                    break
                vis.update_renderer()
                time.sleep(0.1)
        
        # 변환된 소스 메시 생성
        transformed_source_mesh = copy.deepcopy(source_mesh)
        transformed_source_mesh.vertices = np.dot(
            source_mesh.vertices,
            current_transform[:3, :3].T
        ) + current_transform[:3, 3]
        
        return transformed_source_mesh, current_transform
    

    def run_registration(self):
        # 초기 메시 시각화
        if self.visualization:
            visualize_meshes([self.face_smile_mesh, self.laminate_mesh], ["Face", "Laminate"], title="Initial Meshes")
        
        # Y축 정렬
        self.align_y_axis()
        if self.visualization:
            visualize_meshes([self.face_smile_mesh, self.laminate_mesh], ["Face", "Laminate"], title="After Y-axis Alignment")
        print("Y축 정렬 변환 행렬:")
        print(self.transform_matrix)
        
        # 입술 랜드마크 추출
        inner_uv_points, outer_uv_points = self.find_lip_via_analyze_face_landmarks()
        print("입술 UV 좌표:")
        print("내부:", inner_uv_points)
        print("외부:", outer_uv_points)
        
        # 입술 내부 부분 메시 생성
        lip_mesh = self.find_lip_via_convex_hull(inner_uv_points)
        if lip_mesh is None:
            print("입술 메시 생성 실패")
            return None, None
        
        # 입술 메시를 라미네이트 위치로 이동
        lip_mesh = self.align_lip_to_laminate(lip_mesh)
        
        # 전체 메시에 누적된 변환 적용해보기
        moved_smile_mesh = copy.deepcopy(self.face_smile_mesh)
        # 이동 변환만 적용
        translation = self.transform_matrix[:3, 3]
        moved_smile_mesh.vertices = moved_smile_mesh.vertices + translation
        
        # 최종 결과 시각화
        if self.visualization:
            visualize_meshes([lip_mesh, moved_smile_mesh, self.laminate_mesh], 
                            ["Lip", "Moved Face", "Laminate"], 
                            title="Final Result")
        print("최종 누적 변환 행렬:")
        print(self.transform_matrix)

        # 이제 ICP를 이용하여 메시를 매칭시키자
        transformed_mesh, fast_registration_transform_matrix = self.fast_registration_with_vis(lip_mesh, self.laminate_mesh)

        # 최종 변환 한번에 적용
        final_transform = np.dot(
            fast_registration_transform_matrix, 
            self.transform_matrix)
        
        self.transform_matrix = final_transform
        
        moved_smile_mesh.vertices = np.dot(moved_smile_mesh.vertices, fast_registration_transform_matrix[:3, :3].T) + fast_registration_transform_matrix[:3, 3]
        
        if self.visualization:
            visualize_meshes([transformed_mesh, moved_smile_mesh, self.laminate_mesh], 
                            ["Lip", "Moved Face", "Laminate"], 
                            title="Final Result")
        return final_transform, moved_smile_mesh
        

    def visualize_lip_landmarks(self):
        """
        입술 랜드마크를 이미지에 시각화합니다.
        """
        # 이미지 파일 경로 추출 (obj 파일과 같은 이름의 이미지 파일 가정)
        image_path = self.face_smile_path.replace('.obj', '.png')
        if not os.path.exists(image_path):
            image_path = self.face_smile_path.replace('.obj', '.jpg')
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {self.face_smile_path}의 이미지 버전")
        
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
        
        # 이미지 복사 (원본 보존)
        vis_image = image.copy()
        
        # 입술 랜드마크가 이미 추출되어 있는지 확인
        if not hasattr(self, 'inner_lip_points') or not hasattr(self, 'outer_lip_points'):
            # 랜드마크가 없으면 추출
            inner_uv_points, outer_uv_points = self.find_lip_via_analyze_face_landmarks()
        
        # 내부 입술 랜드마크 시각화 (파란색)
        for point in self.inner_lip_points:
            x, y = point
            cv2.circle(vis_image, (x, y), 3, (255, 0, 0), -1)  # 파란색
        
        # 외부 입술 랜드마크 시각화 (빨간색)
        for point in self.outer_lip_points:
            x, y = point
            cv2.circle(vis_image, (x, y), 3, (0, 0, 255), -1)  # 빨간색
        
        # 내부 입술 랜드마크 연결 (파란색)
        for i in range(len(self.inner_lip_points)):
            pt1 = tuple(self.inner_lip_points[i])
            pt2 = tuple(self.inner_lip_points[(i + 1) % len(self.inner_lip_points)])
            cv2.line(vis_image, pt1, pt2, (255, 0, 0), 1)
        
        # 외부 입술 랜드마크 연결 (빨간색)
        for i in range(len(self.outer_lip_points)):
            pt1 = tuple(self.outer_lip_points[i])
            pt2 = tuple(self.outer_lip_points[(i + 1) % len(self.outer_lip_points)])
            cv2.line(vis_image, pt1, pt2, (0, 0, 255), 1)
        
        # 이미지 저장
        output_dir = os.path.dirname(image_path)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        vis_path = os.path.join(output_dir, f"{base_name}_landmarks.png")
        cv2.imwrite(vis_path, vis_image)
        
        print(f"랜드마크 시각화 저장: {vis_path}")
        
        return vis_path


if __name__ == "__main__":
    face_laminate_registration = FaceLaminateRegistration("../../example/data/FaceScan/Smile/Smile.obj", "../../example/data/smile_arch_half.stl", visualization=True)
    final_transform = face_laminate_registration.run_registration()
    print(final_transform)