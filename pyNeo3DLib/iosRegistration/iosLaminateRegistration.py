import numpy as np
from pyNeo3DLib.fileLoader.mesh import Mesh
import copy
from scipy.spatial import ConvexHull
from pyNeo3DLib.visualization.neovis import visualize_meshes
from pyNeo3DLib.iosRegistration.iosAlignment import IosAlignment
import time

# 멀티프로세싱 지원 추가
import multiprocessing as mp
from functools import partial
# queue 관련 유틸리티 추가
from queue import PriorityQueue
import heapq  # heapq 모듈 추가

class IOSLaminateRegistration:
    def __init__(self, ios_path, laminate_path, visualization=False):
        self.ios_path = ios_path
        self.laminate_path = laminate_path
        self.visualization = visualization
        # Initialize transformation matrix (set as 4x4 matrix)
        self.transform_matrix = np.eye(4)
        # time 모듈 참조를 클래스 내부에 명시적으로 저장
        self.time_module = time
        
        # 초기화시에는 메시를 로드하지 않음
        self.ios_mesh = None
        self.laminate_mesh = None

    def run_registration(self):
        print("ios_laminate_registration")

        # 1. Align IOS mesh
        alignmented_ios_mesh, transform_matrix = IosAlignment(self.ios_path).run_analysis()
        
        # PyVista PolyData 객체를 Mesh 객체로 변환
        self.ios_mesh = Mesh()
        # points는 PyVista의 정점 배열, faces는 면 정보
        ios_points = alignmented_ios_mesh.points
        ios_faces_raw = alignmented_ios_mesh.faces
        
        # PyVista 면 배열 변환 (n, v1, v2, ..., vn 형식에서 [v1, v2, ..., vn] 형식으로)
        ios_faces = []
        i = 0
        while i < len(ios_faces_raw):
            n_vertices = ios_faces_raw[i]
            face = ios_faces_raw[i+1:i+1+n_vertices]
            ios_faces.append(face)
            i += n_vertices + 1
        
        self.ios_mesh.vertices = ios_points
        self.ios_mesh.faces = np.array(ios_faces)
        self.transform_matrix = transform_matrix

        # 2. Load laminate model
        self.__load_laminate_model()

        # 3. Find ray mesh intersection approximate
        selected_mesh = self.find_ray_mesh_intersection_approximate(self.ios_mesh)        
        if self.visualization:
            visualize_meshes(
                [self.ios_mesh, selected_mesh], 
                ["Z Aligned IOS", "Selected Region"], 
                title="IOS Compare"
            )

        # 4. Region growing
        region_growing_mesh = self.region_growing(self.ios_mesh, selected_mesh)
        if self.visualization:
            visualize_meshes(
                [self.ios_mesh, selected_mesh, region_growing_mesh], 
                ["Z Aligned IOS", "Selected Region", "Region Growing"], 
                title="IOS Compare"
            )

        if self.visualization:
            visualize_meshes(
                [self.laminate_mesh, region_growing_mesh], 
                ["Laminate", "Region Growing"], 
                title="IOS Compare"
            )

        # 5. Move to origin
        aligned_laminate_mesh = self.move_mask_to_origin(region_growing_mesh)
        if self.visualization:
            visualize_meshes(
                [self.laminate_mesh, aligned_laminate_mesh], 
                ["Laminate", "Aligned Laminate"], 
                title="IOS Compare"
            )

        print("\n=== 4. Transformation matrix after origin movement ===")
        print(self.transform_matrix)

        # 6. ICP registration
        # 시각화가 활성화된 경우 Open3D visualizer 생성
        vis = None
        if self.visualization:
            import open3d as o3d
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name='ICP Registration', width=1280, height=720)
            opt = vis.get_render_option()
            opt.background_color = np.asarray([0.9, 0.9, 0.9])
            opt.point_size = 2.0
            
            # 카메라 설정
            ctr = vis.get_view_control()
            ctr.set_zoom(0.8)
            ctr.set_front([0, 1, 0])  # +y 방향으로 뷰
            ctr.set_up([0, 0, 1])      # z축이 위쪽
            
            # 카메라 설정 적용
            vis.poll_events()
            vis.update_renderer()
        
        transformed_mesh, fast_registration_transform_matrix = self.fast_registration(aligned_laminate_mesh, self.laminate_mesh, vis)
        
        print("\n=== 5. ICP transformation matrix ===")
        print(fast_registration_transform_matrix)

        # Original IOS mesh with all transformations applied in order
        final_ios_mesh = copy.deepcopy(self.ios_mesh)
        
        # Multiply transformation matrices in order
        # 1. OBB, Y, Z alignment (transform_matrix)
        # 2. Origin movement (translation_matrix)
        # 3. ICP transformation (fast_registration_transform_matrix)
        final_transform = np.dot(fast_registration_transform_matrix, 
                                    self.transform_matrix)
        
        if self.visualization:
            # Final transformation applied once
            final_ios_mesh.vertices = np.dot(
                final_ios_mesh.vertices,
                final_transform[:3, :3].T
            ) + final_transform[:3, 3]

        # 최종 행렬 계산 후 반사 보정
        final_transform = self.correct_reflection(final_transform)

        return final_transform
        

    def __load_laminate_model(self):
        print(f"loading model from {self.ios_path} and {self.laminate_path}")
        self.laminate_mesh = Mesh()
        self.laminate_mesh = self.laminate_mesh.from_file(self.laminate_path)
        print(self.laminate_mesh.faces)
 

    def find_ray_mesh_intersection_approximate(self, mesh):
        """
        Shoot a light from the weight center in the +y direction,
        find the outermost points that intersect with the light, and select a rectangular region.
        
        Range adjustment method:
        1. vertical_angle: Vertical angle range (default 10 degrees)
        2. horizontal_angle: Horizontal angle range (default 60 degrees)
        3. min_y_component: Minimum y-component value (default 0.85, larger means selecting only front faces)
        
        Args:
            mesh: Mesh object
            
        Returns:
            selected_mesh: Selected region Mesh object
        """
        vertices = mesh.vertices
        faces = mesh.faces
        
        # 바운딩 박스의 xyz 축 비율 계산
        bbox_min = np.min(vertices, axis=0)
        bbox_max = np.max(vertices, axis=0)
        bbox_size = bbox_max - bbox_min
        
        # xyz 축 비율 계산 및 로그 출력
        print(f"\n=== 바운딩 박스 축 비율 ===")
        print(f"X축 길이: {bbox_size[0]:.2f}")
        print(f"Y축 길이: {bbox_size[1]:.2f}") 
        print(f"Z축 길이: {bbox_size[2]:.2f}")
        print(f"X:Y:Z 비율 = {bbox_size[0]:.2f}:{bbox_size[1]:.2f}:{bbox_size[2]:.2f}")
        
        
        if(bbox_size[2]/bbox_size[0] > 0.5):
            print("Z축 길이가 X축 길이의 50% 이상입니다. 높이 조절 필요")
            center = np.mean(vertices, axis=0) + np.array([0, 0, bbox_size[2]/5])  
        else:
            center = np.mean(vertices, axis=0)
        
        
        # 1. Calculate weight center
        # center = np.mean(vertices, axis=0) #+ np.min(vertices, axis=0))/2
        
        # 2. Define +y direction vector
        y_direction = np.array([0, 1, 0])
        
        # 3. Calculate direction vectors for each vertex (based on weight center)
        directions = vertices - center
        distances = np.linalg.norm(directions, axis=1)
        
        # 4. Normalize direction vectors
        normalized_directions = directions / distances[:, np.newaxis]
        
        # 5. Set angle range (range adjustment possible here)
        vertical_angle = 2  # Vertical angle range
        horizontal_angle = 37  # Horizontal angle range
        min_y_component = 0.80  # Minimum y-component value (cos(approximately 30 degrees))
        
        # 6. Calculate x and z component values
        x_components = np.abs(normalized_directions[:, 0])  # x-axis component
        z_components = np.abs(normalized_directions[:, 2])  # z-axis component
        y_components = normalized_directions[:, 1]  # y-axis component
        
        # 7. Select points based on conditions
        angle_mask = (
            (y_components > min_y_component) &  # 전면만 선택
            (np.abs(normalized_directions[:, 0]) < np.sin(np.radians(horizontal_angle))) &  # 양쪽 수평 범위
            (np.abs(normalized_directions[:, 2]) < np.sin(np.radians(vertical_angle))) &  # 양쪽 수직 범위
            (normalized_directions[:, 1] > 0)  # y+ 방향만 선택
        )
        
        # 8. Group points based on angle
        angle_groups = {}
        for idx in np.where(angle_mask)[0]:
            # Use angle as key (rounded to appropriate precision)
            angle_key = tuple(np.round(normalized_directions[idx], decimals=3))
            
            # Only update if distance is larger in the same angle group
            if angle_key not in angle_groups or distances[idx] > distances[angle_groups[angle_key]]:
                angle_groups[angle_key] = idx
        
        # 9. Select only the farthest points from each angle
        selected_vertices_idx = np.array(list(angle_groups.values()))
        
        # # 8. Selected vertex indices
        # selected_vertices_idx = np.where(angle_mask)[0]
        
        # 9. Find selected faces (select only faces with all selected vertices)
        selected_faces = []
        for face in faces:
            if all(v in selected_vertices_idx for v in face):
                selected_faces.append(face)
        
        selected_faces = np.array(selected_faces)
        
        # 10. Create new vertex index mapping
        vertex_map = {idx: i for i, idx in enumerate(selected_vertices_idx)}
        new_faces = np.array([[vertex_map[v] for v in face] for face in selected_faces])
        
        # 11. Create new mesh
        selected_mesh = Mesh()
        if len(selected_vertices_idx) > 0:
            selected_mesh.vertices = vertices[selected_vertices_idx]
            selected_mesh.faces = new_faces
            if mesh.normals is not None:
                selected_mesh.normals = mesh.normals[selected_vertices_idx]
        
        print(f"Number of selected vertices: {len(selected_vertices_idx)}")
        print(f"Number of created faces: {len(selected_faces)}")
        
        return selected_mesh
    
    def select_region_by_angle(self, mesh, angle_range_x=(-25, 25), angle_range_z=(-5, 5)):
        """
        Select region based on angle range.
        
        Args:
            mesh: Mesh object
            angle_range_x: x-axis angle range (default: -25 degrees ~ 25 degrees)
            angle_range_z: z-axis angle range (default: -5 degrees ~ 5 degrees)
            
        Returns:
            selected_mesh: Selected region Mesh object
        """
        print(f"[Log] Starting region selection based on angle range: X {angle_range_x} degrees, Z {angle_range_z} degrees")
        
        vertices = mesh.vertices
        faces = mesh.faces
        
        # 1. Basic calculation
        weight_center = np.mean(vertices, axis=0)
        vectors = vertices - weight_center
        norms = np.linalg.norm(vectors, axis=1)
        norms[norms == 0] = 1.0
        normalized_vectors = vectors / norms.reshape(-1, 1)
        
        # 2. Select positive Y direction points
        y_components = normalized_vectors[:, 1]
        positive_y_mask = y_components > 0.2  # Select points with y-component greater than 0.2
        
        # 3. Angle-based selection
        selected_vertices = np.zeros(len(vertices), dtype=bool)
        
        for idx in np.where(positive_y_mask)[0]:
            direction = normalized_vectors[idx]
            
            # XY plane projection
            xy_proj = np.array([direction[0], direction[1], 0])
            xy_norm = np.linalg.norm(xy_proj)
            
            # YZ plane projection
            yz_proj = np.array([0, direction[1], direction[2]])
            yz_norm = np.linalg.norm(yz_proj)
            
            if xy_norm > 1e-6 and yz_norm > 1e-6:
                # X angle calculation (XY plane)
                xy_proj_norm = xy_proj / xy_norm
                angle_x = np.degrees(np.arctan2(xy_proj_norm[1], xy_proj_norm[0]) - np.pi/2)
                if angle_x < -90:
                    angle_x += 360
                
                # Z angle calculation (YZ plane)
                yz_proj_norm = yz_proj / yz_norm
                angle_z = np.degrees(np.arctan2(yz_proj_norm[1], yz_proj_norm[2]))
                
                # Angle range check
                if (angle_range_x[0] <= angle_x <= angle_range_x[1] and
                    angle_range_z[0] <= angle_z <= angle_range_z[1]):
                    selected_vertices[idx] = True
        
        # 4. Select faces based on selected vertices
        selected_faces = []
        for i, face in enumerate(faces):
            if np.any(selected_vertices[face]):
                selected_faces.append(i)
        
        # 5. Create new mesh
        selected_faces = np.array(selected_faces)
        used_vertices = np.unique(faces[selected_faces].flatten())
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}
        
        selected_mesh = Mesh()
        selected_mesh.vertices = vertices[used_vertices]
        selected_mesh.faces = np.array([[vertex_map[v] for v in faces[face_idx]] 
                                      for face_idx in selected_faces])
        if mesh.normals is not None:
            selected_mesh.normals = mesh.normals[used_vertices]
        
        print(f"[Log] Number of selected vertices: {len(used_vertices)}")
        print(f"[Log] Number of selected faces: {len(selected_faces)}")
        
        return selected_mesh

    def region_growing(self, mesh, seed_mesh):
        """
        Perform region growing starting from the center point of seed_mesh based on normal vector similarity.
        Use KDTree to efficiently find nearby points.
        
        Args:
            mesh: Full Mesh object
            seed_mesh: Mesh object to use as starting point
            
        Returns:
            grown_mesh: Selected region Mesh object
        """
        from scipy.spatial import KDTree
        import heapq
        
        print("\n=== Region Growing started ===")
        start_time = time.time()
        
        # 1. 메시 데이터 준비 및 검증
        vertices, faces, mesh_stats = self._prepare_mesh_data(mesh)
        print(f"메시 통계: {mesh_stats}")
        
        # 2. 법선 벡터 계산 - 별도 함수로 분리
        vertex_normals = self._compute_vertex_normals(vertices, faces)
        print(f"법선 벡터 계산 완료: {time.time() - start_time:.2f}초")
        
        # 3. 공간 검색 구조 생성
        tree = KDTree(vertices)
        
        # 4. 시드 포인트 찾기
        seed_vertices, seed_stats = self._find_seed_points(tree, seed_mesh)
        print(f"시드 포인트 찾기 완료: {time.time() - start_time:.2f}초")
        print(f"시드 포인트 통계: {seed_stats}")
        
        # 5. 영역 성장 매개변수 설정
        params = self._prepare_region_growing_params(vertices, vertex_normals, seed_vertices)
        
        # 6. 향상된 영역 성장 알고리즘 실행 - 우선순위 큐 사용
        selected_vertices = self._grow_region_with_priority(vertices, vertex_normals, tree, params)
        print(f"영역 성장 완료: {time.time() - start_time:.2f}초")
        
        # 7. 선택된 영역으로 메시 생성
        grown_mesh = self._create_mesh_from_selection(mesh, vertices, faces, selected_vertices, vertex_normals)
        
        # 8. 상단 부분 제거 처리
        result_mesh = self._remove_top_section(grown_mesh)
        
        print("=== Region Growing completed ===")
        print(f"총 소요 시간: {time.time() - start_time:.2f}초\n")
        
        return result_mesh
    
    def _prepare_mesh_data(self, mesh):
        """메시 데이터 준비 및 검증"""
        if mesh is None or not hasattr(mesh, 'vertices') or not hasattr(mesh, 'faces'):
            raise ValueError("유효하지 않은 메시 입력")
        
        vertices = mesh.vertices
        faces = mesh.faces
        
        # 메시 통계 정보
        stats = {
            'vertices': len(vertices),
            'faces': len(faces),
            'bbox_size': np.ptp(vertices, axis=0)
        }
        
        return vertices, faces, stats
    
    def _compute_vertex_normals(self, vertices, faces):
        """정점 법선 벡터 계산 - 벡터화 및 최적화"""
        print("법선 벡터 계산 중...")
        normal_start = time.time()
        
        # 면의 법선 벡터 계산
        v1 = vertices[faces[:, 0]]
        v2 = vertices[faces[:, 1]]
        v3 = vertices[faces[:, 2]]
            
        # 벡터화된 계산
        vec1 = v2 - v1
        vec2 = v3 - v1
        face_normals = np.cross(vec1, vec2)
        
        # 정규화
        face_norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
        face_norms[face_norms < 1e-8] = 1.0  # 0으로 나누기 방지
        face_normals = face_normals / face_norms
            
        # 정점 법선 계산 (메모리 효율적 접근)
        vertex_normals = np.zeros_like(vertices)
        vertex_counts = np.zeros(len(vertices), dtype=np.int32)
            
        # 각 면에서 법선 정보 누적
        for i, face in enumerate(faces):
            for vertex in face:
                vertex_normals[vertex] += face_normals[i]
                vertex_counts[vertex] += 1
            
        # 정규화
        vertex_counts[vertex_counts == 0] = 1  # 0으로 나누기 방지
        vertex_normals = vertex_normals / vertex_counts.reshape(-1, 1)
        
        # 최종 정규화
        norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        vertex_normals = vertex_normals / norms
            
        print(f"법선 계산 완료: {time.time() - normal_start:.2f}초")
        return vertex_normals
        
    def _find_seed_points(self, tree, seed_mesh):
        """시드 포인트 찾기"""
        seed_vertices = seed_mesh.vertices
        
        # 원본 메시에서 가장 가까운 정점 찾기
        distances, indices = tree.query(seed_vertices, k=1)
        
        # 중복 제거하여 시드 정점 집합 생성
        seed_indices = set(indices.flatten())
        
        stats = {
            'seed_vertices': len(seed_vertices),
            'unique_seed_indices': len(seed_indices),
            'avg_distance': np.mean(distances)
        }
        
        return seed_indices, stats
    
    def _prepare_region_growing_params(self, vertices, vertex_normals, seed_indices):
        """영역 성장 매개변수 설정"""
        # 시드 영역의 평균 법선 벡터
        seed_normals = vertex_normals[list(seed_indices)]
        seed_avg_normal = np.mean(seed_normals, axis=0)
        seed_avg_normal = seed_avg_normal / np.linalg.norm(seed_avg_normal)
        
        # 시드 영역의 중심점
        seed_center = np.mean(vertices[list(seed_indices)], axis=0)
        
        # 경계 매개변수 (메시 크기에 따라 동적 조정)
        bbox_diag = np.linalg.norm(np.ptp(vertices, axis=0))
        max_distance = bbox_diag * 0.2  # 메시 전체 크기의 20%로 제한
        
        return {
            'seed_center': seed_center,
            'seed_avg_normal': seed_avg_normal,
            'max_angle_diff': 40.0,  # 법선 벡터 유사도 각도 제한
            'max_distance': max_distance,
            'max_iterations': 100000  # 무한 루프 방지
        }
    
    def _grow_region_with_priority(self, vertices, vertex_normals, tree, params):
        """우선순위 큐를 사용한 향상된 영역 성장 알고리즘"""
        # 초기화
        selected_vertices = np.zeros(len(vertices), dtype=bool)
        seed_center = params['seed_center']
        seed_avg_normal = params['seed_avg_normal']
        
        # 시드 포인트를 찾기 위해 KDTree 사용
        distances, seed_indices = tree.query([seed_center], k=50)
        seed_indices = seed_indices.flatten()
        
        selected_vertices[seed_indices] = True
        
        # 우선순위 큐 기반 처리 (각도 유사성이 높은 순서로 처리)
        priority_queue = []
        visited = np.zeros(len(vertices), dtype=bool)
        visited[seed_indices] = True
        
        # 초기 시드 정점 큐에 추가
        for seed_idx in seed_indices:
            seed_normal = vertex_normals[seed_idx]
            seed_similarity = np.dot(seed_avg_normal, seed_normal)
            # 각도를 우선순위로 사용 (음수로 변환하여 높은 유사성이 먼저 처리되도록)
            heapq.heappush(priority_queue, (-seed_similarity, seed_idx))
        
        iterations = 0
        processed_count = 0
        
        # 영역 성장 실행
        while priority_queue and iterations < params['max_iterations']:
            iterations += 1
            
            # 10000개마다 진행상황 출력
            if iterations % 10000 == 0:
                print(f"영역 성장 진행 중: {iterations}회 반복, {processed_count}개 정점 처리됨")
            
            # 우선순위가 가장 높은(유사도가 높은) 정점 처리
            _, current_vertex = heapq.heappop(priority_queue)
            
            # 이미 처리된 정점이면 스킵
            if not selected_vertices[current_vertex]:
                continue
                
            current_normal = vertex_normals[current_vertex]
            processed_count += 1
            
            # 현재 정점과 시드 중심점과의 거리
            current_distance = np.linalg.norm(vertices[current_vertex] - seed_center)
            if current_distance > params['max_distance'] * 1.5:
                continue
            
            # 주변 정점 탐색 (k=15로 증가하여 더 많은 이웃점 고려)
            distances, neighbors = tree.query(vertices[current_vertex], k=15)
            
            for i, neighbor_idx in enumerate(neighbors):
                # 범위, 이미 방문 여부 확인
                if neighbor_idx >= len(vertices) or selected_vertices[neighbor_idx] or visited[neighbor_idx]:
                    continue
                
                # 거리 제한
                if distances[i] > params['max_distance'] or current_distance > params['max_distance'] * 1.5:
                    continue
                
                neighbor_normal = vertex_normals[neighbor_idx]
                
                # 현재 정점과의 법선 유사도
                current_similarity = np.dot(current_normal, neighbor_normal)
                current_angle_diff = np.degrees(np.arccos(np.clip(current_similarity, -1.0, 1.0)))
                
                # 시드 평균 법선과의 유사도
                seed_similarity = np.dot(seed_avg_normal, neighbor_normal)
                seed_angle_diff = np.degrees(np.arccos(np.clip(seed_similarity, -1.0, 1.0)))
                
                # 조건 확인 (더 엄격하게 조정)
                if current_angle_diff <= params['max_angle_diff'] and seed_angle_diff <= params['max_angle_diff'] * 1.2:
                    selected_vertices[neighbor_idx] = True
                    visited[neighbor_idx] = True
                    
                    # 우선순위 큐에 추가 (각도 유사성 기준)
                    # 음수로 변환하여 높은 유사성이 먼저 처리되도록 함
                    heapq.heappush(priority_queue, (-seed_similarity, neighbor_idx))
        
        print(f"영역 성장 알고리즘 완료: {iterations}회 반복, {processed_count}개 정점 처리, {np.sum(selected_vertices)}개 정점 선택됨")
        return selected_vertices
    
    def _create_mesh_from_selection(self, mesh, vertices, faces, selected_vertices, vertex_normals=None):
        """선택된 정점으로 메시 생성"""
        print("결과 메시 생성 중...")
        mesh_creation_start = time.time()
        
        # 선택된 정점만 포함하는 면 찾기
        selected_faces = []
        for i, face in enumerate(faces):
            if all(selected_vertices[v] for v in face):
                selected_faces.append(i)
        
        if not selected_faces:
            print("경고: 선택된 면이 없습니다. 빈 메시 반환")
            result_mesh = Mesh()
            result_mesh.vertices = np.zeros((0, 3))
            result_mesh.faces = np.zeros((0, 3), dtype=np.int32)
            return result_mesh
        
        # 선택된 면 배열
        selected_faces = np.array(selected_faces)
        
        # 사용된 정점만 추출
        used_vertices = np.unique(faces[selected_faces].flatten())
        
        # 새 인덱스 매핑 생성
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}
        
        # 결과 메시 생성
        result_mesh = Mesh()
        result_mesh.vertices = vertices[used_vertices]
        
        # 면의 정점 인덱스 업데이트
        result_mesh.faces = np.array([[vertex_map[v] for v in faces[face_idx]] 
                                   for face_idx in selected_faces])
        
        # 법선 벡터 전달 (있는 경우)
        if vertex_normals is not None:
            result_mesh.normals = vertex_normals[used_vertices]
        elif mesh.normals is not None:
            result_mesh.normals = mesh.normals[used_vertices]
        
        print(f"결과 메시 생성 완료: 정점 {len(result_mesh.vertices)}개, 면 {len(result_mesh.faces)}개")
        print(f"메시 생성 시간: {time.time() - mesh_creation_start:.2f}초")
        
        return result_mesh
    
    def _remove_top_section(self, mesh, removal_ratio=0.3):
        """상단 부분 제거 처리"""
        if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            print("경고: 빈 메시로 상단 부분 제거를 건너뜁니다.")
            return mesh
            
        print(f"상단 {removal_ratio*100:.0f}% 제거 처리 시작...")
        removal_start = time.time()
        
        try:
            # 원본 메시 백업
            original_mesh = copy.deepcopy(mesh)
            
            # z 좌표 범위 계산
            vertices = mesh.vertices
            z_coords = vertices[:, 2]
            z_min, z_max = np.min(z_coords), np.max(z_coords)
            z_range = z_max - z_min
            
            # z 임계값 계산 (상단 removal_ratio 비율 제거)
            z_threshold = z_max - (z_range * removal_ratio)
            print(f"Z 범위: {z_min:.2f} ~ {z_max:.2f}, 임계값: {z_threshold:.2f}")
            
            # 임계값보다 낮은 정점만 선택
            keep_vertices_mask = z_coords < z_threshold
            selected_count = np.sum(keep_vertices_mask)
            
            # 최소 정점 수 확인
            min_vertices = max(100, int(len(vertices) * 0.3))
            if selected_count < min_vertices:
                print(f"경고: 선택된 정점이 너무 적습니다 ({selected_count} < {min_vertices}). 원본 메시 사용.")
                return original_mesh
            
            # 선택된 정점 집합
            new_vertices = vertices[keep_vertices_mask]
            
            # 법선 벡터 처리
            new_normals = None
            if mesh.normals is not None:
                new_normals = mesh.normals[keep_vertices_mask]
            
            # 인덱스 매핑 생성
            old_to_new_idx = np.full(len(vertices), -1, dtype=np.int32)
            old_to_new_idx[keep_vertices_mask] = np.arange(selected_count)
            
            # 유효한 면만 선택
            new_faces = []
            for face in mesh.faces:
                # 모든 정점이 유지되는 면만 선택
                if all(v < len(keep_vertices_mask) and keep_vertices_mask[v] for v in face):
                    new_face = [old_to_new_idx[v] for v in face]
                    new_faces.append(new_face)
            
            # 최소 면 수 확인
            if len(new_faces) < 50:
                print(f"경고: 선택된 면이 너무 적습니다 ({len(new_faces)} < 50). 원본 메시 사용.")
                return original_mesh
            
            # 결과 메시 생성
            result_mesh = Mesh()
            result_mesh.vertices = new_vertices
            result_mesh.faces = np.array(new_faces)
            if new_normals is not None:
                result_mesh.normals = new_normals
            
            print(f"상단 부분 제거 완료: 정점 {len(mesh.vertices)} -> {len(result_mesh.vertices)}")
            print(f"면 수: {len(mesh.faces)} -> {len(result_mesh.faces)}")
            print(f"처리 시간: {time.time() - removal_start:.2f}초")
            
            return result_mesh
            
        except Exception as e:
            import traceback
            print(f"상단 부분 제거 중 오류 발생: {str(e)}")
            print(traceback.format_exc())
            return mesh  # 오류 발생 시 원본 반환

    def move_mask_to_origin(self, mask_mesh):
        """
        Move the mask mesh so that the point with the largest +y value is at y=0,
        and the point with the smallest -z value is at z=0.
        Transformation matrix is also updated.
        
        Args:
            mask_mesh: Mesh object to move
            
        Returns:
            aligned_mesh: Moved Mesh object
            translation_matrix: Applied transformation matrix
        """
        import time
        move_start_time = time.time()
        
        print(f"\n=== Starting move_mask_to_origin ===")
        print(f"Input mesh: {len(mask_mesh.vertices)} 정점, {len(mask_mesh.faces)} 면")
        
        try:
            # 메시 검증
            if len(mask_mesh.vertices) == 0 or len(mask_mesh.faces) == 0:
                print(f"경고: 빈 메시가 입력되었습니다. 원본 메시를 반환합니다.")
                return mask_mesh, np.eye(4)
                
            vertices = mask_mesh.vertices
            faces = mask_mesh.faces
            
            # 1. Find the largest value in the y direction (point with the largest +y value)
            max_y = np.max(vertices[:, 1])
            
            # 2. Find the smallest value in the z direction (point with the smallest -z value)
            min_z = np.min(vertices[:, 2])
            
            # 3. Calculate translation vector
            # y=0 is needed by moving -max_y
            # z=0 is needed by moving -min_z
            translation = np.array([0, -max_y, -min_z])
            
            print(f"[Log] Translation vector: {translation}")
            
            # 4. Move vertices
            aligned_vertices = vertices + translation
            
            # 5. Create new mesh
            aligned_mesh = Mesh()
            aligned_mesh.vertices = aligned_vertices
            aligned_mesh.faces = faces
            if mask_mesh.normals is not None:
                aligned_mesh.normals = mask_mesh.normals
            
            # 6. Create transformation matrix and update
            translation_matrix = np.eye(4)
            translation_matrix[:3, 3] = translation
            
            # Add new transformation to existing transformation matrix
            self.transform_matrix = np.dot(translation_matrix, self.transform_matrix)
            
            print(f"[Log] Mesh movement completed")
            print(f"  - Y range before movement: [{np.min(vertices[:, 1]):.2f}, {np.max(vertices[:, 1]):.2f}]")
            print(f"  - Y range after movement: [{np.min(aligned_vertices[:, 1]):.2f}, {np.max(aligned_vertices[:, 1]):.2f}]")
            print(f"  - Z range before movement: [{np.min(vertices[:, 2]):.2f}, {np.max(vertices[:, 2]):.2f}]")
            print(f"  - Z range after movement: [{np.min(aligned_vertices[:, 2]):.2f}, {np.max(aligned_vertices[:, 2]):.2f}]")
            print(f"=== move_mask_to_origin 완료: {time.time() - move_start_time:.2f}초 ===\n")
            
            return aligned_mesh
            
        except Exception as e:
            import traceback
            print(f"오류: move_mask_to_origin 중 예외 발생: {str(e)}")
            print(f"상세 오류: {traceback.format_exc()}")
            print(f"=== move_mask_to_origin 실패: {time.time() - move_start_time:.2f}초 ===\n")
            
            # 실패 시 원본 메시와 단위 변환 행렬 반환
            return mask_mesh, np.eye(4)

    def fast_registration(self, source_mesh, target_mesh, vis=None):
        """
        메쉬 등록(registration) 작업을 수행하는 통합 함수입니다.
        여러 초기 위치(현재 위치, x축 +10, x축 -10)에서 ICP를 실행하고
        가장 적합도가 높은 결과를 반환합니다.
        
        Args:
            source_mesh: 소스 메쉬
            target_mesh: 타겟 메쉬
            vis: 시각화 객체 (None인 경우 시각화 없음)
            
        Returns:
            transformed_source_mesh: 변환된 소스 메쉬
            current_transform: 적용된 변환 행렬
        """
        import copy
        import numpy as np
        import open3d as o3d
        import time
        
        # Mesh to Open3D PointCloud conversion
        def mesh_to_pointcloud(mesh):
            # 1. Create point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(mesh.vertices)
            
            # 2. Normal vector assignment or calculation
            if mesh.normals is not None:
                pcd.normals = o3d.utility.Vector3dVector(mesh.normals)
            else:
                temp_mesh = o3d.geometry.TriangleMesh()
                temp_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
                temp_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
                temp_mesh.compute_vertex_normals()
                pcd.normals = temp_mesh.vertex_normals
            
            # 3. Normal vector estimation and consistency check
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            pcd.orient_normals_consistent_tangent_plane(k=100)
            
            return pcd.uniform_down_sample(every_k_points=2)
        
        # 단일 위치에서 ICP 실행 함수
        def run_icp_from_initial_position(source, target, initial_transform, visualizer=None):
            """단일 초기 위치에서 3단계 ICP를 실행하는 함수"""
            current_transform = copy.deepcopy(initial_transform)
            source_pcd = None
            target_pcd = None
            total_iterations = 0  # 총 반복 횟수 카운터
            
            # 시각화 설정
            if visualizer is not None:
                source_pcd = copy.deepcopy(source)
                source_pcd.paint_uniform_color([1, 0, 0])
                
                target_pcd = copy.deepcopy(target)
                target_pcd.paint_uniform_color([0, 0, 1])
                
                # 시각화에 포인트 클라우드 추가
                visualizer.clear_geometries()
                
                # 변환된 소스 추가
                source_transformed = copy.deepcopy(source)
                source_transformed.transform(current_transform)
                source_transformed.paint_uniform_color([1, 0, 0])
                
                visualizer.add_geometry(source_transformed)
                visualizer.add_geometry(target_pcd)
                
                # 카메라 설정
                ctr = visualizer.get_view_control()
                ctr.set_zoom(0.8)
                ctr.set_front([0, 1, 0])  # +y 방향으로 뷰
                ctr.set_up([0, 0, 1])      # z축이 위쪽
                
                visualizer.poll_events()
                visualizer.update_renderer()
            
            # 1단계: 거친 정렬 (coarse alignment)
            print(f"  1단계 ICP 시작...")
            for iteration in range(1000):
                result = o3d.pipelines.registration.registration_icp(
                    source, target,
                    2.0,  # Distance threshold
                    current_transform,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=1e-7,
                        relative_rmse=1e-7,
                        max_iteration=1
                    )
                )
                
                total_iterations += 1  # 반복 횟수 증가
                
                # 시각화 업데이트
                if visualizer is not None and source_pcd is not None and target_pcd is not None:
                    source_transformed = copy.deepcopy(source)
                    source_transformed.transform(current_transform)
                    source_transformed.paint_uniform_color([1, 0, 0])
                    
                    visualizer.clear_geometries()
                    visualizer.add_geometry(source_transformed)
                    visualizer.add_geometry(target_pcd)
                    
                    # 카메라 설정 유지
                    ctr = visualizer.get_view_control()
                    ctr.set_zoom(0.8)
                    ctr.set_front([0, 1, 0])  # +y 방향으로 뷰
                    ctr.set_up([0, 0, 1])      # z축이 위쪽
                    
                    visualizer.poll_events()
                    visualizer.update_renderer()
                
                if np.allclose(result.transformation, current_transform, atol=1e-6):
                    print(f"    1단계 수렴 (반복 {iteration})")
                    break
                    
                current_transform = result.transformation
            
            # 2단계: 중간 정렬 (medium alignment)
            print(f"  2단계 ICP 시작...")
            for iteration in range(1000):
                result = o3d.pipelines.registration.registration_icp(
                    source, target,
                    0.5,  # Distance threshold
                    current_transform,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=1e-7,
                        relative_rmse=1e-7,
                        max_iteration=1
                    )
                )
                
                total_iterations += 1  # 반복 횟수 증가
                
                # 시각화 업데이트
                if visualizer is not None and source_pcd is not None and target_pcd is not None:
                    source_transformed = copy.deepcopy(source)
                    source_transformed.transform(current_transform)
                    source_transformed.paint_uniform_color([1, 0, 0])
                    
                    visualizer.clear_geometries()
                    visualizer.add_geometry(source_transformed)
                    visualizer.add_geometry(target_pcd)
                    
                    # 카메라 설정 유지
                    ctr = visualizer.get_view_control()
                    ctr.set_zoom(0.8)
                    ctr.set_front([0, 1, 0])  # +y 방향으로 뷰
                    ctr.set_up([0, 0, 1])      # z축이 위쪽
                    
                    visualizer.poll_events()
                    visualizer.update_renderer()
                
                if np.allclose(result.transformation, current_transform, atol=1e-6):
                    print(f"    2단계 수렴 (반복 {iteration})")
                    break
                    
                current_transform = result.transformation
            
            # 3단계: 정밀 정렬 (fine alignment)
            print(f"  3단계 ICP 시작...")
            for iteration in range(1000):
                result = o3d.pipelines.registration.registration_icp(
                    source, target,
                    0.3,  # Distance threshold
                    current_transform,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=1e-8,
                        relative_rmse=1e-8,
                        max_iteration=1
                    )
                )
                
                total_iterations += 1  # 반복 횟수 증가
                
                # 시각화 업데이트
                if visualizer is not None and source_pcd is not None and target_pcd is not None:
                    source_transformed = copy.deepcopy(source)
                    source_transformed.transform(current_transform)
                    source_transformed.paint_uniform_color([1, 0, 0])
                    
                    visualizer.clear_geometries()
                    visualizer.add_geometry(source_transformed)
                    visualizer.add_geometry(target_pcd)
                    
                    # 카메라 설정 유지
                    ctr = visualizer.get_view_control()
                    ctr.set_zoom(0.8)
                    ctr.set_front([0, 1, 0])  # +y 방향으로 뷰
                    ctr.set_up([0, 0, 1])      # z축이 위쪽
                    
                    visualizer.poll_events()
                    visualizer.update_renderer()
                
                if np.allclose(result.transformation, current_transform, atol=1e-6):
                    print(f"    3단계 수렴 (반복 {iteration})")
                    break
                    
                current_transform = result.transformation
            
            print(f"  총 반복 횟수: {total_iterations}, 최종 적합도: {result.fitness:.6f}")
            return current_transform, result.fitness, total_iterations
        
        # Mesh to PointCloud conversion
        print("\n메쉬를 포인트 클라우드로 변환 중...")
        source = mesh_to_pointcloud(source_mesh)
        target = mesh_to_pointcloud(target_mesh)
        
        # 서로 다른 초기 위치 설정
        initial_positions = []
        
        # 1. 현재 위치
        current_position = np.eye(4)
        initial_positions.append(("현재 위치", current_position))
        
        # 2. x축 방향 +10 이동
        x_plus_position = np.eye(4)
        x_plus_position[0, 3] = 10.0  # x축 방향으로 +10 이동
        initial_positions.append(("X축 +10 위치", x_plus_position))
        
        # 3. x축 방향 -10 이동
        x_minus_position = np.eye(4)
        x_minus_position[0, 3] = -10.0  # x축 방향으로 -10 이동
        initial_positions.append(("X축 -10 위치", x_minus_position))
        
        # 각 초기 위치에서 ICP 실행 결과 저장
        results = []
        
        for position_name, initial_transform in initial_positions:
            print(f"\n=== {position_name}에서 ICP 등록 시작 ===")
            transform, fitness, total_iterations = run_icp_from_initial_position(source, target, initial_transform, vis)
            results.append((position_name, transform, fitness, total_iterations))
            print(f"=== {position_name} 적합도: {fitness:.6f}, 반복 횟수: {total_iterations} ===")
        
        # 결과가 없는 경우 처리
        if not results:
            print("\n=== 경고: 유효한 등록 결과가 없습니다. 기본 위치 사용 ===")
            return source_mesh, np.eye(4)
            
        # 적합도 기반 최고 결과
        best_by_fitness = max(results, key=lambda x: x[2])
        best_fitness_position, best_fitness_transform, best_fitness, best_fitness_iterations = best_by_fitness
        
        # 반복 횟수 기반 최고 결과
        best_by_iterations = min(results, key=lambda x: x[3])
        best_iterations_position, best_iterations_transform, best_iterations_fitness, best_iterations = best_by_iterations
        
        # 적합도가 비슷한 경우 (상대 차이가 5% 이내) 반복 횟수가 적은 결과 선택
        fitness_threshold = 0.05  # 5% 임계값
        fitness_diff_ratio = abs(best_fitness - best_iterations_fitness) / max(best_fitness, 1e-10)
        
        if fitness_diff_ratio < fitness_threshold:
            print(f"\n적합도 차이가 작음 ({fitness_diff_ratio:.2%}). 반복 횟수가 적은 결과 선택.")
            best_result = best_by_iterations
            best_position_name, best_transform, best_fitness, best_iterations = best_result
            print(f"\n=== 최종 결과 선택: {best_position_name} (적합도: {best_fitness:.6f}, 반복 횟수: {best_iterations}) ===")
        else:
            best_result = best_by_fitness
            best_position_name, best_transform, best_fitness, best_iterations = best_result
            print(f"\n=== 최종 결과 선택: {best_position_name} (적합도: {best_fitness:.6f}, 반복 횟수: {best_iterations}) ===")
        
        # 시각화 창 유지 (필요한 경우)
        if vis is not None:
            print("시각화 창을 닫으려면 창을 닫으세요...")
            try:
                while True:
                    if not vis.poll_events():
                        break
                    vis.update_renderer()
                    time.sleep(0.1)
            except:
                print("시각화가 중단되었습니다.")
        
        # 변환된 소스 메쉬 생성
        transformed_source_mesh = copy.deepcopy(source_mesh)
        transformed_source_mesh.vertices = np.dot(
            source_mesh.vertices,
            best_transform[:3, :3].T
        ) + best_transform[:3, 3]
        
        return transformed_source_mesh, best_transform

    def correct_reflection(self, matrix):
        # 3x3 회전 행렬의 행렬식 계산
        det = np.linalg.det(matrix[:3, :3])
        
        # 행렬식이 음수면 반사 변환이 있음
        if det < 0:
            print(f"반사 변환 감지됨 (행렬식: {det}). 보정 중...")
            # x축 반전 적용 (다른 축을 선택해도 됨)
            reflection_fix = np.eye(4)
            reflection_fix[0, 0] = -1
            return np.dot(reflection_fix, matrix)
        return matrix

if __name__ == "__main__":
    ios_path = "../../example/data/ios_with_smilearch.stl"
    laminate_path = "../../example/data/smile_arch_half.stl"
    
    ios_laminate_registration = IOSLaminateRegistration(ios_path, laminate_path, visualization=True)
    ios_laminate_registration.run_registration()

