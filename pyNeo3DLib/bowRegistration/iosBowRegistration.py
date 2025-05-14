import numpy as np
from pyNeo3DLib.fileLoader.mesh import Mesh
import copy
from scipy.spatial import ConvexHull
from pyNeo3DLib.visualization.neovis import visualize_meshes
import time
from pyNeo3DLib.iosRegistration.iosAlignment import IosAlignment

class IOSBowRegistration:
    def __init__(self, ios_path, centerpin_path, visualization=False):
        self.ios_path = ios_path
        self.centerpin_path = centerpin_path
        self.visualization = visualization
        self.ios_transform_matrix = np.eye(4)
        self.centerpin_transform_matrix = np.eye(4)
        self.__load_centerpin_model()

    def run_registration(self):
        print("ios_centerpin_registration")

        # 1. Align IOS mesh
        alignmented_ios_mesh, transform_matrix = IosAlignment(self.ios_path).run_analysis()
        
        self.ios_mesh = self.__convert_pyvista_mesh_to_mesh(alignmented_ios_mesh)
        self.ios_transform_matrix = transform_matrix

        # 2. Region selection
        selected_mesh = self.find_ray_mesh_intersection_approximate(self.ios_mesh)
        if self.visualization:
            visualize_meshes(
                [self.ios_mesh, selected_mesh], 
                ["Aligned Mesh", "Selected Region"], 
                title="IOS Compare"
            )

        # 3. Region growing
        region_growing_mesh = self.region_growing(self.ios_mesh, selected_mesh)
        if self.visualization:
            visualize_meshes(
                [self.ios_mesh, selected_mesh, region_growing_mesh], 
                ["Aligned Mesh", "Selected Region", "Region Growing"], 
                title="IOS Compare"
            )


        # 4. Move to origin
        moved_origin_centerpin_mesh, centerpin_translation_matrix = self.move_origin_pin_to_found_pin(region_growing_mesh)
        if self.visualization:
            visualize_meshes(
                [region_growing_mesh, moved_origin_centerpin_mesh], 
                ["Found pin", "Moved origin pin"], 
                title="IOS Compare"
            )

        print("\n=== 4. Transformation matrix after origin centerpin movement ===")
        print(self.centerpin_transform_matrix)

        # 7. ICP registration
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
            # ctr.set_zoom(0.8)
            ctr.set_front([0, 1, 1])  # +y 방향으로 뷰
            ctr.set_up([0, 1, 0])      # z축이 위쪽
            
            # 카메라 설정 적용
            vis.poll_events()
            vis.update_renderer()
        
        transformed_mesh, fast_registration_transform_matrix = self.fast_registration(moved_origin_centerpin_mesh, region_growing_mesh, vis)
        
        print("\n=== 5. ICP transformation matrix ===")
        print(fast_registration_transform_matrix)

        # Original IOS mesh with all transformations applied in order
        final_ios_mesh = copy.deepcopy(self.ios_mesh)
        
        # Multiply transformation matrices in order
        # 1. OBB, Y, Z alignment (transform_matrix)
        # 2. Origin movement (translation_matrix)
        # 3. ICP transformation (fast_registration_transform_matrix)
        final_transform = np.dot(fast_registration_transform_matrix, 
                                    self.ios_transform_matrix)
        
        if self.visualization:
            # Final transformation applied once
            final_ios_mesh.vertices = np.dot(
                final_ios_mesh.vertices,
                final_transform[:3, :3].T
            ) + final_transform[:3, 3]

        return final_transform
        

    def __convert_pyvista_mesh_to_mesh(self, pyvista_mesh):
        # PyVista PolyData 객체를 Mesh 객체로 변환
        mesh = Mesh()
        # points는 PyVista의 정점 배열, faces는 면 정보
        points = pyvista_mesh.points
        faces_raw = pyvista_mesh.faces
        
        # PyVista 면 배열 변환 (n, v1, v2, ..., vn 형식에서 [v1, v2, ..., vn] 형식으로)
        faces = []
        i = 0
        while i < len(faces_raw):
            n_vertices = faces_raw[i]
            face = faces_raw[i+1:i+1+n_vertices]
            faces.append(face)
            i += n_vertices + 1
        
        mesh.vertices = points
        mesh.faces = np.array(faces)
        
        return mesh

    def __load_centerpin_model(self):
        print(f"loading model from {self.centerpin_path}")
        self.origin_centerpin_mesh = Mesh()
        self.origin_centerpin_mesh = self.origin_centerpin_mesh.from_file(self.centerpin_path)

        print(self.origin_centerpin_mesh.faces)
    
    def find_ray_mesh_intersection_approximate(self, mesh):
        """
        무게 중심에서 -z 방향으로 광선을 쏘는 것처럼 동작하여,
        해당 방향의 가장 바깥쪽 점들을 찾아 직사각형 영역을 선택합니다.
        이후, 선택된 점들 중 z값이 하위 5%에 있는 정점들과 연결된 면들을 선택합니다.
        
        Args:
            mesh: Mesh 객체
            
        Returns:
            selected_mesh: 선택된 영역의 Mesh 객체
        """
        vertices = mesh.vertices
        faces = mesh.faces
        
        # 1. 무게 중심 계산
        center = np.mean(vertices, axis=0)
        
        # 2. 각 정점까지의 방향 벡터 계산 (무게 중심 기준)
        directions = vertices - center
        distances = np.linalg.norm(directions, axis=1)
        
        # 3. 방향 벡터 정규화 (0으로 나누기 방지)
        normalized_directions = np.zeros_like(directions)
        non_zero_mask = distances > 1e-12
        normalized_directions[non_zero_mask] = directions[non_zero_mask] / distances[non_zero_mask, np.newaxis]
        
        # 4. 각도 범위 및 주 방향 임계값 설정
        angle_spread_y_degrees = 50    # y축 방향 확산 각도 (도)
        angle_spread_x_degrees = 50   # x축 방향 확산 각도 (도)
        primary_direction_component_threshold = 0.4
        
        # 5. 각 방향 성분 계산
        abs_x_components = np.abs(normalized_directions[:, 0])
        abs_y_components = np.abs(normalized_directions[:, 1])
        z_components = normalized_directions[:, 2]
        
        # 6. -z 방향 선택을 위한 조건에 따라 점 선택
        angle_mask = (
            (z_components < -primary_direction_component_threshold) &
            (abs_x_components < np.sin(np.radians(angle_spread_x_degrees))) &
            (abs_y_components < np.sin(np.radians(angle_spread_y_degrees)))
        )
        
        # 7. 각도에 따라 점 그룹화 (바깥쪽 점 선택)
        angle_groups = {}
        for idx in np.where(angle_mask)[0]:
            angle_key = tuple(np.round(normalized_directions[idx], decimals=3))
            if angle_key not in angle_groups or distances[idx] > distances[angle_groups[angle_key]]:
                angle_groups[angle_key] = idx
        
        # 각도 필터링 후 선택된 정점
        filtered_vertices_idx = list(angle_groups.values())
        
        if not filtered_vertices_idx:
            print("각도 기반 필터링 후 선택된 정점이 없습니다.")
            return Mesh()
        
        # 8. 선택된 점들의 z 좌표를 기준으로 하위 5% 선택
        filtered_vertices = vertices[filtered_vertices_idx]
        z_values = filtered_vertices[:, 2]  # z 좌표만 추출
        z_percentile = 1  # 하위 5% 선택 (조절 가능)
        z_threshold = np.percentile(z_values, z_percentile)
        
        # z값이 z_threshold 이하인 정점들 선택
        bottom_layer_mask = z_values <= z_threshold
        bottom_layer_indices = np.array(filtered_vertices_idx)[bottom_layer_mask]
        
        if len(bottom_layer_indices) == 0:
            print(f"z값 하위 {z_percentile}%에 해당하는 정점이 없습니다.")
            return Mesh()
        
        print(f"총 필터링된 정점 수: {len(filtered_vertices_idx)}, 하위 {z_percentile}% z값 정점 수: {len(bottom_layer_indices)}")
        
        # 9. 원본 메쉬의 모든 정점에 대한 마스크 생성 (빠른 참조용)
        bottom_vertices_mask = np.zeros(len(vertices), dtype=bool)
        bottom_vertices_mask[bottom_layer_indices] = True
        
        # 10. 선택된 점들이 하나라도 포함된 면들 선택
        selected_faces_indices = []
        for i, face in enumerate(faces):
            # 면의 정점들 중 하나라도 하위 5% z값 정점이면 선택
            if np.any(bottom_vertices_mask[face]):
                selected_faces_indices.append(i)
        
        if not selected_faces_indices:
            print("선택된 정점들로 구성된 면이 없습니다.")
            return Mesh()
        
        # 11. 선택된 면들에 포함된 모든 정점들 수집
        selected_faces = faces[selected_faces_indices]
        used_vertices_indices = np.unique(selected_faces.flatten())
        
        # 12. 새로운 메쉬 생성
        selected_mesh = Mesh()
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices_indices)}
        
        selected_mesh.vertices = vertices[used_vertices_indices]
        selected_mesh.faces = np.array([[vertex_map[v] for v in face] for face in selected_faces])
        
        if mesh.normals is not None:
            selected_mesh.normals = mesh.normals[used_vertices_indices]
        
        print(f"Number of selected vertices: {len(selected_mesh.vertices)}")
        print(f"Number of created faces: {len(selected_mesh.faces)}")
        
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
        import time
        from scipy.spatial import KDTree
        
        print("\n=== Region Growing started ===")
        start_time = time.time()
        
        vertices = mesh.vertices
        faces = mesh.faces
        
        # 1. Calculate normal vectors
        print("1. Computing normal vectors...")
        normals_start = time.time()
        
        def compute_vertex_normals(vertices, faces):
            # Calculate face normals
            v1 = vertices[faces[:, 0]]
            v2 = vertices[faces[:, 1]]
            v3 = vertices[faces[:, 2]]
            
            vec1 = v2 - v1
            vec2 = v3 - v1
            
            face_normals = np.cross(vec1, vec2)
            face_norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
            face_norms[face_norms == 0] = 1.0
            face_normals = face_normals / face_norms
            
            # Calculate vertex normals (average of face normals)
            vertex_normals = np.zeros_like(vertices)
            vertex_counts = np.zeros(len(vertices))
            
            for i, face in enumerate(faces):
                for vertex in face:
                    vertex_normals[vertex] += face_normals[i]
                    vertex_counts[vertex] += 1
            
            # Normalize
            vertex_counts[vertex_counts == 0] = 1
            vertex_normals = vertex_normals / vertex_counts.reshape(-1, 1)
            norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vertex_normals = vertex_normals / norms
            
            return vertex_normals
        
        vertex_normals = compute_vertex_normals(vertices, faces)
        print(f"  - Normal vectors calculation completed: {time.time() - normals_start:.2f} seconds")
        
        # 2. Create KDTree
        print("2. Creating KDTree...")
        tree_start = time.time()
        tree = KDTree(vertices)
        print(f"  - KDTree creation completed: {time.time() - tree_start:.2f} seconds")
        
        # 3. Find starting points
        print("3. Finding starting points...")
        seed_start = time.time()
        
        # Find seed_mesh vertices and their nearest points in the original mesh
        seed_vertices = seed_mesh.vertices
        distances, indices = tree.query(seed_vertices, k=1)
        start_vertices = set(indices)
        
        print(f"  - Starting points found: {time.time() - seed_start:.2f} seconds")
        print(f"  - Number of starting points: {len(start_vertices)}")
        
        # 4. Region growing parameter setting
        max_angle_diff = 40.0  # 각도 제한을 3.0에서 2.0으로 줄임
        max_distance = np.ptp(vertices, axis=0).max() * 0.2  # 거리 제한을 2%에서 1%로 줄임
        
        # seed 영역의 평균 법선 벡터 계산
        seed_normals = vertex_normals[list(start_vertices)]
        seed_avg_normal = np.mean(seed_normals, axis=0)
        seed_avg_normal = seed_avg_normal / np.linalg.norm(seed_avg_normal)
        
        # seed 영역의 중심점 계산
        seed_center = np.mean(vertices[list(start_vertices)], axis=0)
        
        # 5. Region growing
        selected_vertices = np.zeros(len(vertices), dtype=bool)
        selected_vertices[list(start_vertices)] = True
        queue = list(start_vertices)
        in_queue = np.zeros(len(vertices), dtype=bool)
        in_queue[list(start_vertices)] = True
        
        while queue:
            current_vertex = queue.pop(0)
            current_normal = vertex_normals[current_vertex]
            
            # 현재 점과 seed 중심점과의 거리 계산
            current_distance = np.linalg.norm(vertices[current_vertex] - seed_center)
            
            # Find nearby points
            distances, neighbors = tree.query(vertices[current_vertex], k=10)  # k=20에서 k=10으로 줄임
            
            for i, neighbor_idx in enumerate(neighbors):
                if neighbor_idx >= len(vertices) or selected_vertices[neighbor_idx] or in_queue[neighbor_idx]:
                    continue
                
                # 거리 제한 (현재 점과 seed 중심점과의 거리 고려)
                if distances[i] > max_distance or current_distance > max_distance * 1.5:  # 2배에서 1.5배로 줄임
                    continue
                
                neighbor_normal = vertex_normals[neighbor_idx]
                
                # 1. 현재 점과의 법선 벡터 비교
                current_similarity = np.dot(current_normal, neighbor_normal)
                current_angle_diff = np.degrees(np.arccos(np.clip(current_similarity, -1.0, 1.0)))
                
                # 2. seed 평균 법선 벡터와의 비교
                seed_similarity = np.dot(seed_avg_normal, neighbor_normal)
                seed_angle_diff = np.degrees(np.arccos(np.clip(seed_similarity, -1.0, 1.0)))
                
                # 두 조건 모두 만족해야 선택 (seed_angle_diff 기준을 더 엄격하게)
                if current_angle_diff <= max_angle_diff and seed_angle_diff <= max_angle_diff * 1.2:  # 1.5배에서 1.2배로 줄임
                    selected_vertices[neighbor_idx] = True
                    queue.append(neighbor_idx)
                    in_queue[neighbor_idx] = True
        
        # 6. Select faces based on selected vertices
        print("5. Creating result mesh...")
        result_start_time = time.time()  # 변수명을 더 명확하게 변경
        
        selected_faces = []
        for i, face in enumerate(faces):
            if all(selected_vertices[v] for v in face):
                selected_faces.append(i)
        
        selected_faces = np.array(selected_faces)
        used_vertices = np.unique(faces[selected_faces].flatten())
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}
        
        grown_mesh = Mesh()
        grown_mesh.vertices = vertices[used_vertices]
        
        # 면 인덱스 재매핑 - numpy.int32를 int로 변환하여 인덱싱 문제 방지
        processed_faces = []
        for face_idx in selected_faces:
            # int()로 numpy.int32를 파이썬 int로 변환
            face = faces[int(face_idx)]
            processed_faces.append([vertex_map[v] for v in face])
        
        grown_mesh.faces = np.array(processed_faces)
        
        if mesh.normals is not None:
            grown_mesh.normals = vertex_normals[used_vertices]
        
        print(f"  - Result mesh creation completed: {time.time() - result_start_time:.2f} seconds")
        
        return grown_mesh
    
    def move_origin_pin_to_found_pin(self, found_pin_mesh):
        """
        origin_centerpin_mesh를 found_pin_mesh와 근접하게 이동시키는 함수
        좌표중심을 맞추는 방식으로 이동
        
        Args:
            found_pin_mesh: 타겟 메시
            
        Returns:
            aligned_mesh: 이동된 메시
            translation_matrix: 이동 변환 행렬
        """
        import copy
        import numpy as np
        
        # 좌표중심 계산
        origin_min = np.min(self.origin_centerpin_mesh.vertices, axis=0)
        origin_max = np.max(self.origin_centerpin_mesh.vertices, axis=0)
        origin_center = (origin_min + origin_max) / 2
        
        found_min = np.min(found_pin_mesh.vertices, axis=0)
        found_max = np.max(found_pin_mesh.vertices, axis=0)
        found_center = (found_min + found_max) / 2
        
        # 이동 벡터 계산 (found_center - origin_center)
        translation_vector = found_center - origin_center
        
        # 변환 행렬 생성
        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = translation_vector
        
        # 변환 행렬 저장
        self.centerpin_transform_matrix = translation_matrix
        
        # 이동된 메시 생성
        aligned_mesh = copy.deepcopy(self.origin_centerpin_mesh)
        aligned_mesh.vertices = self.origin_centerpin_mesh.vertices + translation_vector
        
        print(f"센터핀 메시 정렬: 이동 벡터 = {translation_vector}")
        
        return aligned_mesh, translation_matrix

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
            # 1. Create point cloud with uniform sampling across faces
            # Create temporary triangle mesh
            temp_mesh = o3d.geometry.TriangleMesh()
            temp_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
            temp_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
            
            # Sample points uniformly from mesh surface
            pcd = temp_mesh.sample_points_uniformly(number_of_points=len(mesh.vertices)*3)
            
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

        # 최종 결과 시각화
        if vis is not None:
            # 소스와 타겟 포인트 클라우드 복사
            source_pcd = copy.deepcopy(source)
            target_pcd = copy.deepcopy(target)
            
            # 색상 설정
            source_pcd.paint_uniform_color([1, 0, 0])  # 빨간색
            target_pcd.paint_uniform_color([0, 0, 1])  # 파란색
            
            # 최종 변환 적용
            source_pcd.transform(best_transform)
            
            # 시각화 업데이트
            vis.clear_geometries()
            vis.add_geometry(source_pcd)
            vis.add_geometry(target_pcd)
            
            
            vis.poll_events()
            vis.update_renderer()
            
            print("\n최종 정렬 결과가 시각화되었습니다.")
            
            # 시각화 창 유지 (필요한 경우)
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

if __name__ == "__main__":
    ios_bow_registration = IOSBowRegistration("../../example/data/ios_with_smilearch.stl", "../../example/data/center_pin.stl", visualization=True)
    ios_bow_registration.run_registration()
