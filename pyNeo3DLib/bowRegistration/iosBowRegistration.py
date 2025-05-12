import numpy as np
from pyNeo3DLib.fileLoader.mesh import Mesh
import copy
from scipy.spatial import ConvexHull
from pyNeo3DLib.visualization.neovis import visualize_meshes


class IOSBowRegistration:
    def __init__(self, ios_path, laminate_path, visualization=False):
        self.ios_path = ios_path
        self.laminate_path = laminate_path
        self.visualization = visualization
        # Initialize transformation matrix (set as 4x4 matrix)
        self.transform_matrix = np.eye(4)

        self.__load_models()

    def run_registration(self):
        print("ios_laminate_registration")
        aligned_ios_mesh, ios_rotation_matrix = self.align_with_obb(self.ios_mesh)
    
        if self.visualization:
        # IOS mesh에 transformation matrix applied
            transformed_ios_mesh = copy.deepcopy(self.ios_mesh)
            transformed_ios_mesh.vertices = np.dot(
                self.ios_mesh.vertices,
                self.transform_matrix[:3, :3].T
            ) + self.transform_matrix[:3, 3]

            visualize_meshes(
                    [aligned_ios_mesh, transformed_ios_mesh, self.ios_mesh], 
                    ["Aligned IOS", "Transformed IOS"], 
                    title="IOS Compare"
                )
        
        
        # Transformation matrix output
        print("\n=== 1. Transformation matrix after OBB alignment ===")
        print(self.transform_matrix)

        # OBB center and weight center calculation
        # obb_center, weight_center = ios_laminate_registration.get_obb_center_and_weight_center(aligned_ios_mesh)
        
        # 2. Y-axis alignment
        y_aligned_ios_mesh, y_rotation_matrix = self.find_y_direction(aligned_ios_mesh)
        
        
        if self.visualization:
            visualize_meshes(
                [aligned_ios_mesh, y_aligned_ios_mesh], 
                ["Aligned IOS", "Y Aligned IOS"], 
                title="IOS Compare"
            )
        
        print("\n=== 2. Transformation matrix after Y-axis alignment ===")
        print(self.transform_matrix)

        # 3. Z-axis alignment
        z_aligned_ios_mesh, z_rotation_matrix = self.find_z_direction(y_aligned_ios_mesh)

        if self.visualization:
            visualize_meshes(
                [y_aligned_ios_mesh, z_aligned_ios_mesh], 
                ["Y Aligned IOS", "Z Aligned IOS"], 
                title="IOS Compare"
            )
        
        print("\n=== 3. Transformation matrix after Z-axis alignment ===")
        print(self.transform_matrix)
        
        # 4. Region selection
        selected_mesh = self.find_ray_mesh_intersection_approximate(z_aligned_ios_mesh)
        if self.visualization:
            visualize_meshes(
                [z_aligned_ios_mesh, selected_mesh], 
                ["Z Aligned IOS", "Selected Region"], 
                title="IOS Compare"
            )

        # 5. Region growing
        region_growing_mesh = self.region_growing(z_aligned_ios_mesh, selected_mesh)
        if self.visualization:
            visualize_meshes(
                [z_aligned_ios_mesh, selected_mesh, region_growing_mesh], 
                ["Z Aligned IOS", "Selected Region", "Region Growing"], 
                title="IOS Compare"
            )

        if self.visualization:
            visualize_meshes(
                [self.laminate_mesh, region_growing_mesh], 
                ["Laminate", "Region Growing"], 
                title="IOS Compare"
            )

        # 6. Move to origin
        aligned_laminate_mesh, translation_matrix = self.move_mask_to_origin(region_growing_mesh)
        if self.visualization:
            visualize_meshes(
                [self.laminate_mesh, aligned_laminate_mesh], 
                ["Laminate", "Aligned Laminate"], 
                title="IOS Compare"
            )

        print("\n=== 4. Transformation matrix after origin movement ===")
        print(self.transform_matrix)

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

        return final_transform
        

    def __load_models(self):
        print(f"loading model from {self.ios_path} and {self.laminate_path}")
        self.ios_mesh = Mesh()
        self.laminate_mesh = Mesh()

        self.ios_mesh = self.ios_mesh.from_file(self.ios_path)
        self.laminate_mesh = self.laminate_mesh.from_file(self.laminate_path)

        print(self.ios_mesh.faces)
        print(self.laminate_mesh.faces)
        return self.ios_mesh, self.laminate_mesh
    
    def find_obb(self, mesh):
        """
        Find the OBB (Oriented Bounding Box) of the mesh.
        
        Args:
            mesh: Mesh object
            
        Returns:
            obb_center: OBB center point
            obb_axes: OBB axes vectors (3x3 matrix)
            obb_extents: OBB size (length in each axis direction)
        """
        # 1. Find convex hull
        hull = ConvexHull(mesh.vertices)
        hull_points = mesh.vertices[hull.vertices]
        
        # 2. Calculate convex hull center point
        hull_center = np.mean(hull_points, axis=0)
        
        # 3. Transform convex hull vertices to be centered around the center point
        centered_points = hull_points - hull_center
        
        # 4. Calculate covariance matrix
        cov_matrix = np.cov(centered_points.T)
        
        # 5. Find principal axes using eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 6. Sort eigenvalues (descending order)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 7. Set OBB axes vectors
        obb_axes = np.zeros((3, 3))
        obb_axes[:, 0] = eigenvectors[:, 0]  # Longest axis
        obb_axes[:, 1] = eigenvectors[:, 1]  # Middle length axis
        obb_axes[:, 2] = eigenvectors[:, 2]  # Shortest axis
        
        # 8. Project convex hull vertices onto OBB axes
        projected_points = np.dot(centered_points, obb_axes)
        
        # 9. Calculate minimum/maximum coordinates in each axis direction
        min_coords = np.min(projected_points, axis=0)
        max_coords = np.max(projected_points, axis=0)
        
        # 10. Calculate OBB size
        obb_extents = max_coords - min_coords
        
        # 11. Calculate OBB center point
        obb_center = hull_center + np.dot((min_coords + max_coords) / 2, obb_axes.T)
        
        return obb_center, obb_axes, obb_extents

    def align_with_obb(self, ios_mesh):
        """
        Align the IOS mesh only based on the OBB axes. The laminate mesh remains in its original position.
        The shortest axis is z-axis, the longest axis is x-axis, and the middle length axis is y-axis.
        
        Specific transformation process:
        1. Calculate OBB of IOS mesh and its center point and axes
        2. Sort mesh based on OBB axes
        
        Args:
            ios_mesh: IOS mesh
            
        Returns:
            aligned_ios_mesh: Sorted IOS mesh
            rotation_matrix: IOS mesh rotation matrix
        """
        # 1. Find OBB
        obb_center, obb_axes, obb_extents = self.find_obb(ios_mesh)
        
        # 2. Transform IOS mesh to be centered around OBB center point
        ios_vertices_centered = ios_mesh.vertices - obb_center
        
        # 3. Sort mesh based on OBB axes
        ios_vertices_aligned = np.dot(ios_vertices_centered, obb_axes)
        
        # Create sorted IOS mesh
        aligned_ios_mesh = Mesh()
        aligned_ios_mesh.vertices = ios_vertices_aligned
        aligned_ios_mesh.faces = ios_mesh.faces
        aligned_ios_mesh.normals = ios_mesh.normals
        
        # Result storage
        self.aligned_ios_mesh = aligned_ios_mesh
        self.rotation_matrix = obb_axes  # Use OBB axes as rotation matrix
        
        # Update transformation matrix
        # 1. Create transformation matrix for center point movement
        center_translation = np.eye(4)
        center_translation[:3, 3] = -obb_center
        
        # 2. Expand rotation matrix to 4x4 matrix
        rotation_4x4 = np.eye(4)
        rotation_4x4[:3, :3] = obb_axes
        
        # 3. Accumulate transformation matrix (rotate then move center point)
        self.transform_matrix = np.dot(rotation_4x4, center_translation)
        
        return aligned_ios_mesh, obb_axes
    
    
   
    
    def find_y_direction(self, mesh):
        """
        Find the y-axis direction of the mesh.
        The direction is determined based on the relative position of the weight center and OBB center.
        
        Args:
            mesh: Mesh object
            
        Returns:
            aligned_mesh: Mesh object with y-axis aligned
            rotation_matrix: Applied rotation matrix
        """
        vertices = mesh.vertices
        faces = mesh.faces
        
        # 1. Calculate weight center and OBB center
        weight_center = np.mean(vertices, axis=0)
        
        # OBB center calculation
        # Move vertices to be centered around the center point
        centered_vertices = vertices - weight_center
        # Calculate covariance matrix
        cov_matrix = np.cov(centered_vertices.T)
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        # Project vertices to be aligned with principal axes
        projected_vertices = np.dot(centered_vertices, eigenvectors)
        # Calculate minimum/maximum coordinates in each axis direction
        min_coords = np.min(projected_vertices, axis=0)
        max_coords = np.max(projected_vertices, axis=0)
        # OBB center calculation (average of minimum/maximum coordinates)
        obb_center_projected = (min_coords + max_coords) / 2
        # Convert OBB center point to original coordinate system
        obb_center = np.dot(obb_center_projected, eigenvectors.T) + weight_center
        
        # 2. Calculate y-axis direction based on y-coordinate difference between weight center and OBB center
        y_direction = 1 if weight_center[1] > obb_center[1] else -1
        
        # 3. Create rotation matrix
        rotation_matrix = np.eye(3)
        if y_direction == -1:
            # Rotate 180 degrees if y-axis direction is opposite
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, 1]
            ])
        
        # 4. Sort mesh
        aligned_vertices = np.dot(vertices, rotation_matrix.T)
        
        # Create sorted mesh
        aligned_mesh = Mesh()
        aligned_mesh.vertices = aligned_vertices
        aligned_mesh.faces = mesh.faces
        aligned_mesh.normals = mesh.normals
        
        # 5. Update transformation matrix
        rotation_4x4 = np.eye(4)
        rotation_4x4[:3, :3] = rotation_matrix
        self.transform_matrix = np.dot(rotation_4x4, self.transform_matrix)
        
        return aligned_mesh, rotation_matrix
    
    def find_z_direction(self, mesh, upper_jaw=True):
        """
        Find the z-axis direction of the mesh.
        The direction is determined based on the relative position of the weight center and OBB center.
        
        Args:
            mesh: Mesh object
            upper_jaw: True for upper jaw (teeth direction is down), False for lower jaw (teeth direction is up)
            
        Returns:
            aligned_mesh: Mesh object with z-axis aligned
            rotation_matrix: Applied rotation matrix
        """
        vertices = mesh.vertices
        faces = mesh.faces
        
        # 1. Calculate weight center and OBB center
        weight_center = np.mean(vertices, axis=0)
        
        # OBB center calculation
        # Move vertices to be centered around the center point
        centered_vertices = vertices - weight_center
        # Calculate covariance matrix
        cov_matrix = np.cov(centered_vertices.T)
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        # Project vertices to be aligned with principal axes
        projected_vertices = np.dot(centered_vertices, eigenvectors)
        # Calculate minimum/maximum coordinates in each axis direction
        min_coords = np.min(projected_vertices, axis=0)
        max_coords = np.max(projected_vertices, axis=0)
        # OBB center calculation (average of minimum/maximum coordinates)
        obb_center_projected = (min_coords + max_coords) / 2
        # Convert OBB center point to original coordinate system
        obb_center = np.dot(obb_center_projected, eigenvectors.T) + weight_center
        
        # 2. Calculate z-axis direction based on z-coordinate difference between weight center and OBB center
        # weight center is above OBB center if weight_up is True
        weight_up = weight_center[2] > obb_center[2]
        
        # 3. Determine rotation based on upper_jaw and weight_up status
        # Rotation needed if upper_jaw is True and weight_up is True (upper jaw and weight center above)
        # Rotation needed if upper_jaw is False and weight_up is False (lower jaw and weight center below)
        need_rotation = (upper_jaw and weight_up) or (not upper_jaw and not weight_up)
        
        # 4. Create rotation matrix
        rotation_matrix = np.eye(3)
        if need_rotation:
            # Only z-axis is reversed (y-axis remains unchanged)
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, -1]
            ])
        
        # 5. Sort mesh
        aligned_vertices = np.dot(vertices, rotation_matrix.T)
        
        # Create sorted mesh
        aligned_mesh = Mesh()
        aligned_mesh.vertices = aligned_vertices
        aligned_mesh.faces = mesh.faces
        aligned_mesh.normals = mesh.normals
        
        # 6. Update transformation matrix
        rotation_4x4 = np.eye(4)
        rotation_4x4[:3, :3] = rotation_matrix
        self.transform_matrix = np.dot(rotation_4x4, self.transform_matrix)
        
        return aligned_mesh, rotation_matrix
    
    def create_ball(self, radius, center):
        """
        Create a sphere with given radius and center point.
        
        Args:
            radius: Sphere radius
            center: Sphere center point coordinates (numpy array)
            
        Returns:
            ball_mesh: Mesh object representing the sphere
        """
        # Create vertices and faces to represent the sphere
        phi = np.linspace(0, np.pi, 20)
        theta = np.linspace(0, 2 * np.pi, 20)
        phi, theta = np.meshgrid(phi, theta)

        # Convert from spherical coordinates to Cartesian coordinates
        x = radius * np.sin(phi) * np.cos(theta) + center[0]
        y = radius * np.sin(phi) * np.sin(theta) + center[1]
        z = radius * np.cos(phi) + center[2]

        # Create vertex array
        vertices = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

        # Create faces
        faces = []
        n_phi, n_theta = phi.shape
        for i in range(n_phi - 1):
            for j in range(n_theta - 1):
                v1 = i * n_theta + j
                v2 = i * n_theta + (j + 1)
                v3 = (i + 1) * n_theta + (j + 1)
                v4 = (i + 1) * n_theta + j
                faces.append([v1, v2, v3])
                faces.append([v1, v3, v4])

        # Create Mesh object
        ball_mesh = Mesh()
        ball_mesh.vertices = vertices
        ball_mesh.faces = np.array(faces)
        
        return ball_mesh
    
    def find_ray_mesh_intersection_approximate(self, mesh):
        """
        무게 중심에서 -z 방향으로 광선을 쏘는 것처럼 동작하여,
        해당 방향의 가장 바깥쪽 점들을 찾아 직사각형 영역을 선택합니다.
        
        범위 조정 방법 (내부 하드코딩된 값):
        1. angle_spread_y_degrees: y축 방향 확산 각도 (기본값 2도)
        2. angle_spread_x_degrees: x축 방향 확산 각도 (기본값 37도)
        3. primary_direction_component_threshold: -z 방향 선택을 위한 z 성분의 최소 절댓값 임계값.
                                                 (예: 0.80은 정규화된 방향 벡터의 z 성분이 -0.80보다 작아야 함을 의미)
        
        Args:
            mesh: Mesh 객체
            
        Returns:
            selected_mesh: 선택된 영역의 Mesh 객체
        """
        vertices = mesh.vertices
        faces = mesh.faces
        
        # 1. 무게 중심 계산
        center = np.mean(vertices, axis=0)
        
        # 2. -z 방향 벡터 정의 (개념적이며, 마스크 생성에 직접 사용되지는 않음)
        # z_neg_direction = np.array([0, 0, -1])
        
        # 3. 각 정점까지의 방향 벡터 계산 (무게 중심 기준)
        directions = vertices - center
        distances = np.linalg.norm(directions, axis=1)
        
        # 4. 방향 벡터 정규화
        # 거리가 0인 경우 (정점이 정확히 중앙에 있는 경우) NaN 값 방지를 위해 처리합니다.
        # 이러한 점들은 [0,0,0] 정규화된 방향을 가지며, angle_mask 조건에 의해 선택되지 않습니다.
        normalized_directions = np.zeros_like(directions)
        non_zero_dist_mask = distances > 1e-12 # 매우 작은 엡실론 사용
        # 거리가 0이 아닌 경우에만 나누기 연산 수행
        if np.any(non_zero_dist_mask):
            normalized_directions[non_zero_dist_mask] = \
                directions[non_zero_dist_mask] / distances[non_zero_dist_mask, np.newaxis]

        # 5. 각도 범위 및 주 방향 임계값 설정
        angle_spread_y_degrees = 50    # y축 방향 확산 각도 (도)
        angle_spread_x_degrees = 50   # x축 방향 확산 각도 (도)
        # 이 값은 정규화된 방향 벡터의 z 성분이 -0.80보다 작아야 함을 의미합니다.
        primary_direction_component_threshold = 0.80 
        
        # 6. 정규화된 방향 벡터로부터 관련 성분 값 계산
        # x축 확산을 위한 x 성분 절댓값
        abs_x_components = np.abs(normalized_directions[:, 0])
        # y축 확산을 위한 y 성분 절댓값
        abs_y_components = np.abs(normalized_directions[:, 1])
        # 주 방향(-z) 선택을 위한 z 성분
        z_components = normalized_directions[:, 2]
        
        # 7. -z 방향 선택을 위한 조건에 따라 점 선택
        angle_mask = (
            (z_components < -primary_direction_component_threshold) &  # -z 방향으로 강하게 향하는 점 선택
            (abs_x_components < np.sin(np.radians(angle_spread_x_degrees))) &  # x축 확산 제한
            (abs_y_components < np.sin(np.radians(angle_spread_y_degrees)))    # y축 확산 제한
        )
        
        # 8. 각도에 따라 점 그룹화
        angle_groups = {}
        for idx in np.where(angle_mask)[0]:
            # 각도를 키로 사용 (적절한 정밀도로 반올림)
            angle_key = tuple(np.round(normalized_directions[idx], decimals=3))
            
            # 동일한 각도 그룹 내에서 거리가 더 먼 경우에만 업데이트
            if angle_key not in angle_groups or distances[idx] > distances[angle_groups[angle_key]]:
                angle_groups[angle_key] = idx
        
        # 9. 각 각도에서 가장 멀리 있는 점들만 선택
        selected_vertices_idx = np.array(list(angle_groups.values()))
        
        # # 8. 선택된 정점 인덱스 (이전 방식)
        # selected_vertices_idx = np.where(angle_mask)[0]
        
        # 9. 선택된 면 찾기 (모든 정점이 선택된 면만 선택)
        selected_faces = []
        for face in faces:
            if all(v in selected_vertices_idx for v in face):
                selected_faces.append(face)
        
        selected_faces = np.array(selected_faces)
        
        # 10. 새로운 정점 인덱스 매핑 생성
        vertex_map = {idx: i for i, idx in enumerate(selected_vertices_idx)}
        new_faces = np.array([[vertex_map[v] for v in face] for face in selected_faces])
        
        # 11. 새로운 메쉬 생성
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
        grown_mesh.faces = np.array([[vertex_map[v] for v in faces[face_idx]] 
                                   for face_idx in selected_faces])
        if mesh.normals is not None:
            grown_mesh.normals = vertex_normals[used_vertices]
        
        print(f"  - Result mesh creation completed: {time.time() - result_start_time:.2f} seconds")
        
        # 상단 10% 제거 (z축 기준)
        top_removal_start_time = time.time()
        
        try:
            # 1. 원본 메시 백업
            print(f"  - 상단 20% 제거 시작: 원본 메시 정점 수 = {len(grown_mesh.vertices)}, 면 수 = {len(grown_mesh.faces)}")
            original_grown_mesh = copy.deepcopy(grown_mesh)
            
            # 2. 메시의 z 좌표 범위 계산
            z_coords = grown_mesh.vertices[:, 2]  # z 좌표만 추출
            z_min, z_max = np.min(z_coords), np.max(z_coords)
            z_range = z_max - z_min
            
            # 3. 상단 20%에 해당하는 z 좌표 임계값 계산
            z_threshold = z_max - (z_range * 0.3)
            print(f"  - Z 좌표 범위: {z_min:.2f} ~ {z_max:.2f}, 임계값: {z_threshold:.2f}")
            
            # 4. z 좌표가 임계값보다 낮은 정점만 선택 (상단 20% 제외)
            keep_vertices_mask = z_coords < z_threshold
            selected_count = np.sum(keep_vertices_mask)
            
            print(f"  - 상단 20% 제거 후 남은 정점 수: {selected_count} / {len(z_coords)}")
            
            # 5. 선택된 정점 수 확인 - 최소 정점 수는 메시 크기에 비례하도록 설정
            min_vertices = max(100, int(len(grown_mesh.vertices) * 0.3))  # 최소한 원본 메시의 30% 이상이 남아야 함
            
            if selected_count < min_vertices:
                print(f"  - 경고: 상단 20% 제거 후 남은 정점이 너무 적습니다 ({selected_count} < {min_vertices}). 원본 메시를 사용합니다.")
                print(f"  - 상단 20% 제거 시간: {time.time() - top_removal_start_time:.2f}초")
                
                print("=== Region Growing completed ===")
                print(f"Number of selected vertices: {len(used_vertices)}")
                print(f"Number of selected faces: {len(selected_faces)}")
                print(f"Total time taken: {time.time() - start_time:.2f} seconds\n")
                
                return grown_mesh  # 원본 메시 반환
            
            # 6. 선택된 정점만 유지
            print(f"  - 새 정점 배열 생성 중...")
            new_vertices = grown_mesh.vertices[keep_vertices_mask]
            if grown_mesh.normals is not None:
                new_normals = grown_mesh.normals[keep_vertices_mask]
            print(f"  - 새 정점 배열 생성 완료: {len(new_vertices)} 정점")
            
            # 7. 원래 정점 인덱스와 새로운 인덱스 간 매핑 생성
            print(f"  - 정점 인덱스 매핑 생성 중...")
            old_to_new_idx = np.full(len(grown_mesh.vertices), -1, dtype=np.int32)
            old_to_new_idx[keep_vertices_mask] = np.arange(np.sum(keep_vertices_mask))
            print(f"  - 정점 인덱스 매핑 생성 완료")
            
            # 8. 유지된 정점에 대한 페이스만 유지
            print(f"  - 새 페이스 배열 생성 중... 총 {len(grown_mesh.faces)} 개의 페이스 처리")
            new_faces = []
            valid_faces = 0
            
            # 처리 중인 면의 개수를 주기적으로 출력하여 진행 상황 추적
            total_faces = len(grown_mesh.faces)
            checkpoint = max(1, total_faces // 10)  # 10% 단위로 진행 상황 출력
            
            for i, face in enumerate(grown_mesh.faces):
                if i % checkpoint == 0:
                    print(f"  - 페이스 처리 중: {i}/{total_faces} ({i/total_faces*100:.1f}%)")
                
                # 모든 정점이 유지되는 페이스만 선택
                if all(v < len(keep_vertices_mask) for v in face):  # 인덱스 범위 확인
                    if all(keep_vertices_mask[v] for v in face):
                        # 정점 인덱스 업데이트
                        try:
                            new_face = [old_to_new_idx[v] for v in face]
                            if all(idx != -1 for idx in new_face):  # 모든 인덱스가 유효한지 확인
                                new_faces.append(new_face)
                                valid_faces += 1
                        except Exception as e:
                            print(f"  - 경고: 페이스 {i}, 정점 {face}의 인덱스 변환 중 오류: {e}")
            
            print(f"  - 새 페이스 배열 생성 완료: {len(new_faces)} 페이스 (유효: {valid_faces})")
            
            # 9. 페이스 수 확인
            min_faces = max(50, int(len(grown_mesh.faces) * 0.3))  # 최소한 원본 메시의 30% 이상이 남아야 함
            
            if len(new_faces) < min_faces:
                print(f"  - 경고: 상단 20% 제거 후 남은 페이스가 너무 적습니다 ({len(new_faces)} < {min_faces}). 원본 메시를 사용합니다.")
                print(f"  - 상단 20% 제거 시간: {time.time() - top_removal_start_time:.2f}초")
                
                print("=== Region Growing completed ===")
                print(f"Number of selected vertices: {len(used_vertices)}")
                print(f"Number of selected faces: {len(selected_faces)}")
                print(f"Total time taken: {time.time() - start_time:.2f} seconds\n")
                
                return grown_mesh  # 원본 메시 반환
            
            # 10. numpy 배열로 변환
            print(f"  - 페이스 배열을 numpy로 변환 중...")
            if len(new_faces) > 0:
                new_faces = np.array(new_faces)
                print(f"  - 페이스 배열 변환 완료: 형태 {new_faces.shape}")
            else:
                print(f"  - 경고: 변환할 페이스가 없습니다. 원본 메시를 사용합니다.")
                
                print("=== Region Growing completed ===")
                print(f"Number of selected vertices: {len(used_vertices)}")
                print(f"Number of selected faces: {len(selected_faces)}")
                print(f"Total time taken: {time.time() - start_time:.2f} seconds\n")
                
                return grown_mesh  # 원본 메시 반환
            
            # 11. 새로운 메시 생성
            print(f"  - 새 메시 객체 생성 중...")
            top_removed_mesh = Mesh()
            top_removed_mesh.vertices = new_vertices
            top_removed_mesh.faces = new_faces
            if grown_mesh.normals is not None:
                top_removed_mesh.normals = new_normals
            print(f"  - 새 메시 객체 생성 완료")
            
            print(f"  - 상단 20% 제거 완료: 정점 {len(grown_mesh.vertices)} -> {len(new_vertices)}")
            print(f"  - 상단 20% 제거 완료: 면 {len(grown_mesh.faces)} -> {len(new_faces)}")
            print(f"  - 상단 20% 제거 시간: {time.time() - top_removal_start_time:.2f}초")
            
            # 12. 메시 일관성 검사
            print(f"  - 메시 일관성 검사 중...")
            if (len(top_removed_mesh.vertices) > 0 and len(top_removed_mesh.faces) > 0 and 
                np.max(top_removed_mesh.faces) < len(top_removed_mesh.vertices)):
                print(f"  - 메시 일관성 검사 통과")
                result_mesh = top_removed_mesh
            else:
                print(f"  - 경고: 메시 일관성 검사 실패. 원본 메시를 사용합니다.")
                result_mesh = grown_mesh
            
            print("=== Region Growing completed ===")
            print(f"Number of selected vertices: {len(used_vertices)}")
            print(f"Number of selected faces: {len(selected_faces)}")
            print(f"Total time taken: {time.time() - start_time:.2f} seconds\n")
            
            return result_mesh
            
        except Exception as e:
            import traceback
            print(f"  - 오류: 상단 20% 제거 중 예외 발생: {str(e)}")
            print(f"  - 상세 오류: {traceback.format_exc()}")
            print(f"  - 상단 20% 제거 실패. 원본 메시를 사용합니다.")
            print(f"  - 상단 20% 제거 시간: {time.time() - top_removal_start_time:.2f}초")
            
            print("=== Region Growing completed ===")
            print(f"Number of selected vertices: {len(used_vertices)}")
            print(f"Number of selected faces: {len(selected_faces)}")
            print(f"Total time taken: {time.time() - start_time:.2f} seconds\n")
            
            return grown_mesh  # 원본 메시 반환
    
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
            
            return aligned_mesh, translation_matrix
            
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

if __name__ == "__main__":
    ios_bow_registration = IOSBowRegistration("../../example/data/ios_with_smilearch.stl", "../../example/data/smile_arch_half.stl", visualization=True)
    ios_bow_registration.run_registration()
