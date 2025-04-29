import numpy as np
from pyNeo3DLib.fileLoader.mesh import Mesh
import copy
from scipy.spatial import ConvexHull
from pyNeo3DLib.visualization.neovis import visualize_meshes


class IOSLaminateRegistration:
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
        transformed_mesh, fast_registration_transform_matrix = self.fast_registration(aligned_laminate_mesh, self.laminate_mesh)

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
        
        # 1. Calculate weight center
        center = np.mean(vertices, axis=0)
        
        # 2. Define +y direction vector
        y_direction = np.array([0, 1, 0])
        
        # 3. Calculate direction vectors for each vertex (based on weight center)
        directions = vertices - center
        distances = np.linalg.norm(directions, axis=1)
        
        # 4. Normalize direction vectors
        normalized_directions = directions / distances[:, np.newaxis]
        
        # 5. Set angle range (range adjustment possible here)
        vertical_angle = 2  # Vertical angle range
        horizontal_angle = 45  # Horizontal angle range
        min_y_component = 0.80  # Minimum y-component value (cos(approximately 30 degrees))
        
        # 6. Calculate x and z component values
        x_components = np.abs(normalized_directions[:, 0])  # x-axis component
        z_components = np.abs(normalized_directions[:, 2])  # z-axis component
        y_components = normalized_directions[:, 1]  # y-axis component
        
        # 7. Select points based on conditions
        angle_mask = (
            (y_components > min_y_component) &  # Select only front faces
            (x_components < np.sin(np.radians(horizontal_angle))) &  # Horizontal range
            (z_components < np.sin(np.radians(vertical_angle))) &  # Vertical range
            (normalized_directions[:, 1] > 0)  # Select only +y direction
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
            distances, neighbors = tree.query(vertices[current_vertex], k=20)  # k=20에서 k=10으로 줄임
            
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
        
        print("=== Region Growing completed ===")
        print(f"Number of selected vertices: {len(used_vertices)}")
        print(f"Number of selected faces: {len(selected_faces)}")
        print(f"Total time taken: {time.time() - start_time:.2f} seconds\n")
        
        return grown_mesh
    
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
        
        return aligned_mesh, translation_matrix

    def fast_registration(self, source_mesh, target_mesh, vis=None):
        if self.visualization:
            return self.fast_registration_with_vis(source_mesh, target_mesh, vis)
        else:
            return self.fast_registration_without_vis(source_mesh, target_mesh, vis)

    def fast_registration_without_vis(self, source_mesh, target_mesh, vis=None):
        """
        Perform ICP registration in 3 steps and visualize the process.
        Returns:
            transformed_source_mesh: Transformed source mesh
            transform_matrix: Applied transformation matrix
        """
        import open3d as o3d
        import copy
        import time
        import numpy as np
        
        # Mesh to Open3D PointCloud conversion
        def mesh_to_pointcloud(mesh):
            # 1. Create point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(mesh.vertices)
            
            # 2. Normal vector processing
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
            
            pcd.uniform_down_sample(every_k_points=2)
            
            return pcd
        
        # Mesh to PointCloud conversion
        print("\nConverting Mesh to PointCloud...")
        source = mesh_to_pointcloud(source_mesh)
        target = mesh_to_pointcloud(target_mesh)
        
        # ICP execution
        print("\nStarting 1st ICP registration...")
        current_transform = np.eye(4)
        
        for iteration in range(1000):
            result = o3d.pipelines.registration.registration_icp(
                source, target,
                3.0,  # Distance threshold
                current_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-7,
                    relative_rmse=1e-7,
                    max_iteration=1
                )
            )
            
            if np.allclose(result.transformation, current_transform, atol=1e-6):
                print(f"  - ICP converged (iteration {iteration})")
                break
                
            current_transform = result.transformation
        
        print("Starting 2nd ICP registration...")
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
            
            
            if np.allclose(result.transformation, current_transform, atol=1e-6):
                print(f"  - ICP converged (iteration {iteration})")
                break
                
            current_transform = result.transformation
        
        print("Starting 3rd ICP registration...")
        for iteration in range(1000):
            result = o3d.pipelines.registration.registration_icp(
                source, target,
                0.01,  # Distance threshold
                current_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-8,
                    relative_rmse=1e-8,
                    max_iteration=1
                )
            )
            
            
            if np.allclose(result.transformation, current_transform, atol=1e-6):
                print(f"  - ICP converged (iteration {iteration})")
                break
                
            current_transform = result.transformation
        
        print("\n=== Registration completed ===")
        print(f"Final fitness: {result.fitness:.6f}")
        
        # Create transformed source mesh
        transformed_source_mesh = copy.deepcopy(source_mesh)
        transformed_source_mesh.vertices = np.dot(
            source_mesh.vertices,
            current_transform[:3, :3].T
        ) + current_transform[:3, 3]
        
        return transformed_source_mesh, current_transform



    def fast_registration_with_vis(self, source_mesh, target_mesh, vis=None):
        """
        Perform ICP registration in 3 steps and visualize the process.
        Returns:
            transformed_source_mesh: Transformed source mesh
            transform_matrix: Applied transformation matrix
        """
        import open3d as o3d
        import copy
        import time
        import numpy as np
        
        # Mesh to Open3D PointCloud conversion
        def mesh_to_pointcloud(mesh):
            # 1. Create point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(mesh.vertices)
            
            # 2. Normal vector processing
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
            
            pcd.uniform_down_sample(every_k_points=2)
            
            return pcd
        
        # Create visualization window
        if vis is None:
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name='Registration', width=1280, height=720)
            opt = vis.get_render_option()
            opt.background_color = np.asarray([0.9, 0.9, 0.9])
            opt.point_size = 2.0
            
            # Camera setting (+y direction to -y direction view)
            ctr = vis.get_view_control()
            ctr.set_zoom(0.8)
            ctr.set_front([0, -1, 0])  # View -y direction
            ctr.set_up([0, 0, 1])      # z-axis is up
            
            # Force camera setting application
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.1)
        
        
        # Converting Mesh to PointCloud
        print("\nConverting Mesh to PointCloud...")
        source = mesh_to_pointcloud(source_mesh)
        target = mesh_to_pointcloud(target_mesh)
        
        # Set source to red, target to blue
        source.paint_uniform_color([1, 0, 0])
        target.paint_uniform_color([0, 0, 1])
        
        # Visualize initial state
        vis.clear_geometries()
        vis.add_geometry(source)
        vis.add_geometry(target)
        
        # Reset camera view
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        ctr.set_front([0, -1, 0])
        ctr.set_up([0, 0, 1])
        
        vis.poll_events()
        vis.update_renderer()
        time.sleep(1)
        
        # Execute ICP
        print("\nStarting 1st ICP registration...")
        current_transform = np.eye(4)
        
        for iteration in range(1000):
            result = o3d.pipelines.registration.registration_icp(
                source, target,
                3.0,  # Distance threshold
                current_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6,
                    relative_rmse=1e-6,
                    max_iteration=1
                )
            )
            
            if iteration % 20 == 0:  # Visualize every iteration
                print(f"  - ICP iteration {iteration}: fitness = {result.fitness:.6f}")
                
                # Update visualization
                source_temp = copy.deepcopy(source)
                source_temp.transform(result.transformation)
                vis.clear_geometries()
                vis.add_geometry(source_temp)
                vis.add_geometry(target)
                
                # Reset camera view every iteration
                ctr = vis.get_view_control()
                ctr.set_zoom(0.8)
                ctr.set_front([0, -1, 0])
                ctr.set_up([0, 0, 1])
                
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.05)  # Control animation speed
            
            if np.allclose(result.transformation, current_transform, atol=1e-6):
                print(f"  - ICP converged (iteration {iteration})")
                break
                
            current_transform = result.transformation
        
        print("Starting 2nd ICP registration...")
        for iteration in range(1000):
            result = o3d.pipelines.registration.registration_icp(
                source, target,
                0.3,  # Distance threshold
                current_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6,
                    relative_rmse=1e-6,
                    max_iteration=1
                )
            )
            
            if iteration % 20 == 0:  # Visualize every iteration
                print(f"  - ICP iteration {iteration}: fitness = {result.fitness:.6f}")
                
                # Update visualization
                source_temp = copy.deepcopy(source)
                source_temp.transform(result.transformation)
                vis.clear_geometries()
                vis.add_geometry(source_temp)
                vis.add_geometry(target)
                
                # Reset camera view every iteration
                ctr = vis.get_view_control()
                ctr.set_zoom(0.8)
                ctr.set_front([0, -1, 0])
                ctr.set_up([0, 0, 1])
                
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.05)  # Control animation speed
            
            if np.allclose(result.transformation, current_transform, atol=1e-6):
                print(f"  - ICP converged (iteration {iteration})")
                break
                
            current_transform = result.transformation
        
        print("Starting 3rd ICP registration...")
        for iteration in range(1000):
            result = o3d.pipelines.registration.registration_icp(
                source, target,
                0.05,  # Distance threshold
                current_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-8,
                    relative_rmse=1e-8,
                    max_iteration=1
                )
            )
            
            if iteration % 20 == 0:  # Visualize every iteration
                print(f"  - ICP iteration {iteration}: fitness = {result.fitness:.6f}")
                
                # Update visualization
                source_temp = copy.deepcopy(source)
                source_temp.transform(result.transformation)
                vis.clear_geometries()
                vis.add_geometry(source_temp)
                vis.add_geometry(target)
                
                # Reset camera view every iteration
                ctr = vis.get_view_control()
                ctr.set_zoom(0.8)
                ctr.set_front([0, -1, 0])
                ctr.set_up([0, 0, 1])
                
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.05)  # Control animation speed
            
            if np.allclose(result.transformation, current_transform, atol=1e-6):
                print(f"  - ICP converged (iteration {iteration})")
                break
                
            current_transform = result.transformation
        
        print("\n=== Registration completed ===")
        print(f"Final fitness: {result.fitness:.6f}")
        
        # Keep visualization window open and allow mouse interaction
        while True:
            if not vis.poll_events():
                break
            vis.update_renderer()
            time.sleep(0.1)
        
        # Create transformed source mesh
        transformed_source_mesh = copy.deepcopy(source_mesh)
        transformed_source_mesh.vertices = np.dot(
            source_mesh.vertices,
            current_transform[:3, :3].T
        ) + current_transform[:3, 3]
        
        return transformed_source_mesh, current_transform

if __name__ == "__main__":
    ios_laminate_registration = IOSLaminateRegistration("../../example/data/ios_with_smilearch.stl", "../../example/data/smile_arch_half.stl", visualization=True)
    ios_laminate_registration.run_registration()
