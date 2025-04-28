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
        # 변환 행렬 초기화 (4x4 행렬로 설정)
        self.transform_matrix = np.eye(4)

        self.__load_models()

    def run_registration(self):
        print("Starting IOS laminate registration")
        aligned_ios_mesh, ios_rotation_matrix = self.align_with_obb(self.ios_mesh)
    
        if self.visualization:
            # Apply transformation matrix to IOS mesh
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
        
        # Print transformation matrix
        print("\n=== 1. Transformation matrix after OBB alignment ===")
        print(self.transform_matrix)

        # Calculate OBB center and weight center
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

        print("\n=== 4. Transformation matrix after moving to origin ===")
        print(self.transform_matrix)

        # 7. ICP registration
        transformed_mesh, fast_registration_transform_matrix = self.fast_registration(aligned_laminate_mesh, self.laminate_mesh)

        print("\n=== 5. ICP transformation matrix ===")
        print(fast_registration_transform_matrix)

        # Apply all transformations to original IOS mesh in sequence
        final_ios_mesh = copy.deepcopy(self.ios_mesh)
        
        # Multiply transformation matrices in sequence
        # 1. OBB, Y, Z alignment (transform_matrix)
        # 2. Move to origin (translation_matrix)
        # 3. ICP transformation (fast_registration_transform_matrix)
        final_transform = np.dot(fast_registration_transform_matrix, 
                                    self.transform_matrix)
        
        if self.visualization:
            # Apply final transformation at once
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
        메쉬의 OBB(Oriented Bounding Box)를 찾습니다.
        
        Args:
            mesh: Mesh 객체
            
        Returns:
            obb_center: OBB 중심점
            obb_axes: OBB 축 벡터 (3x3 행렬)
            obb_extents: OBB 크기 (각 축 방향의 길이)
        """
        # 1. 볼록 껍질 찾기
        hull = ConvexHull(mesh.vertices)
        hull_points = mesh.vertices[hull.vertices]
        
        # 2. 볼록 껍질의 중심점 계산
        hull_center = np.mean(hull_points, axis=0)
        
        # 3. 볼록 껍질의 정점들을 중심점 기준으로 변환
        centered_points = hull_points - hull_center
        
        # 4. 공분산 행렬 계산
        cov_matrix = np.cov(centered_points.T)
        
        # 5. 고유값 분해로 주축 찾기
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 6. 고유값을 기준으로 정렬 (내림차순)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 7. OBB 축 벡터 설정
        obb_axes = np.zeros((3, 3))
        obb_axes[:, 0] = eigenvectors[:, 0]  # 가장 긴 축
        obb_axes[:, 1] = eigenvectors[:, 1]  # 중간 길이 축
        obb_axes[:, 2] = eigenvectors[:, 2]  # 가장 짧은 축
        
        # 8. 볼록 껍질의 정점들을 OBB 축으로 투영
        projected_points = np.dot(centered_points, obb_axes)
        
        # 9. 각 축 방향의 최소/최대 좌표 계산
        min_coords = np.min(projected_points, axis=0)
        max_coords = np.max(projected_points, axis=0)
        
        # 10. OBB 크기 계산
        obb_extents = max_coords - min_coords
        
        # 11. OBB 중심점 계산
        obb_center = hull_center + np.dot((min_coords + max_coords) / 2, obb_axes.T)
        
        return obb_center, obb_axes, obb_extents

    def align_with_obb(self, ios_mesh):
        """
        OBB 축을 기준으로 IOS 메쉬만 정렬합니다. 라미네이트 메쉬는 원래 위치를 유지합니다.
        가장 짧은 축은 z축, 가장 긴 축은 x축, 중간 길이 축은 y축으로 정렬됩니다.
        
        구체적인 변환 과정:
        1. IOS 메쉬의 OBB를 찾아 중심점과 축을 계산
        2. OBB 축을 기준으로 메쉬를 정렬
        
        Args:
            ios_mesh: IOS 메쉬
            
        Returns:
            aligned_ios_mesh: 정렬된 IOS 메쉬
            rotation_matrix: IOS 메쉬 회전 행렬
        """
        # 1. OBB 찾기
        obb_center, obb_axes, obb_extents = self.find_obb(ios_mesh)
        
        # 2. IOS 메쉬를 OBB 중심점 기준으로 변환
        ios_vertices_centered = ios_mesh.vertices - obb_center
        
        # 3. OBB 축을 기준으로 메쉬 정렬
        ios_vertices_aligned = np.dot(ios_vertices_centered, obb_axes)
        
        # 정렬된 IOS 메쉬 생성
        aligned_ios_mesh = Mesh()
        aligned_ios_mesh.vertices = ios_vertices_aligned
        aligned_ios_mesh.faces = ios_mesh.faces
        aligned_ios_mesh.normals = ios_mesh.normals
        
        # 결과 저장
        self.aligned_ios_mesh = aligned_ios_mesh
        self.rotation_matrix = obb_axes  # OBB 축을 회전 행렬로 사용
        
        # 변환 행렬 업데이트
        # 1. 중심점 이동을 위한 변환 행렬 생성
        center_translation = np.eye(4)
        center_translation[:3, 3] = -obb_center
        
        # 2. 회전 행렬을 4x4 행렬로 확장
        rotation_4x4 = np.eye(4)
        rotation_4x4[:3, :3] = obb_axes
        
        # 3. 변환 행렬 누적 (회전 후 중심점 이동)
        self.transform_matrix = np.dot(rotation_4x4, center_translation)
        
        return aligned_ios_mesh, obb_axes
    
    
   
    
    def find_y_direction(self, mesh):
        """
        메쉬의 y축 방향을 찾습니다.
        weight center와 OBB center의 상대적 위치를 기반으로 방향을 결정합니다.
        
        Args:
            mesh: Mesh 객체
            
        Returns:
            aligned_mesh: y축이 정렬된 Mesh 객체
            rotation_matrix: 적용된 회전 행렬
        """
        vertices = mesh.vertices
        faces = mesh.faces
        
        # 1. weight center와 OBB center 계산
        weight_center = np.mean(vertices, axis=0)
        
        # OBB center 계산
        # 정점을 중심점으로 이동
        centered_vertices = vertices - weight_center
        # 공분산 행렬 계산
        cov_matrix = np.cov(centered_vertices.T)
        # 고유값 분해
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        # 정점을 주축 방향으로 투영
        projected_vertices = np.dot(centered_vertices, eigenvectors)
        # 각 축 방향의 최소/최대 좌표 계산
        min_coords = np.min(projected_vertices, axis=0)
        max_coords = np.max(projected_vertices, axis=0)
        # OBB 중심점 계산 (최소/최대 좌표의 중간점)
        obb_center_projected = (min_coords + max_coords) / 2
        # OBB 중심점을 원래 좌표계로 변환
        obb_center = np.dot(obb_center_projected, eigenvectors.T) + weight_center
        
        # 2. weight center와 OBB center의 y좌표 차이로 방향 결정
        y_direction = 1 if weight_center[1] > obb_center[1] else -1
        
        # 3. 회전 행렬 생성
        rotation_matrix = np.eye(3)
        if y_direction == -1:
            # y축 방향이 반대인 경우 180도 회전
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, 1]
            ])
        
        # 4. 메쉬 정렬
        aligned_vertices = np.dot(vertices, rotation_matrix.T)
        
        # 정렬된 메쉬 생성
        aligned_mesh = Mesh()
        aligned_mesh.vertices = aligned_vertices
        aligned_mesh.faces = mesh.faces
        aligned_mesh.normals = mesh.normals
        
        # 5. 변환 행렬 업데이트
        rotation_4x4 = np.eye(4)
        rotation_4x4[:3, :3] = rotation_matrix
        self.transform_matrix = np.dot(rotation_4x4, self.transform_matrix)
        
        return aligned_mesh, rotation_matrix
    
    def find_z_direction(self, mesh, upper_jaw=True):
        """
        메쉬의 z축 방향을 찾습니다.
        weight center와 OBB center의 상대적 위치를 기반으로 방향을 결정합니다.
        
        Args:
            mesh: Mesh 객체
            upper_jaw: True면 상악(치아 방향이 아래쪽), False면 하악(치아 방향이 위쪽)
            
        Returns:
            aligned_mesh: z축이 정렬된 Mesh 객체
            rotation_matrix: 적용된 회전 행렬
        """
        vertices = mesh.vertices
        faces = mesh.faces
        
        # 1. weight center와 OBB center 계산
        weight_center = np.mean(vertices, axis=0)
        
        # OBB center 계산
        # 정점을 중심점으로 이동
        centered_vertices = vertices - weight_center
        # 공분산 행렬 계산
        cov_matrix = np.cov(centered_vertices.T)
        # 고유값 분해
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        # 정점을 주축 방향으로 투영
        projected_vertices = np.dot(centered_vertices, eigenvectors)
        # 각 축 방향의 최소/최대 좌표 계산
        min_coords = np.min(projected_vertices, axis=0)
        max_coords = np.max(projected_vertices, axis=0)
        # OBB 중심점 계산 (최소/최대 좌표의 중간점)
        obb_center_projected = (min_coords + max_coords) / 2
        # OBB 중심점을 원래 좌표계로 변환
        obb_center = np.dot(obb_center_projected, eigenvectors.T) + weight_center
        
        # 2. weight center와 OBB center의 z좌표 차이로 방향 결정
        # weight center가 OBB center보다 위에 있으면 weight_up은 True
        weight_up = weight_center[2] > obb_center[2]
        
        # 3. upper_jaw와 weight_up 상태에 따라 회전 여부 결정
        # upper_jaw가 True이고 weight_up이 True이면 회전 필요 (상악이고 weight center가 위에 있음)
        # upper_jaw가 False이고 weight_up이 False이면 회전 필요 (하악이고 weight center가 아래에 있음)
        need_rotation = (upper_jaw and weight_up) or (not upper_jaw and not weight_up)
        
        # 4. 회전 행렬 생성
        rotation_matrix = np.eye(3)
        if need_rotation:
            # z축만 반전 (y축은 유지)
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, -1]
            ])
        
        # 5. 메쉬 정렬
        aligned_vertices = np.dot(vertices, rotation_matrix.T)
        
        # 정렬된 메쉬 생성
        aligned_mesh = Mesh()
        aligned_mesh.vertices = aligned_vertices
        aligned_mesh.faces = mesh.faces
        aligned_mesh.normals = mesh.normals
        
        # 6. 변환 행렬 업데이트
        rotation_4x4 = np.eye(4)
        rotation_4x4[:3, :3] = rotation_matrix
        self.transform_matrix = np.dot(rotation_4x4, self.transform_matrix)
        
        return aligned_mesh, rotation_matrix
    
    def create_ball(self, radius, center):
        """
        주어진 반지름과 중심점으로 구를 생성합니다.
        
        Args:
            radius: 구의 반지름
            center: 구의 중심점 좌표 (numpy array)
            
        Returns:
            ball_mesh: 구를 나타내는 Mesh 객체
        """
        # 구를 표현하기 위한 정점과 면을 생성
        phi = np.linspace(0, np.pi, 20)
        theta = np.linspace(0, 2 * np.pi, 20)
        phi, theta = np.meshgrid(phi, theta)

        # 구면 좌표계에서 직교 좌표계로 변환
        x = radius * np.sin(phi) * np.cos(theta) + center[0]
        y = radius * np.sin(phi) * np.sin(theta) + center[1]
        z = radius * np.cos(phi) + center[2]

        # 정점 배열 생성
        vertices = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

        # 면 생성
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

        # Mesh 객체 생성
        ball_mesh = Mesh()
        ball_mesh.vertices = vertices
        ball_mesh.faces = np.array(faces)
        
        return ball_mesh
    
    def find_ray_mesh_intersection_approximate(self, mesh):
        """
        Shoots a pyramid-shaped light from the weight center in the +y direction,
        and finds the outermost points that intersect with the light to select a square-shaped region.
        
        Range adjustment methods:
        1. vertical_angle: Up/down angle range (default 10 degrees)
        2. horizontal_angle: Left/right angle range (default 60 degrees)
        3. min_y_component: Minimum y-direction component value (default 0.85, higher values select more front-facing faces)
        
        Args:
            mesh: Mesh object
            
        Returns:
            selected_mesh: Selected region's Mesh object
        """
        vertices = mesh.vertices
        faces = mesh.faces
        
        # 1. 무게중심 계산
        center = np.mean(vertices, axis=0)
        
        # 2. +y 방향 벡터 정의
        y_direction = np.array([0, 1, 0])
        
        # 3. 각 정점의 방향 벡터 계산 (무게중심 기준)
        directions = vertices - center
        distances = np.linalg.norm(directions, axis=1)
        
        # 4. 방향 벡터 정규화
        normalized_directions = directions / distances[:, np.newaxis]
        
        # 5. 각도 범위 설정 (여기서 범위 조절 가능)
        vertical_angle = 2  # 위아래 각도 범위
        horizontal_angle = 30  # 양옆 각도 범위
        min_y_component = 0.80  # y방향 최소 성분 값 (cos(약 30도))
        
        # 6. x축과 z축 방향 성분 계산
        x_components = np.abs(normalized_directions[:, 0])  # x축 성분
        z_components = np.abs(normalized_directions[:, 2])  # z축 성분
        y_components = normalized_directions[:, 1]  # y축 성분
        
        # 7. 조건에 맞는 점들 선택
        angle_mask = (
            (y_components > min_y_component) &  # 앞쪽을 바라보는 면만 선택
            (x_components < np.sin(np.radians(horizontal_angle))) &  # 양옆 범위
            (z_components < np.sin(np.radians(vertical_angle))) &  # 위아래 범위
            (normalized_directions[:, 1] > 0)  # +y 방향만 선택
        )
        
        # 8. 각도 기준으로 점들을 그룹화
        angle_groups = {}
        for idx in np.where(angle_mask)[0]:
            # 각도를 키로 사용 (적절한 정밀도로 반올림)
            angle_key = tuple(np.round(normalized_directions[idx], decimals=3))
            
            # 같은 각도 그룹에서 거리가 더 큰 경우에만 업데이트
            if angle_key not in angle_groups or distances[idx] > distances[angle_groups[angle_key]]:
                angle_groups[angle_key] = idx
        
        # 9. 각 각도별로 가장 먼 점들만 선택
        selected_vertices_idx = np.array(list(angle_groups.values()))
        
        # # 8. 선택된 정점 인덱스
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
        print(f"Number of faces in created mesh: {len(selected_faces)}")
        
        return selected_mesh
    
    def select_region_by_angle(self, mesh, angle_range_x=(-25, 25), angle_range_z=(-5, 5)):
        """
        Selects a region based on angle ranges.
        
        Args:
            mesh: Mesh object
            angle_range_x: X-axis angle range (default: -25 to 25 degrees)
            angle_range_z: Z-axis angle range (default: -5 to 5 degrees)
            
        Returns:
            selected_mesh: Selected region's Mesh object
        """
        print(f"[Log] Starting angle-based region selection: X {angle_range_x} degrees, Z {angle_range_z} degrees")
        
        vertices = mesh.vertices
        faces = mesh.faces
        
        # 1. 기본 계산
        weight_center = np.mean(vertices, axis=0)
        vectors = vertices - weight_center
        norms = np.linalg.norm(vectors, axis=1)
        norms[norms == 0] = 1.0
        normalized_vectors = vectors / norms.reshape(-1, 1)
        
        # 2. 양의 Y 방향 점 선택
        y_components = normalized_vectors[:, 1]
        positive_y_mask = y_components > 0.2  # y 방향 성분이 0.2보다 큰 점들만 선택
        
        # 3. 각도 기반 선택
        selected_vertices = np.zeros(len(vertices), dtype=bool)
        
        for idx in np.where(positive_y_mask)[0]:
            direction = normalized_vectors[idx]
            
            # XY 평면 투영
            xy_proj = np.array([direction[0], direction[1], 0])
            xy_norm = np.linalg.norm(xy_proj)
            
            # YZ 평면 투영
            yz_proj = np.array([0, direction[1], direction[2]])
            yz_norm = np.linalg.norm(yz_proj)
            
            if xy_norm > 1e-6 and yz_norm > 1e-6:
                # X 각도 계산 (XY 평면)
                xy_proj_norm = xy_proj / xy_norm
                angle_x = np.degrees(np.arctan2(xy_proj_norm[1], xy_proj_norm[0]) - np.pi/2)
                if angle_x < -90:
                    angle_x += 360
                
                # Z 각도 계산 (YZ 평면)
                yz_proj_norm = yz_proj / yz_norm
                angle_z = np.degrees(np.arctan2(yz_proj_norm[1], yz_proj_norm[2]))
                
                # 각도 범위 체크
                if (angle_range_x[0] <= angle_x <= angle_range_x[1] and
                    angle_range_z[0] <= angle_z <= angle_range_z[1]):
                    selected_vertices[idx] = True
        
        # 4. 선택된 정점들로 면 선택
        selected_faces = []
        for i, face in enumerate(faces):
            if np.any(selected_vertices[face]):
                selected_faces.append(i)
        
        # 5. 새로운 메쉬 생성
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
        Performs region growing starting from around the seed mesh's center point,
        based on normal vector similarity. Uses KDTree for efficient nearest point search.
        
        Args:
            mesh: Complete Mesh object
            seed_mesh: Mesh object of the region to use as starting point
            
        Returns:
            grown_mesh: Mesh object of the region selected by region growing
        """
        import time
        from scipy.spatial import KDTree
        
        print("\n=== Starting Region Growing ===")
        start_time = time.time()
        
        vertices = mesh.vertices
        faces = mesh.faces
        
        # 1. Calculate normal vectors
        print("1. Calculating normal vectors...")
        normals_start = time.time()
        
        def compute_vertex_normals(vertices, faces):
            # 면 법선 계산
            v1 = vertices[faces[:, 0]]
            v2 = vertices[faces[:, 1]]
            v3 = vertices[faces[:, 2]]
            
            vec1 = v2 - v1
            vec2 = v3 - v1
            
            face_normals = np.cross(vec1, vec2)
            face_norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
            face_norms[face_norms == 0] = 1.0
            face_normals = face_normals / face_norms
            
            # 정점 법선 계산 (면 법선의 평균)
            vertex_normals = np.zeros_like(vertices)
            vertex_counts = np.zeros(len(vertices))
            
            for i, face in enumerate(faces):
                for vertex in face:
                    vertex_normals[vertex] += face_normals[i]
                    vertex_counts[vertex] += 1
            
            # 정규화
            vertex_counts[vertex_counts == 0] = 1
            vertex_normals = vertex_normals / vertex_counts.reshape(-1, 1)
            norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vertex_normals = vertex_normals / norms
            
            return vertex_normals
        
        vertex_normals = compute_vertex_normals(vertices, faces)
        print(f"  - Normal vector calculation completed: {time.time() - normals_start:.2f} seconds")
        
        # 2. Create KDTree
        print("2. Creating KDTree...")
        tree_start = time.time()
        tree = KDTree(vertices)
        print(f"  - KDTree creation completed: {time.time() - tree_start:.2f} seconds")
        
        # 3. Find starting points
        print("3. Finding starting points...")
        seed_start = time.time()
        
        # seed_mesh의 정점들과 가장 가까운 원본 메쉬의 정점들 찾기
        seed_vertices = seed_mesh.vertices
        distances, indices = tree.query(seed_vertices, k=1)
        start_vertices = set(indices)
        
        print(f"  - Starting point search completed: {time.time() - seed_start:.2f} seconds")
        print(f"  - Number of starting points: {len(start_vertices)}")
        
        # 4. Region growing parameter setting
        max_angle_diff = 35.0  # Maximum allowable angle difference between normal vectors (degrees)
        max_distance = np.ptp(vertices, axis=0).max() * 0.05  # Maximum distance is 2% of mesh size
        
        # Calculate average normal vector of starting points
        avg_normal = np.mean(vertex_normals[list(start_vertices)], axis=0)
        avg_normal = avg_normal / np.linalg.norm(avg_normal)
        
        # 5. Region growing
        print("4. Running region growing...")
        growing_start = time.time()
        
        selected_vertices = np.zeros(len(vertices), dtype=bool)
        selected_vertices[list(start_vertices)] = True
        queue = list(start_vertices)
        in_queue = np.zeros(len(vertices), dtype=bool)
        in_queue[list(start_vertices)] = True
        
        iteration = 0
        last_log_time = time.time()
        
        while queue:
            current_time = time.time()
            if current_time - last_log_time > 1.0:  # Log every second
                print(f"  - Currently selected vertices: {np.sum(selected_vertices)}, Queue size: {len(queue)}")
                last_log_time = current_time
            
            iteration += 1
            if iteration % 10000 == 0:
                print(f"  - {iteration}th iteration...")
            
            current_vertex = queue.pop(0)
            
            # Find nearby vertices
            distances, neighbors = tree.query(vertices[current_vertex], k=20)
            
            for i, neighbor_idx in enumerate(neighbors):
                if neighbor_idx >= len(vertices) or selected_vertices[neighbor_idx] or in_queue[neighbor_idx]:
                    continue
                
                if distances[i] > max_distance:
                    continue
                
                # Normal vector similarity test
                neighbor_normal = vertex_normals[neighbor_idx]
                similarity = np.dot(avg_normal, neighbor_normal)
                angle_diff = np.degrees(np.arccos(np.clip(similarity, -1.0, 1.0)))
                
                if angle_diff <= max_angle_diff:
                    selected_vertices[neighbor_idx] = True
                    queue.append(neighbor_idx)
                    in_queue[neighbor_idx] = True
        
        print(f"  - Region growing completed: {time.time() - growing_start:.2f} seconds")
        print(f"  - Total iterations: {iteration}")
        
        # 6. Select faces from selected vertices
        print("5. Creating result mesh...")
        result_start = time.time()
        
        selected_faces = []
        for i, face in enumerate(faces):
            if np.any(selected_vertices[face]):
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
        
        print(f"  - Result mesh creation completed: {time.time() - result_start:.2f} seconds")
        
        print("=== Region Growing Completed ===")
        print(f"Number of selected vertices: {len(used_vertices)}")
        print(f"Number of selected faces: {len(selected_faces)}")
        print(f"Total time taken: {time.time() - start_time:.2f} seconds\n")
        
        return grown_mesh
    
    def move_mask_to_origin(self, mask_mesh):
        """
        Moves the mask mesh so that the most +y point is at y=0 and
        the most -z point is at z=0. Also updates the transformation matrix.
        
        Args:
            mask_mesh: Mesh object to move
            
        Returns:
            aligned_mesh: Moved Mesh object
            translation_matrix: Applied transformation matrix
        """
        vertices = mask_mesh.vertices
        faces = mask_mesh.faces
        
        # 1. Find the largest value in the y direction (most +y point)
        max_y = np.max(vertices[:, 1])
        
        # 2. Find the smallest value in the z direction (most -z point)
        min_z = np.min(vertices[:, 2])
        
        # 3. Calculate translation vector
        # y=0 needs to be moved by -max_y
        # z=0 needs to be moved by -min_z
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
        Performs ICP registration in 3 stages.
        Returns:
            transformed_source_mesh: Transformed source mesh
            transform_matrix: Applied transformation matrix
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
        
        # Mesh를 PointCloud로 변환
        print("\nConverting Mesh to PointCloud...")
        source = mesh_to_pointcloud(source_mesh)
        target = mesh_to_pointcloud(target_mesh)
        
        # ICP 실행
        print("\nStarting 1st ICP registration...")
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
            
            if np.allclose(result.transformation, current_transform, atol=1e-6):
                print(f"  - ICP converged (iteration {iteration})")
                break
                
            current_transform = result.transformation
        
        print("Starting 2nd ICP registration...")
        for iteration in range(1000):
            result = o3d.pipelines.registration.registration_icp(
                source, target,
                0.5,  # 거리 임계값
                current_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6,
                    relative_rmse=1e-6,
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
                0.1,  # 거리 임계값
                current_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6,
                    relative_rmse=1e-6,
                    max_iteration=1
                )
            )
            
            if np.allclose(result.transformation, current_transform, atol=1e-6):
                print(f"  - ICP converged (iteration {iteration})")
                break
                
            current_transform = result.transformation
        
        print("\n=== Registration completed ===")
        print(f"Final fitness: {result.fitness:.6f}")
        
        # 변환된 소스 메시 생성
        transformed_source_mesh = copy.deepcopy(source_mesh)
        transformed_source_mesh.vertices = np.dot(
            source_mesh.vertices,
            current_transform[:3, :3].T
        ) + current_transform[:3, 3]
        
        return transformed_source_mesh, current_transform



    def fast_registration_with_vis(self, source_mesh, target_mesh, vis=None):
        """
        Performs ICP registration in 3 stages with visualization.
        Returns:
            transformed_source_mesh: Transformed source mesh
            transform_matrix: Applied transformation matrix
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
        if vis is None:
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name='Registration', width=1280, height=720)
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
        print("\nConverting Mesh to PointCloud...")
        source = mesh_to_pointcloud(source_mesh)
        target = mesh_to_pointcloud(target_mesh)
        
        # 소스는 빨간색, 타겟은 파란색으로 설정
        source.paint_uniform_color([1, 0, 0])
        target.paint_uniform_color([0, 0, 1])
        
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
        print("\nStarting 1st ICP registration...")
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
                print(f"  - ICP iteration {iteration}: fitness = {result.fitness:.6f}")
                
                # 시각화 업데이트
                source_temp = copy.deepcopy(source)
                source_temp.transform(result.transformation)
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
                print(f"  - ICP converged (iteration {iteration})")
                break
                
            current_transform = result.transformation
        
        print("Starting 2nd ICP registration...")
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
                print(f"  - ICP iteration {iteration}: fitness = {result.fitness:.6f}")
                
                # 시각화 업데이트
                source_temp = copy.deepcopy(source)
                source_temp.transform(result.transformation)
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
                print(f"  - ICP converged (iteration {iteration})")
                break
                
            current_transform = result.transformation
        
        print("Starting 3rd ICP registration...")
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
                print(f"  - ICP iteration {iteration}: fitness = {result.fitness:.6f}")
                
                # 시각화 업데이트
                source_temp = copy.deepcopy(source)
                source_temp.transform(result.transformation)
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
                print(f"  - ICP converged (iteration {iteration})")
                break
                
            current_transform = result.transformation
        
        print("\n=== Registration completed ===")
        print(f"Final fitness: {result.fitness:.6f}")
        
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

if __name__ == "__main__":
    ios_laminate_registration = IOSLaminateRegistration("../../example/data/ios_with_smilearch.stl", "../../example/data/smile_arch_half.stl", visualization=True)
    ios_laminate_registration.run_registration()
