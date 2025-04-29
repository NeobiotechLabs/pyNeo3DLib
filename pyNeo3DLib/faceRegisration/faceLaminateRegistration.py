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
from scipy.ndimage import binary_fill_holes

class FaceMeshAnalyzer:
    def __init__(self):
        """Initialize MediaPipe Face Mesh"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        
        # Lip landmark indices
        self.outer_lips = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
                          291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
        self.inner_lips = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
                          308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
  


class FaceLaminateRegistration:
    def __init__(self, face_path, laminate_path, visualization=False):
        self.face_smile_path = face_path
        self.laminate_path = laminate_path
        self.visualization = visualization
        # Initialize transformation matrix (set as 4x4 matrix)
        self.transform_matrix = np.eye(4)

        self.__load_models()

    def __load_models(self):
        self.face_smile_mesh = Mesh.from_file(self.face_smile_path)
        self.laminate_mesh = Mesh.from_file(self.laminate_path)

        return self.face_smile_mesh, self.laminate_mesh
    
    def apply_transformation(self, transformation_matrix):
        """
        Apply transformation to face_smile_mesh and accumulate in transform_matrix.
        
        Args:
            transformation_matrix (np.ndarray): 4x4 transformation matrix
        """
        # Accumulate transformation matrix
        self.transform_matrix = np.dot(transformation_matrix, self.transform_matrix)
        
        # Convert to homogeneous coordinates (4xN matrix)
        vertices = self.face_smile_mesh.vertices
        vertices_homogeneous = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
        # Convert from homogeneous coordinates to 3D coordinates
        transformed_vertices = np.dot(vertices_homogeneous, self.transform_matrix.T)
        self.face_smile_mesh.vertices = transformed_vertices[:, :3]
        
        # Apply only rotation to normal vectors (no translation)
        if self.face_smile_mesh.normals is not None:
            normals = self.face_smile_mesh.normals
            rotation_matrix = self.transform_matrix[:3, :3]
            self.face_smile_mesh.normals = np.dot(normals, rotation_matrix.T)
    
    def align_y_axis(self):
        """
        Apply 180-degree rotation transformation around Z-axis.
        """
        # Create transformation matrix for 180-degree rotation around Z-axis
        angle = np.pi  # 180 degrees
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0, 0],
            [np.sin(angle), np.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Apply transformation
        self.apply_transformation(rotation_matrix)

    def find_lip_via_analyze_face_landmarks(self):
        """
        Extract lip landmarks from face image and convert to UV coordinates.
        """
        analyzer = FaceMeshAnalyzer()
        
        # Extract image file path (assuming image file has same name as obj file)
        image_path = self.face_smile_path.replace('.obj', '.png')
        if not os.path.exists(image_path):
            image_path = self.face_smile_path.replace('.obj', '.jpg')
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: image version of {self.face_smile_path}")
        
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
        
        # Extract lip landmarks
        inner_points = []
        outer_points = []
        
        # Extract inner lip landmarks
        for idx in analyzer.inner_lips:
            landmark = face_landmarks.landmark[idx]
            # Convert image coordinates to pixel coordinates (0~1 range to pixel coordinates)
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            inner_points.append([x, y])
        
        # Extract outer lip landmarks
        for idx in analyzer.outer_lips:
            landmark = face_landmarks.landmark[idx]
            # Convert image coordinates to pixel coordinates (0~1 range to pixel coordinates)
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            outer_points.append([x, y])
        
        # Convert to UV coordinates (normalize and flip coordinates)
        inner_uv_points = self._normalize_and_flip_coordinates(inner_points, (w, h))
        outer_uv_points = self._normalize_and_flip_coordinates(outer_points, (w, h))
        
        # Save results
        self.inner_lip_points = inner_points
        self.outer_lip_points = outer_points
        self.inner_lip_uv = inner_uv_points
        self.outer_lip_uv = outer_uv_points
        
        return inner_uv_points, outer_uv_points
    
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
    
    def is_point_in_polygon(self, point, polygon, epsilon=1e-10):
        """Check if a point is inside a polygon (Ray Casting Algorithm)"""
        x, y = point
        n = len(polygon)
        inside = False
        
        # Handle points on boundary line
        def on_segment(p, q, r):
            if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
                d = (r[1] - p[1]) * (q[0] - p[0]) - (q[1] - p[1]) * (r[0] - p[0])
                if abs(d) < epsilon:
                    return True
            return False
        
        # Check points on boundary line
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


    def find_lip_regions(self, mesh, inner_uv_points, margin=0.005):
        """Detect inner lip regions with vectorized optimization"""
        start_time = time.time()
        print(f"[Time Measurement] Lip region detection started")
        
        # Prepare mesh data
        prep_start = time.time()
        faces = np.asarray(mesh.faces)
        uvs = np.asarray(mesh.uvs)
        face_uvs = np.asarray(mesh.face_uvs)
        print(f"[Time Measurement] Mesh data preparation: {time.time() - prep_start:.4f} seconds")
        
        # Convert to numpy array and ensure closed boundary
        inner_uv_points = np.array(inner_uv_points)
        if not np.allclose(inner_uv_points[0], inner_uv_points[-1]):
            inner_uv_points = np.vstack([inner_uv_points, inner_uv_points[0]])
        
        # Calculate bounding box with margin
        min_uv = np.min(inner_uv_points, axis=0) - margin
        max_uv = np.max(inner_uv_points, axis=0) + margin
        
        # Get all triangle UV coordinates and centers at once
        all_triangle_uvs = uvs[face_uvs]
        triangle_centers = np.mean(all_triangle_uvs, axis=1)
        
        # Quick bounding box filtering - vectorized
        bbox_mask = np.all((triangle_centers >= min_uv) & (triangle_centers <= max_uv), axis=1)
        candidate_indices = np.where(bbox_mask)[0]
        
        # Process candidate triangles
        triangle_start = time.time()
        print(f"[Time Measurement] Processing {len(candidate_indices)} candidate triangles")
        
        # Vectorized triangle processing
        candidate_centers = triangle_centers[candidate_indices]
        candidate_uvs = all_triangle_uvs[candidate_indices]
        
        # Check all vertices are within bounding box
        vertices_in_bbox = np.all(
            (candidate_uvs >= min_uv) & (candidate_uvs <= max_uv),
            axis=(1, 2)
        )
        
        inner_triangles = set()
        
        # Process triangles in larger batches
        batch_size = 5000  # 더 큰 배치 사이즈 사용
        for i in range(0, len(candidate_indices), batch_size):
            batch_end = min(i + batch_size, len(candidate_indices))
            batch_indices = candidate_indices[i:batch_end]
            
            # Get batch data
            centers_batch = candidate_centers[i:batch_end]
            uvs_batch = candidate_uvs[i:batch_end]
            bbox_batch = vertices_in_bbox[i:batch_end]
            
            # Check centers first
            for j, (idx, center, uvs, in_bbox) in enumerate(zip(batch_indices, centers_batch, uvs_batch, bbox_batch)):
                if not in_bbox:
                    continue
                    
                if self.is_point_in_polygon(center, inner_uv_points):
                    inner_triangles.add(idx)
                    continue
                
                # Only check vertices if center is outside
                for vertex_uv in uvs:
                    if self.is_point_in_polygon(vertex_uv, inner_uv_points):
                        inner_triangles.add(idx)
                    break
        
        print(f"[Time Measurement] Triangle processing completed: {time.time() - triangle_start:.4f} seconds")
        print(f"[Time Measurement] Number of selected triangles: {len(inner_triangles)}")
        
        # Collect vertices efficiently using numpy operations
        vertex_start = time.time()
        triangle_array = np.array(list(inner_triangles))
        selected_faces = faces[triangle_array]
        inner_vertices = np.unique(selected_faces.flatten())
        
        print(f"[Time Measurement] Vertex collection completed: {time.time() - vertex_start:.4f} seconds")
        print(f"[Time Measurement] Number of selected vertices: {len(inner_vertices)}")
        
        total_time = time.time() - start_time
        print(f"[Time Measurement] Total time for lip region detection: {total_time:.4f} seconds")
        
        return inner_vertices.tolist()

    def find_lip_via_convex_hull(self, inner_uv_points):
        """
        Select mesh inside lips using inner lip UV coordinates.
        
        Args:
            inner_uv_points: List of inner lip UV coordinates [[u1, v1], [u2, v2], ...]
            
        Returns:
            Partial mesh composed of selected vertices
        """
        start_time = time.time()
        print(f"[Time Measurement] Lip mesh generation started")
        
        # Convert UV coordinates to numpy array
        uv_start = time.time()
        inner_uv_points = np.array(inner_uv_points)
        print(f"[Time Measurement] UV coordinate conversion: {time.time() - uv_start:.4f} seconds")
        
        # Find lip region vertices
        region_start = time.time()
        selected_vertices = self.find_lip_regions(self.face_smile_mesh, inner_uv_points)
        print(f"[Time Measurement] Finding lip region vertices: {time.time() - region_start:.4f} seconds")
        
        print(f"Number of selected vertices: {len(selected_vertices)}")
        
        if len(selected_vertices) == 0:
            print("Warning: No vertices selected!")
            return None
        
        # Save selected vertices
        self.lip_vertices = selected_vertices
        
        # Create partial mesh from selected vertices
        mesh_start = time.time()
        lip_mesh = self.face_smile_mesh.extract_mesh_from_vertices(selected_vertices)
        print(f"[Time Measurement] Partial mesh generation: {time.time() - mesh_start:.4f} seconds")
        
        total_time = time.time() - start_time
        print(f"[Time Measurement] Total time for lip mesh generation: {total_time:.4f} seconds")
        
        return lip_mesh
    
    def align_lip_to_laminate(self, lip_mesh):
        """Move lip mesh to laminate mesh position with improved alignment"""
        # Calculate center points of each mesh
        lip_center = np.mean(lip_mesh.vertices, axis=0)
        laminate_center = np.mean(self.laminate_mesh.vertices, axis=0)
        
        # Calculate translation vector
        translation = laminate_center - lip_center
        
        # Create translation transformation matrix
        transform_matrix = np.eye(4)
        transform_matrix[:3, 3] = translation
        
        # Apply transformation to lip mesh
        vertices = lip_mesh.vertices
        vertices_homogeneous = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
        transformed_vertices = np.dot(vertices_homogeneous, transform_matrix.T)
        lip_mesh.vertices = transformed_vertices[:, :3]
        
        # Apply only rotation to normal vectors
        if lip_mesh.normals is not None:
            normals = lip_mesh.normals
            rotation_matrix = transform_matrix[:3, :3]
            lip_mesh.normals = np.dot(normals, rotation_matrix.T)
        
        # Accumulate transformation matrix
        self.transform_matrix = np.dot(transform_matrix, self.transform_matrix)
        
        return lip_mesh


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
        
        # Convert Mesh to Open3D PointCloud
        def mesh_to_pointcloud(mesh):
            # 1. Create point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(mesh.vertices)
            
            # 2. Process normal vectors
            if mesh.normals is not None:
                pcd.normals = o3d.utility.Vector3dVector(mesh.normals)
            else:
                temp_mesh = o3d.geometry.TriangleMesh()
                temp_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
                temp_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
                temp_mesh.compute_vertex_normals()
                pcd.normals = temp_mesh.vertex_normals
            
            # 3. Estimate normal direction and check consistency
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            pcd.orient_normals_consistent_tangent_plane(k=100)
            
            pcd.uniform_down_sample(every_k_points=2)
            
            return pcd
        
        # Create visualization window
        if vis is None and self.visualization:
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name='Registration', width=1920, height=1080)
            opt = vis.get_render_option()
            opt.background_color = np.asarray([0.9, 0.9, 0.9])
            opt.point_size = 2.0
            
            # Camera settings (view from +y direction to -y direction)
            ctr = vis.get_view_control()
            ctr.set_zoom(0.8)
            ctr.set_front([0, -1, 0])  # -y direction to look at
            ctr.set_up([0, 0, 1])      # z-axis up
            
            # Force camera settings
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.1)
        
        # Convert Mesh to PointCloud
        print("\nConverting Mesh to PointCloud...")
        source = mesh_to_pointcloud(source_mesh)
        target = mesh_to_pointcloud(target_mesh)
        
        # Set source to red, target to blue
        source.paint_uniform_color([1, 0, 0])
        target.paint_uniform_color([0, 0, 1])
        
        if self.visualization:
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
        
        # ICP execution
        print("\nStarting 1st ICP registration...")
        current_transform = np.eye(4)
        
        for iteration in range(1000):
            result = o3d.pipelines.registration.registration_icp(
                source, target,
                1.0,  # Distance threshold
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
                if self.visualization:
                    vis.clear_geometries()
                    vis.add_geometry(source_temp)
                    vis.add_geometry(target)
                
                # Reset camera view each iteration
                    ctr = vis.get_view_control()
                    ctr.set_zoom(0.8)
                    ctr.set_front([0, -1, 0])
                    ctr.set_up([0, 0, 1])
                    
                    vis.poll_events()
                    vis.update_renderer()
                    time.sleep(0.05)  # Adjust animation speed
            
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
                if self.visualization:
                    vis.clear_geometries()
                    vis.add_geometry(source_temp)
                    vis.add_geometry(target)
                
                    # Reset camera view each iteration
                    ctr = vis.get_view_control()
                    ctr.set_zoom(0.8)
                    ctr.set_front([0, -1, 0])
                    ctr.set_up([0, 0, 1])
                    
                    vis.poll_events()
                    vis.update_renderer()
                    time.sleep(0.05)  # Adjust animation speed
            
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
                if self.visualization:
                    vis.clear_geometries()
                    vis.add_geometry(source_temp)
                    vis.add_geometry(target)
                    
                    # Reset camera view each iteration
                    ctr = vis.get_view_control()
                    ctr.set_zoom(0.8)
                    ctr.set_front([0, -1, 0])
                    ctr.set_up([0, 0, 1])
                    
                    vis.poll_events()
                    vis.update_renderer()
                    time.sleep(0.05)  # Adjust animation speed
            
            if np.allclose(result.transformation, current_transform, atol=1e-6):
                print(f"  - ICP converged (iteration {iteration})")
                break
                
            current_transform = result.transformation
        
        print("\n=== Registration completed ===")
        print(f"Final fitness: {result.fitness:.6f}")
        
        if self.visualization:
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
    

    def run_registration(self):
        # Initial mesh visualization
        if self.visualization:
            visualize_meshes([self.face_smile_mesh, self.laminate_mesh], ["Face", "Laminate"], title="Initial Meshes")
        
        # Y-axis alignment
        self.align_y_axis()
        if self.visualization:
            visualize_meshes([self.face_smile_mesh, self.laminate_mesh], ["Face", "Laminate"], title="After Y-axis Alignment")
        print("Y-axis alignment transformation matrix:")
        print(self.transform_matrix)
        
        # Extract lip landmarks
        inner_uv_points, outer_uv_points = self.find_lip_via_analyze_face_landmarks()
        print("Lip UV coordinates:")
        print("Inner:", inner_uv_points)
        print("Outer:", outer_uv_points)
        
        # Create inner lip partial mesh
        lip_mesh = self.find_lip_via_convex_hull(inner_uv_points)
        if lip_mesh is None:
            print("Lip mesh generation failed")
            return None, None
        
        # Move lip mesh to laminate position
        lip_mesh = self.align_lip_to_laminate(lip_mesh)
        
        # Apply accumulated transformation to entire mesh
        moved_smile_mesh = copy.deepcopy(self.face_smile_mesh)
        # Apply only translation transformation
        translation = self.transform_matrix[:3, 3]
        moved_smile_mesh.vertices = moved_smile_mesh.vertices + translation
        
        # Final result visualization
        if self.visualization:
            visualize_meshes([lip_mesh, moved_smile_mesh, self.laminate_mesh], 
                            ["Lip", "Moved Face", "Laminate"], 
                            title="Final Result")
        print("Final accumulated transformation matrix:")
        print(self.transform_matrix)

        # Now match meshes using ICP
        transformed_mesh, fast_registration_transform_matrix = self.fast_registration_with_vis(lip_mesh, self.laminate_mesh)

        # Apply final transformation at once
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
        Visualize lip landmarks on the image.
        """
        # Extract image file path (assuming image file has same name as obj file)
        image_path = self.face_smile_path.replace('.obj', '.png')
        if not os.path.exists(image_path):
            image_path = self.face_smile_path.replace('.obj', '.jpg')
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: image version of {self.face_smile_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # Copy image (preserve original)
        vis_image = image.copy()
        
        # Check if lip landmarks are already extracted
        if not hasattr(self, 'inner_lip_points') or not hasattr(self, 'outer_lip_points'):
            # Extract landmarks if not already extracted
            inner_uv_points, outer_uv_points = self.find_lip_via_analyze_face_landmarks()
        
        # Visualize inner lip landmarks (blue)
        for point in self.inner_lip_points:
            x, y = point
            cv2.circle(vis_image, (x, y), 3, (255, 0, 0), -1)  # Blue
        
        # Visualize outer lip landmarks (red)
        for point in self.outer_lip_points:
            x, y = point
            cv2.circle(vis_image, (x, y), 3, (0, 0, 255), -1)  # Red
        
        # Connect inner lip landmarks (blue)
        for i in range(len(self.inner_lip_points)):
            pt1 = tuple(self.inner_lip_points[i])
            pt2 = tuple(self.inner_lip_points[(i + 1) % len(self.inner_lip_points)])
            cv2.line(vis_image, pt1, pt2, (255, 0, 0), 1)
        
        # Connect outer lip landmarks (red)
        for i in range(len(self.outer_lip_points)):
            pt1 = tuple(self.outer_lip_points[i])
            pt2 = tuple(self.outer_lip_points[(i + 1) % len(self.outer_lip_points)])
            cv2.line(vis_image, pt1, pt2, (0, 0, 255), 1)
        
        # Save image
        output_dir = os.path.dirname(image_path)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        vis_path = os.path.join(output_dir, f"{base_name}_landmarks.png")
        cv2.imwrite(vis_path, vis_image)
        
        print(f"Landmark visualization saved: {vis_path}")
        
        return vis_path


if __name__ == "__main__":
    face_laminate_registration = FaceLaminateRegistration("../../example/data/FaceScan/Smile/Smile.obj", "../../example/data/smile_arch_half.stl", visualization=True)
    final_transform = face_laminate_registration.run_registration()
    print(final_transform)