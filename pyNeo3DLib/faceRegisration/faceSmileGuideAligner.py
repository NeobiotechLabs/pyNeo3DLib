
import copy
import numpy as np
import open3d as o3d
from pathlib import Path
import time
import os
import sys

# Add parent directory to path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from .faceAlignModule.face_lip_extractor import FaceLipExtractor
from .faceAlignModule.lip_preprocessing import align_face_axis, filter_by_normal_direction

class FaceSmileGuideAligner:
    """
    Aligns a face scan 3D model to a smile arch using lip extraction and ICP.
    """

    def __init__(self):
        pass

    def align(self, face_scan_path: str, smile_arch_path: str, visualize: bool = False) -> np.ndarray:
        """
        Aligns the face scan to the smile arch.

        Args:
            face_scan_path: Path to the face scan file (PLY or OBJ).
            smile_arch_path: Path to the smile arch half file (STL or other).
            visualize: Whether to visualize the steps.

        Returns:
            np.ndarray: 4x4 Transformation Matrix that aligns the face scan to the smile arch.
        """
        face_scan_path = Path(face_scan_path)
        folder_path = face_scan_path.parent
        
        print(f"Aligning Face Scan: {face_scan_path}")
        print(f"Target Smile Arch: {smile_arch_path}")

        # 1. Load Full Mesh
        print("\n1. Loading full mesh...")
        full_mesh = o3d.io.read_triangle_mesh(str(face_scan_path))
        full_mesh.compute_vertex_normals()
        print(f"  Open3D mesh: {len(full_mesh.vertices)} vertices")

        # 2. Align Face Axis (Feature-based)
        print("\n2. Aligning face axis...")
        # We need FaceLipExtractor context for alignment and extraction
        with FaceLipExtractor() as extractor:
            # align_face_axis expects a folder path to find textures, but uses the mesh we pass
            # It also re-parses files in the folder. We assume the folder contains the necessary files.
            aligned_mesh, align_transform = align_face_axis(full_mesh, extractor, str(folder_path))

            # 3. Extract Lip Region
            print("\n3. Extracting lip region...")
            # This returns indices based on the parser's vertex order
            lip_result = extractor.extract_lip_region(str(folder_path))
            
            if lip_result is None:
                print("Error: Failed to extract lip region.")
                return np.eye(4)

            # Manually parse vertices to match the indices from extract_lip_region
            if face_scan_path.suffix.lower() == '.ply':
                mesh_data = extractor.parse_ply(str(face_scan_path))
                parser_vertices = mesh_data['vertices']
                if 'normals' in mesh_data and not np.allclose(mesh_data['normals'], 0):
                    parser_normals = mesh_data['normals']
                else:
                    # Compute normal using open3d if missing in parser
                    temp_mesh = o3d.geometry.TriangleMesh()
                    temp_mesh.vertices = o3d.utility.Vector3dVector(parser_vertices)
                    if 'faces' in mesh_data:
                         # Handle faces if present (check if tuple or array)
                        faces = mesh_data['faces']
                        if len(faces) > 0 and isinstance(faces[0], tuple):
                            face_v_indices = [f[0] for f in faces]
                            temp_mesh.triangles = o3d.utility.Vector3iVector(face_v_indices)
                        else:
                            temp_mesh.triangles = o3d.utility.Vector3iVector(faces)
                        temp_mesh.compute_vertex_normals()
                        parser_normals = np.asarray(temp_mesh.vertex_normals)
                    else:
                        parser_normals = np.zeros_like(parser_vertices)

            elif face_scan_path.suffix.lower() == '.obj':
                mesh_data = extractor.parse_obj(str(face_scan_path))
                parser_vertices = mesh_data['vertices']
                # OBJ parser doesn't return normals usually, compute them
                temp_mesh = o3d.geometry.TriangleMesh()
                temp_mesh.vertices = o3d.utility.Vector3dVector(parser_vertices)
                # Reconstruct faces for normal computation
                faces = mesh_data['faces']
                if faces:
                    face_v_indices = [f[0] for f in faces]
                    temp_mesh.triangles = o3d.utility.Vector3iVector(face_v_indices)
                    temp_mesh.compute_vertex_normals()
                    parser_normals = np.asarray(temp_mesh.vertex_normals)
                else:
                    parser_normals = np.zeros_like(parser_vertices)
            else:
                print(f"Unsupported file format: {face_scan_path.suffix}")
                return np.eye(4)

            all_lip_indices = lip_result['all_vertex_indices']
            
            # Apply align_transform to parser vertices
            parser_vertices_homo = np.hstack([parser_vertices, np.ones((len(parser_vertices), 1))])
            transformed_vertices = (align_transform @ parser_vertices_homo.T).T[:, :3]
            
            # Rotate normals
            rotation = align_transform[:3, :3]
            transformed_normals = (rotation @ parser_normals.T).T

            # 4. Filter by Normal Direction
            print("\n4. Filtering vertices (Normal +Y, within 45deg)...")
            strict_angle = 45
            threshold = np.cos(np.radians(strict_angle))
            
            filtered_indices = filter_by_normal_direction(
                vertices=transformed_vertices,
                normals=transformed_normals,
                vertex_indices=all_lip_indices,
                front_direction=np.array([0.0, 1.0, 0.0]),
                threshold=threshold
            )
            
            filtered_vertices = transformed_vertices[filtered_indices]

            # 5. Clustering (DBSCAN)
            print("\n5. Clustering filtered points (Weighted DBSCAN)...")
            cluster_mask = self._cluster_vertices(filtered_vertices)
            final_teeth_vertices = filtered_vertices[cluster_mask]
            
            if len(final_teeth_vertices) == 0:
                print("Error: No vertices remaining after clustering.")
                return align_transform # Return best guess so far

        # 6. Prepare for ICP
        # Source: Extracted Upper Teeth (currently in 'Aligned' space)
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(final_teeth_vertices)
        source.paint_uniform_color([1.0, 1.0, 0.0]) # Yellow

        # Target: Smile Arch
        target_mesh = o3d.io.read_triangle_mesh(smile_arch_path)
        target_mesh.compute_vertex_normals()
        target_mesh.paint_uniform_color([0.7, 0.7, 0.7]) # Grey
        target_pcd = target_mesh.sample_points_poisson_disk(number_of_points=5000)

        # 7. Initial Alignment (Translation)
        # Calculate Translation T
        src_points = np.asarray(source.points)
        src_center = np.mean(src_points, axis=0) # PointCloud center
        src_max = np.max(src_points, axis=0)
        src_min = np.min(src_points, axis=0)

        tgt_points = np.asarray(target_mesh.vertices)
        tgt_max = np.max(tgt_points, axis=0)
        tgt_min = np.min(tgt_points, axis=0)

        translation = np.zeros(3)
        translation[0] = -src_center[0] # Center X -> 0
        translation[1] = tgt_max[1] - src_max[1] # Max Y align
        translation[2] = tgt_min[2] - src_min[2] # Min Z align

        T_init = np.eye(4)
        T_init[:3, 3] = translation
        
        print(f"\nInitial Translation: {translation}")
        
        # Apply T_init to source
        source.transform(T_init)
        
        # Cumulative transform: T_init @ align_transform
        # Note: 'align_transform' brings original -> aligned
        # 'T_init' brings aligned -> init_pose
        # So current state represents (T_init @ align_transform) * Original

        # 8. ICP
        print("\nStarting ICP (Robust Point-to-Plane with Tukey Loss)...")
        
        # Prepare Visualizer if requested
        vis = None
        if visualize:
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="Face Scan Alignment (ICP)", width=1200, height=800)
            vis.add_geometry(source)
            vis.add_geometry(target_mesh)
            
            ctr = vis.get_view_control()
            ctr.set_front([0, 1, 0])
            ctr.set_up([0, 0, 1])
            ctr.set_lookat([0, 0, 0])
            ctr.set_zoom(0.6)

        # Estimate normals for source for Point-to-Plane
        source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30))

        # Iterative ICP with changing threshold
        current_transform = np.eye(4) # Transform accumulated during ICP
        threshold = 2.0
        
        loss = o3d.pipelines.registration.TukeyLoss(k=0.5)
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)

        steps = 60
        for i in range(steps):
            if i == 20: threshold = 1.5
            elif i == 40: threshold = 0.5
            
            reg_p2plane = o3d.pipelines.registration.registration_icp(
                source, target_pcd, threshold, np.identity(4),
                estimation_method,
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1)
            )
            
            source.transform(reg_p2plane.transformation)
            current_transform = reg_p2plane.transformation @ current_transform
            
            # Re-estimate normals
            source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30))

            if visualize:
                vis.update_geometry(source)
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.01)

        if visualize:
            print("Alignment Finished. Showing result...")
            vis.run()
            vis.destroy_window()

        # Calculate Final Transform
        # Sequence: Original -> [Align] -> [Init Translation] -> [ICP]
        # Matrix: ICP @ Init @ Align
        final_transform = current_transform @ T_init @ align_transform
        
        # Optional: Final Full Visualization
        if visualize:
            print("\nVisualizing Final Full Face...")
             # Apply final transform to the original full mesh (need to reload or use copy logic, but here we can just transform the Open3D aligned mesh)
            # aligned_mesh is (Align * Original)
            # We need to apply (ICP @ Init) to aligned_mesh
            final_vis_mesh = copy.deepcopy(aligned_mesh)
            final_vis_mesh.transform(current_transform @ T_init)
            
            final_vis_mesh.paint_uniform_color([1.0, 0.8, 0.6])
            
            o3d.visualization.draw_geometries(
                [final_vis_mesh, target_mesh],
                window_name="Final Result",
                front=[0, 1, 0], lookat=[0, 0, 0], up=[0, 0, 1], zoom=0.6
            )

        return final_transform

    def _cluster_vertices(self, vertices: np.ndarray) -> np.ndarray:
        """
        Applies Weighted DBSCAN clustering to filter out noise and select the upper teeth cluster.
        Returns a boolean mask for the vertices.
        """
        if len(vertices) == 0:
            return np.zeros(0, dtype=bool)

        threshold_x = 1.5
        threshold_y = 1.0
        threshold_z = 0.5

        scaled_vertices = np.copy(vertices)
        scaled_vertices[:, 0] /= threshold_x
        scaled_vertices[:, 1] /= threshold_y
        scaled_vertices[:, 2] /= threshold_z

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(scaled_vertices)
        
        labels = np.array(pcd.cluster_dbscan(eps=1.0, min_points=10, print_progress=False))
        unique_labels = np.unique(labels)
        
        candidates = []
        for label in unique_labels:
            if label == -1: continue
            
            mask = (labels == label)
            points = vertices[mask]
            count = len(points)
            mean_z = np.mean(points[:, 2])

            if count > 1000:
                candidates.append({
                    'label': label,
                    'count': count,
                    'mean_z': mean_z,
                    'mask': mask
                })
        
        final_mask = np.zeros(len(vertices), dtype=bool)
        if candidates:
            # Select cluster with highest Z (upper teeth)
            best_cluster = max(candidates, key=lambda x: x['mean_z'])
            final_mask = best_cluster['mask']
        
        return final_mask

if __name__ == "__main__":
    # Test script similar to the original entry point but using the class
    import argparse
    parser = argparse.ArgumentParser(description="Align Face Scan using FaceScanAligner Class")
    parser.add_argument('--sample', '-s', type=int, default=1, help="Sample index")
    parser.add_argument('--visualize', '-v', action='store_true', help="Visualize alignment")
    args = parser.parse_args()

    sample_id = args.sample
    # Construct paths assuming standard structure for testing
    face_scan_dir = Path(f'./3dmodel/facescan/sample_{sample_id}')
    ply_files = list(face_scan_dir.glob('*.ply'))
    obj_files = list(face_scan_dir.glob('*.obj'))
    
    if ply_files:
        face_path = str(ply_files[0])
    elif obj_files:
        face_path = str(obj_files[0])
    else:
        print(f"No mesh found in {face_scan_dir}")
        exit(1)
        
    smile_path = './3dmodel/smile_arch_half.stl'
    
    aligner = FaceSmileGuideAligner()
    transform = aligner.align(face_path, smile_path, visualize=args.visualize)
    
    print("\nFinal Transformation Matrix:")
    print(transform)
