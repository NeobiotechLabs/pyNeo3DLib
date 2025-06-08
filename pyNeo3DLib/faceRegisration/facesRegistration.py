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

class FacesRegistration:
    def __init__(self, face_smile_mesh, transform_matrix_for_smile, face_rest_path, face_retraction_path, visualization=False):
        self.face_smile_mesh = face_smile_mesh
        self.face_rest_path = face_rest_path
        self.face_retraction_path = face_retraction_path
        self.visualization = visualization

        self.transform_matrix_for_rest = transform_matrix_for_smile
        self.transform_matrix_for_retraction =transform_matrix_for_smile

        self.__load_models()
        
    def __load_models(self):
        self.face_rest_mesh = Mesh.from_file(self.face_rest_path)
        self.face_retraction_mesh = Mesh.from_file(self.face_retraction_path)

        self.face_rest_mesh.vertices = np.dot(self.face_rest_mesh.vertices, self.transform_matrix_for_rest[:3, :3].T) + self.transform_matrix_for_rest[:3, 3]
        self.face_retraction_mesh.vertices = np.dot(self.face_retraction_mesh.vertices, self.transform_matrix_for_retraction[:3, :3].T) + self.transform_matrix_for_retraction[:3, 3]

        return self.face_smile_mesh, self.face_rest_mesh, self.face_retraction_mesh
            

    def match_weight_centers(self):
        """
        Align the centroids of three meshes (face_smile_mesh, face_rest_mesh, face_retraction_mesh)
        based on the centroid of face_smile_mesh.
        
        Returns:
            Transformed face_rest_mesh and face_retraction_mesh
        """
        # Calculate centroid of face_smile_mesh (Open3D 메시인 경우)
        if hasattr(self.face_smile_mesh, 'vertices') and hasattr(self.face_smile_mesh.vertices, '__len__'):
            # Open3D TriangleMesh인 경우
            smile_vertices = np.asarray(self.face_smile_mesh.vertices)
        else:
            # 커스텀 Mesh인 경우
            smile_vertices = self.face_smile_mesh.vertices
        smile_center = np.mean(smile_vertices, axis=0)
        
        # Calculate centroid of face_rest_mesh and transform
        rest_center = np.mean(self.face_rest_mesh.vertices, axis=0)
        rest_translation = smile_center - rest_center
        self.face_rest_mesh.vertices = self.face_rest_mesh.vertices + rest_translation
        
        # Update transformation matrix for face_rest_mesh
        rest_transform = np.eye(4)
        rest_transform[:3, 3] = rest_translation
        self.transform_matrix_for_rest = np.dot(rest_transform, self.transform_matrix_for_rest)
        
        # Calculate centroid of face_retraction_mesh and transform
        retraction_center = np.mean(self.face_retraction_mesh.vertices, axis=0)
        retraction_translation = smile_center - retraction_center
        self.face_retraction_mesh.vertices = self.face_retraction_mesh.vertices + retraction_translation
        
        # Update transformation matrix for face_retraction_mesh
        retraction_transform = np.eye(4)
        retraction_transform[:3, 3] = retraction_translation
        self.transform_matrix_for_retraction = np.dot(retraction_transform, self.transform_matrix_for_retraction)
        
        print(f"Centroid alignment completed:")
        print(f"  - Smile mesh centroid: {smile_center}")
        print(f"  - Rest mesh centroid: {rest_center} -> {np.mean(self.face_rest_mesh.vertices, axis=0)}")
        print(f"  - Retraction mesh centroid: {retraction_center} -> {np.mean(self.face_retraction_mesh.vertices, axis=0)}")
        print(f"  - Rest mesh transformation matrix updated")
        print(f"  - Retraction mesh transformation matrix updated")
        
        return self.face_rest_mesh, self.face_retraction_mesh

    
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
        
        # Mesh를 Open3D PointCloud로 변환
        def mesh_to_pointcloud(mesh):
            # 1. 포인트 클라우드 생성
            pcd = o3d.geometry.PointCloud()
            
            # 메시 타입에 따라 vertices 처리
            if hasattr(mesh, 'vertices') and hasattr(mesh.vertices, '__len__'):
                # Open3D TriangleMesh인 경우
                vertices = np.asarray(mesh.vertices)
                faces = np.asarray(mesh.triangles) if hasattr(mesh, 'triangles') else mesh.faces
                normals = np.asarray(mesh.vertex_normals) if hasattr(mesh, 'vertex_normals') else mesh.normals
            else:
                # 커스텀 Mesh인 경우
                vertices = mesh.vertices
                faces = mesh.faces
                normals = mesh.normals
            
            pcd.points = o3d.utility.Vector3dVector(vertices)
            
            down_sample_rate = 10
            # 먼저 다운샘플링 수행
            pcd = pcd.uniform_down_sample(every_k_points=down_sample_rate)
            
            # 2. 법선 벡터 처리 (다운샘플된 점들에 대해서만)
            if normals is not None:
                normals_sampled = np.asarray(normals)[::down_sample_rate]  # 다운샘플링과 동일한 비율로 법선 벡터 선택
                pcd.normals = o3d.utility.Vector3dVector(normals_sampled)
            else:
                # 다운샘플된 점들에 대해서만 법선 계산
                temp_mesh = o3d.geometry.TriangleMesh()
                temp_mesh.vertices = pcd.points
                temp_mesh.triangles = o3d.utility.Vector3iVector(faces)
                temp_mesh.compute_vertex_normals()
                pcd.normals = temp_mesh.vertex_normals
            
            # 3. 법선 방향 추정 및 일관성 확인 (다운샘플된 점들에 대해서만)
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            pcd.orient_normals_consistent_tangent_plane(k=100)
            
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
        print("\nConverting Mesh to PointCloud...")
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
        print("\nStarting 1st ICP registration...")
        current_transform = np.eye(4)
        
        for iteration in range(1000):
            result = o3d.pipelines.registration.registration_icp(
                source, target,
                2.0,  # 거리 임계값
                current_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-7,
                    relative_rmse=1e-7,
                    max_iteration=1
                )
            )
            
            if iteration % 10 == 0:  # Visualize every iteration
                print(f"  - ICP iteration {iteration}: fitness = {result.fitness:.6f}")
                
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
                    time.sleep(0.05)  # Adjust animation speed
            
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
                    relative_fitness=1e-8,
                    relative_rmse=1e-8,
                    max_iteration=1
                )
            )
            
            if iteration % 20 == 0:  # Visualize every iteration
                print(f"  - ICP iteration {iteration}: fitness = {result.fitness:.6f}")
                
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
                    time.sleep(0.05)  # Adjust animation speed
            
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
                    relative_fitness=1e-8,
                    relative_rmse=1e-8,
                    max_iteration=1
                )
            )
            
            if iteration % 20 == 0:  # Visualize every iteration
                print(f"  - ICP iteration {iteration}: fitness = {result.fitness:.6f}")
                
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
        
        # 변환된 소스 메시 생성
        transformed_source_mesh = copy.deepcopy(source_mesh)
        transformed_source_mesh.vertices = np.dot(
            source_mesh.vertices,
            current_transform[:3, :3].T
        ) + current_transform[:3, 3]
        
        return transformed_source_mesh, current_transform        


    def run_registration(self):
        if self.visualization:
            visualize_meshes([self.face_smile_mesh, self.face_rest_mesh, self.face_retraction_mesh], ["Smile", "Rest", "Retraction"], title="Initial Meshes")

        self.match_weight_centers()

        if self.visualization:
            visualize_meshes([self.face_smile_mesh, self.face_rest_mesh, self.face_retraction_mesh], ["Smile", "Rest", "Retraction"], title="Initial Meshes")

        moved_rest_mesh, rest_transform_matrix = self.fast_registration_with_vis(self.face_rest_mesh, self.face_smile_mesh)

        moved_retraction_mesh, retraction_transform_matrix = self.fast_registration_with_vis(self.face_retraction_mesh, self.face_smile_mesh)

        self.transform_matrix_for_rest = np.dot(rest_transform_matrix, self.transform_matrix_for_rest)
        self.transform_matrix_for_retraction = np.dot(retraction_transform_matrix, self.transform_matrix_for_retraction)

        if self.visualization:
            visualize_meshes([moved_rest_mesh, moved_retraction_mesh, self.face_smile_mesh], ["Rest", "Retraction", "Smile"], title="Final Meshes")

        return self.transform_matrix_for_rest, self.transform_matrix_for_retraction


if __name__ == "__main__":
    smile_mesh = Mesh.from_file("../../example/data/FaceScan/Smile/Smile.obj")
    transform_matrix = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
    faces_registration = FacesRegistration(smile_mesh, transform_matrix, "../../example/data/FaceScan/Rest/Smile.obj", "../../example/data/FaceScan/Retraction/Smile.obj", visualization=True)
    faces_registration.run_registration()

    print(faces_registration.transform_matrix_for_rest)
    print(faces_registration.transform_matrix_for_retraction)

    
        