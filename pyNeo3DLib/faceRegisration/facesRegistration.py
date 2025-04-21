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
    def __init__(self, face_smile_path, face_rest_path, face_retraction_path, visualization=False):
        self.face_smile_path = face_smile_path
        self.face_rest_path = face_rest_path
        self.face_retraction_path = face_retraction_path
        self.visualization = visualization

        self.transform_matrix_for_rest = np.eye(4)
        self.transform_matrix_for_retraction = np.eye(4)

        self.__load_models()
        
    def __load_models(self):
        self.face_smile_mesh = Mesh.from_file(self.face_smile_path)
        self.face_rest_mesh = Mesh.from_file(self.face_rest_path)
        self.face_retraction_mesh = Mesh.from_file(self.face_retraction_path)

        return self.face_smile_mesh, self.face_rest_mesh, self.face_retraction_mesh
    
    def match_weight_centers(self):
        """
        세 개의 메시(face_smile_mesh, face_rest_mesh, face_retraction_mesh)의 무게중심을
        face_smile_mesh의 무게중심을 기준으로 맞춥니다.
        
        Returns:
            변환된 face_rest_mesh와 face_retraction_mesh
        """
        # face_smile_mesh의 무게중심 계산
        smile_center = np.mean(self.face_smile_mesh.vertices, axis=0)
        
        # face_rest_mesh의 무게중심 계산 및 변환
        rest_center = np.mean(self.face_rest_mesh.vertices, axis=0)
        rest_translation = smile_center - rest_center
        self.face_rest_mesh.vertices = self.face_rest_mesh.vertices + rest_translation
        
        # face_rest_mesh의 변환 행렬 업데이트
        rest_transform = np.eye(4)
        rest_transform[:3, 3] = rest_translation
        self.transform_matrix_for_rest = np.dot(rest_transform, self.transform_matrix_for_rest)
        
        # face_retraction_mesh의 무게중심 계산 및 변환
        retraction_center = np.mean(self.face_retraction_mesh.vertices, axis=0)
        retraction_translation = smile_center - retraction_center
        self.face_retraction_mesh.vertices = self.face_retraction_mesh.vertices + retraction_translation
        
        # face_retraction_mesh의 변환 행렬 업데이트
        retraction_transform = np.eye(4)
        retraction_transform[:3, 3] = retraction_translation
        self.transform_matrix_for_retraction = np.dot(retraction_transform, self.transform_matrix_for_retraction)
        
        print(f"무게중심 정렬 완료:")
        print(f"  - Smile 메시 무게중심: {smile_center}")
        print(f"  - Rest 메시 무게중심: {rest_center} -> {np.mean(self.face_rest_mesh.vertices, axis=0)}")
        print(f"  - Retraction 메시 무게중심: {retraction_center} -> {np.mean(self.face_retraction_mesh.vertices, axis=0)}")
        print(f"  - Rest 메시 변환 행렬 업데이트 완료")
        print(f"  - Retraction 메시 변환 행렬 업데이트 완료")
        
        return self.face_rest_mesh, self.face_retraction_mesh

    
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
            
            down_sample_rate = 5
            # 먼저 다운샘플링 수행
            pcd = pcd.uniform_down_sample(every_k_points=down_sample_rate)
            
            # 2. 법선 벡터 처리 (다운샘플된 점들에 대해서만)
            if mesh.normals is not None:
                normals = np.asarray(mesh.normals)[::down_sample_rate]  # 다운샘플링과 동일한 비율로 법선 벡터 선택
                pcd.normals = o3d.utility.Vector3dVector(normals)
            else:
                # 다운샘플된 점들에 대해서만 법선 계산
                temp_mesh = o3d.geometry.TriangleMesh()
                temp_mesh.vertices = pcd.points
                temp_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
                temp_mesh.compute_vertex_normals()
                pcd.normals = temp_mesh.vertex_normals
            
            # 3. 법선 방향 추정 및 일관성 확인 (다운샘플된 점들에 대해서만)
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            pcd.orient_normals_consistent_tangent_plane(k=100)
            
            return pcd
            # # 1. 포인트 클라우드 생성
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(mesh.vertices)
            
            # # 2. 법선 벡터 처리
            # if mesh.no;.//;. rmals is not None:
            #     pcd.normals = o3d.utility.Vector3dVector(mesh.normals)
            # else:
            #     temp_mesh = o3d.geometry.TriangleMesh()
            #     temp_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
            #     temp_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
            #     temp_mesh.compute_vertex_normals()
            #     pcd.normals = temp_mesh.vertex_normals
            
            # # 3. 법선 방향 추정 및 일관성 확인
            # pcd.estimate_normals(
            #     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            # )
            # pcd.orient_normals_consistent_tangent_plane(k=100)
            
            # pcd.uniform_down_sample(every_k_points=200)
            
            # return pcd
        
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
                1.5,  # 거리 임계값
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


if __name__ == "__main__":
    faces_registration = FacesRegistration("../../example/data/FaceScan/Smile/Smile.obj", "../../example/data/FaceScan/Rest/Smile.obj", "../../example/data/FaceScan/Retraction/Smile.obj", visualization=True)
    faces_registration.run_registration()

    print(faces_registration.transform_matrix_for_rest)
    print(faces_registration.transform_matrix_for_retraction)

    
        