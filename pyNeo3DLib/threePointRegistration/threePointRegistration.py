import numpy as np
import time
import copy
from scipy.spatial import KDTree
import open3d as o3d
from typing import List, Dict, Any, Tuple
from ..fileLoader.mesh import Mesh

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("PyVista가 설치되지 않았습니다. 시각화 기능이 제한됩니다.")


# ===== 정확도 관련 상수 설정 (테스트용 조정 가능) =====
"""
정확도에 영향을 주는 주요 매개변수들을 여기서 조정할 수 있습니다.

테스트 결과 기준:
- ICP_DISTANCE_THRESHOLD = 0.1: 적합도 1.58%, RMSE 0.066mm (추천)
- ICP_DISTANCE_THRESHOLD = 0.05: 적합도 0.34%, RMSE 0.026mm (최고 정밀도)
- ICP_DISTANCE_THRESHOLD = 1.0: 적합도 68.7%, RMSE 0.46mm (넓은 매칭)
"""

# Region Growing 관련 상수
MAX_DISTANCE_FROM_SEED = 10.0        # 시드 점에서 최대 허용 거리 (mm) - 영역 크기 제한
DEFAULT_REGION_GROWING_RADIUS = 0.5  # 기본 region growing 반경 (mm) - 단계별 확산 거리
DEFAULT_NORMAL_SIMILARITY_THRESHOLD = 0.8  # 기본 법선 유사성 임계값 (0.0~1.0) - 높을수록 엄격

# ICP 관련 상수 - 정확도의 핵심!
ICP_DISTANCE_THRESHOLD = 1.0         # ICP 거리 임계값 (mm) - 매칭 기준 거리 (0.05~1.0 추천)
ICP_RELATIVE_FITNESS = 1e-7         # ICP 수렴 기준: fitness 변화량 - 작을수록 정밀
ICP_RELATIVE_RMSE = 1e-7            # ICP 수렴 기준: RMSE 변화량 - 작을수록 정밀
DEFAULT_ICP_MAX_ITERATIONS = 5000    # 기본 ICP 최대 반복 횟수 - 많을수록 정확하지만 느림

# 기타 정확도 관련 상수
VERTEX_NORMAL_EPSILON = 1e-8         # 법선 벡터 계산 시 0 방지용 epsilon


class ThreePointRegistration:
    """
    3점 정합을 수행하는 클래스
    
    주어진 두 메시와 각각의 3개 이상의 점 좌표를 사용하여:
    1. 각 점에서 가장 가까운 메시 정점을 찾고
    2. Region growing으로 주변 영역을 선택하고  
    3. ICP 알고리즘으로 정합하여 변환 행렬을 반환
    """
    
    def __init__(self, target_mesh_path: str, source_mesh_path: str, 
                 target_points: List[Dict[str, float]], source_points: List[Dict[str, float]],
                 region_growing_radius: float = DEFAULT_REGION_GROWING_RADIUS, 
                 icp_max_iterations: int = DEFAULT_ICP_MAX_ITERATIONS,
                 normal_similarity_threshold: float = DEFAULT_NORMAL_SIMILARITY_THRESHOLD, 
                 visualization: bool = False):
        """
        초기화
        
        Args:
            target_mesh_path: 타겟 메시 파일 경로
            source_mesh_path: 소스 메시 파일 경로  
            target_points: 타겟 메시의 점 좌표 리스트 [{"x": float, "y": float, "z": float}, ...]
            source_points: 소스 메시의 점 좌표 리스트 [{"x": float, "y": float, "z": float}, ...]
            region_growing_radius: Region growing 반경
            icp_max_iterations: ICP 최대 반복 횟수
            normal_similarity_threshold: 법선 벡터 유사성 임계값
            visualization: 시각화 여부
        """
        self.target_mesh_path = target_mesh_path
        self.source_mesh_path = source_mesh_path
        self.target_points = target_points
        self.source_points = source_points
        self.region_growing_radius = region_growing_radius
        self.icp_max_iterations = icp_max_iterations
        self.normal_similarity_threshold = normal_similarity_threshold
        self.visualization = visualization
        
        # 점 개수 검증
        if len(target_points) < 3 or len(source_points) < 3:
            raise ValueError("각 메시마다 최소 3개의 점이 필요합니다.")
        
        if len(target_points) != len(source_points):
            raise ValueError("타겟과 소스의 점 개수가 일치해야 합니다.")
    
    async def run_registration(self) -> np.ndarray:
        """
        3점 정합 실행
        
        Returns:
            np.ndarray: 4x4 변환 행렬
        """
        print(f"\n=== 3점 정합 시작 ===")
        start_time = time.time()
        
        # 1. 메시 로드
        print("1. 메시 로딩 중...")
        target_mesh = Mesh.from_file(self.target_mesh_path)
        source_mesh = Mesh.from_file(self.source_mesh_path)
        print(f"   타겟 메시: {len(target_mesh.vertices)} 정점, {len(target_mesh.faces)} 면")
        print(f"   소스 메시: {len(source_mesh.vertices)} 정점, {len(source_mesh.faces)} 면")
        
        # 2. 각 점에서 가장 가까운 메시 정점 찾기
        print("2. 가장 가까운 정점 찾기...")
        target_vertex_indices = self._find_closest_vertices(target_mesh, self.target_points)
        source_vertex_indices = self._find_closest_vertices(source_mesh, self.source_points)
        print(f"   타겟 정점 인덱스: {target_vertex_indices}")
        print(f"   소스 정점 인덱스: {source_vertex_indices}")
        
        # 3. Region Growing으로 각 점 주변 영역 선택
        print("3. Region Growing 수행 중...")
        target_regions = self._perform_region_growing_for_points(target_mesh, target_vertex_indices)
        source_regions = self._perform_region_growing_for_points(source_mesh, source_vertex_indices)
        
        # 4. Kabsch 알고리즘으로 초기 정렬
        print("4. Kabsch 알고리즘으로 초기 정렬 수행 중...")
        initial_transformation = self._perform_kabsch_alignment(
            source_mesh, target_mesh, source_vertex_indices, target_vertex_indices
        )
        
        # 5. 초기 변환 적용된 소스 영역 생성
        print("5. 초기 변환 적용 중...")
        transformed_source_regions = self._apply_transformation_to_mesh(source_regions, initial_transformation)
        
        # 6. ICP로 정밀 정합 수행
        print("6. ICP로 정밀 정합 수행 중...")
        icp_transformation = self._perform_icp_registration(transformed_source_regions, target_regions)
        
        # 7. 최종 변환 행렬 계산 (Kabsch + ICP)
        transformation_matrix = np.dot(icp_transformation, initial_transformation)
        
        # 8. 시각화 (옵션)
        if self.visualization:
            print("8. 결과 시각화...")
            self._visualize_registration_result(
                target_mesh, source_mesh, 
                target_regions, source_regions,
                transformation_matrix,
                target_vertex_indices, source_vertex_indices,
                initial_transformation
            )
        
        elapsed_time = time.time() - start_time
        print(f"=== 3점 정합 완료 (소요시간: {elapsed_time:.2f}초) ===\n")
        
        return transformation_matrix
    
    def _find_closest_vertices(self, mesh: Mesh, points: List[Dict[str, float]]) -> List[int]:
        """
        주어진 점 좌표에서 가장 가까운 메시 정점들을 찾기
        
        Args:
            mesh: 메시 객체
            points: 점 좌표 리스트
            
        Returns:
            List[int]: 가장 가까운 정점들의 인덱스 리스트
        """
        print(f"   KDTree 생성 중... (정점 수: {len(mesh.vertices)})")
        tree = KDTree(mesh.vertices)
        
        closest_vertices = []
        for i, point in enumerate(points):
            point_coords = np.array([point["x"], point["y"], point["z"]])
            distance, idx = tree.query(point_coords, k=1)
            closest_vertices.append(idx)
            print(f"   점 {i+1}: 좌표 {point_coords} -> 정점 {idx} (거리: {distance:.3f})")
            
        return closest_vertices
    
    def _perform_region_growing_for_points(self, mesh: Mesh, seed_vertex_indices: List[int]) -> Mesh:
        """
        여러 시드 포인트에서 Region Growing 수행
        
        Args:
            mesh: 전체 메시
            seed_vertex_indices: 시드 정점 인덱스 리스트
            
        Returns:
            Mesh: 선택된 영역들을 합친 메시
        """
        print(f"   시드 포인트 {len(seed_vertex_indices)}개에서 Region Growing 시작")
        
        # 1. 메시 데이터 준비
        vertices = mesh.vertices
        faces = mesh.faces
        
        # 2. 법선 벡터 계산
        vertex_normals = self._compute_vertex_normals(vertices, faces)
        
        # 3. KDTree 생성
        tree = KDTree(vertices)
        
        # 4. 모든 시드 포인트에서 Region Growing 수행
        all_selected_vertices = set()
        
        for i, seed_idx in enumerate(seed_vertex_indices):
            print(f"   시드 포인트 {i+1}/{len(seed_vertex_indices)} 처리 중...")
            selected_vertices = self._region_growing_from_seed(
                vertices, vertex_normals, tree, seed_idx
            )
            all_selected_vertices.update(selected_vertices)
            print(f"   시드 {i+1}에서 {len(selected_vertices)}개 정점 선택됨 ({MAX_DISTANCE_FROM_SEED}mm 제한 적용)")
        
        print(f"   총 {len(all_selected_vertices)}개 정점 선택됨")
        
        # 5. 선택된 정점들로 새 메시 생성
        return self._create_mesh_from_vertices(mesh, list(all_selected_vertices))
    
    def _compute_vertex_normals(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """
        정점 법선 벡터 계산
        
        Args:
            vertices: 정점 배열
            faces: 면 배열
            
        Returns:
            np.ndarray: 정점 법선 벡터 배열
        """
        # 면 법선 계산
        v1 = vertices[faces[:, 0]]
        v2 = vertices[faces[:, 1]]
        v3 = vertices[faces[:, 2]]
        
        vec1 = v2 - v1
        vec2 = v3 - v1
        
        face_normals = np.cross(vec1, vec2)
        face_norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
        face_norms[face_norms == 0] = VERTEX_NORMAL_EPSILON
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
        norms[norms == 0] = VERTEX_NORMAL_EPSILON
        vertex_normals = vertex_normals / norms
        
        return vertex_normals
    
    def _region_growing_from_seed(self, vertices: np.ndarray, vertex_normals: np.ndarray, 
                                tree: KDTree, seed_idx: int) -> List[int]:
        """
        단일 시드에서 Region Growing 수행 (시드 점 기준 최대 거리 제한)
        
        Args:
            vertices: 정점 배열
            vertex_normals: 정점 법선 배열
            tree: KDTree 객체
            seed_idx: 시드 정점 인덱스
            
        Returns:
            List[int]: 선택된 정점 인덱스 리스트
        """
        selected_vertices = set()
        queue = [seed_idx]
        selected_vertices.add(seed_idx)
        
        seed_normal = vertex_normals[seed_idx]
        seed_vertex = vertices[seed_idx]  # 시드 점 좌표 저장
        
        while queue:
            current_idx = queue.pop(0)
            current_vertex = vertices[current_idx]
            
            # 반경 내 이웃 정점들 찾기
            neighbor_indices = tree.query_ball_point(current_vertex, self.region_growing_radius)
            
            for neighbor_idx in neighbor_indices:
                if neighbor_idx in selected_vertices:
                    continue
                
                neighbor_vertex = vertices[neighbor_idx]
                
                # 시드 점에서의 거리 확인 (핵심 추가 부분!)
                distance_from_seed = np.linalg.norm(neighbor_vertex - seed_vertex)
                if distance_from_seed > MAX_DISTANCE_FROM_SEED:
                    continue  # 설정된 최대 거리를 넘으면 제외
                
                # 법선 벡터 유사성 검사
                neighbor_normal = vertex_normals[neighbor_idx]
                similarity = np.dot(seed_normal, neighbor_normal)
                
                if similarity > self.normal_similarity_threshold:
                    selected_vertices.add(neighbor_idx)
                    queue.append(neighbor_idx)
        
        return list(selected_vertices)
    
    def _perform_kabsch_alignment(self, source_mesh: Mesh, target_mesh: Mesh,
                                source_vertex_indices: List[int], target_vertex_indices: List[int]) -> np.ndarray:
        """
        Kabsch 알고리즘을 사용하여 초기 정렬 수행
        
        Args:
            source_mesh: 소스 메시
            target_mesh: 타겟 메시
            source_vertex_indices: 소스 정점 인덱스들
            target_vertex_indices: 타겟 정점 인덱스들
            
        Returns:
            np.ndarray: 4x4 초기 변환 행렬
        """
        # 대응점들 추출
        source_points = source_mesh.vertices[source_vertex_indices]
        target_points = target_mesh.vertices[target_vertex_indices]
        
        print(f"   소스 점들: {source_points.shape}")
        print(f"   타겟 점들: {target_points.shape}")
        
        # 중심점 계산
        source_centroid = np.mean(source_points, axis=0)
        target_centroid = np.mean(target_points, axis=0)
        
        print(f"   소스 중심점: {source_centroid}")
        print(f"   타겟 중심점: {target_centroid}")
        
        # 중심점으로 이동
        source_centered = source_points - source_centroid
        target_centered = target_points - target_centroid
        
        # 교차 공분산 행렬 계산
        H = np.dot(source_centered.T, target_centered)
        
        # SVD 분해
        U, S, Vt = np.linalg.svd(H)
        
        # 회전 행렬 계산
        R = np.dot(Vt.T, U.T)
        
        # 반사 방지 (determinant가 음수인 경우)
        if np.linalg.det(R) < 0:
            print("   반사 변환 감지, 보정 중...")
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)
        
        # 이동 벡터 계산
        t = target_centroid - np.dot(R, source_centroid)
        
        # 4x4 변환 행렬 생성
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = R
        transformation_matrix[:3, 3] = t
        
        # 정렬 품질 평가
        transformed_source_points = self._transform_points(source_points, transformation_matrix)
        rmse = np.sqrt(np.mean(np.sum((transformed_source_points - target_points)**2, axis=1)))
        
        print(f"   Kabsch 정렬 RMSE: {rmse:.6f}")
        print(f"   회전 행렬 행렬식: {np.linalg.det(R):.6f}")
        
        return transformation_matrix
    
    def _apply_transformation_to_mesh(self, mesh: Mesh, transformation_matrix: np.ndarray) -> Mesh:
        """
        메시에 변환 행렬을 적용하여 새로운 메시 생성
        
        Args:
            mesh: 원본 메시
            transformation_matrix: 4x4 변환 행렬
            
        Returns:
            Mesh: 변환된 메시
        """
        transformed_mesh = copy.deepcopy(mesh)
        transformed_mesh.vertices = self._transform_points(mesh.vertices, transformation_matrix)
        
        # 법선 벡터도 변환 (회전만 적용)
        if mesh.normals is not None:
            rotation_matrix = transformation_matrix[:3, :3]
            transformed_mesh.normals = np.dot(mesh.normals, rotation_matrix.T)
        
        return transformed_mesh
    
    def _create_mesh_from_vertices(self, original_mesh: Mesh, vertex_indices: List[int]) -> Mesh:
        """
        선택된 정점들로부터 새 메시 생성
        
        Args:
            original_mesh: 원본 메시
            vertex_indices: 선택된 정점 인덱스 리스트
            
        Returns:
            Mesh: 새로 생성된 메시
        """
        vertex_set = set(vertex_indices)
        
        # 선택된 정점들만 추출
        selected_vertices = original_mesh.vertices[vertex_indices]
        
        # 정점 인덱스 매핑 생성
        old_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(vertex_indices)}
        
        # 선택된 정점들로만 구성된 면들 찾기
        selected_faces = []
        for face in original_mesh.faces:
            if all(v_idx in vertex_set for v_idx in face):
                new_face = [old_to_new_idx[v_idx] for v_idx in face]
                selected_faces.append(new_face)
        
        # 새 메시 생성
        new_mesh = Mesh()
        new_mesh.vertices = selected_vertices
        new_mesh.faces = np.array(selected_faces)
        
        # 법선 벡터가 있다면 복사
        if original_mesh.normals is not None:
            new_mesh.normals = original_mesh.normals[vertex_indices]
        
        return new_mesh
    
    def _perform_icp_registration(self, source_regions: Mesh, target_regions: Mesh) -> np.ndarray:
        """
        ICP 알고리즘으로 정합 수행
        
        Args:
            source_regions: 소스 영역 메시
            target_regions: 타겟 영역 메시
            
        Returns:
            np.ndarray: 4x4 변환 행렬
        """
        print(f"   소스 영역: {len(source_regions.vertices)} 정점")
        print(f"   타겟 영역: {len(target_regions.vertices)} 정점")
        
        # Mesh를 Open3D PointCloud로 변환
        source_pcd = self._mesh_to_pointcloud(source_regions)
        target_pcd = self._mesh_to_pointcloud(target_regions)
        
        print(f"   소스 포인트클라우드: {len(source_pcd.points)} 점")
        print(f"   타겟 포인트클라우드: {len(target_pcd.points)} 점")
        
        # ICP 정합 수행
        threshold = ICP_DISTANCE_THRESHOLD  # 거리 임계값
        
        # 초기 변환 행렬
        initial_transform = np.eye(4)
        
        print(f"   ICP 시작 (최대 반복: {self.icp_max_iterations}, 임계값: {threshold})")
        
        # Point-to-Point ICP
        result = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd,
            threshold,
            initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=ICP_RELATIVE_FITNESS,
                relative_rmse=ICP_RELATIVE_RMSE,
                max_iteration=self.icp_max_iterations
            )
        )
        
        print(f"   ICP 완료 - 적합도: {result.fitness:.6f}, RMSE: {result.inlier_rmse:.6f}")
        
        return result.transformation
    
    def _mesh_to_pointcloud(self, mesh: Mesh) -> o3d.geometry.PointCloud:
        """
        Mesh를 Open3D PointCloud로 변환
        
        Args:
            mesh: 변환할 메시
            
        Returns:
            o3d.geometry.PointCloud: 변환된 포인트클라우드
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(mesh.vertices)
        
        # 법선 벡터가 있다면 설정
        if mesh.normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(mesh.normals)
        else:
            # 법선 벡터 추정
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
        
        return pcd
    
    def _visualize_registration_result(self, target_mesh: Mesh, source_mesh: Mesh,
                                     target_regions: Mesh, source_regions: Mesh,
                                     transformation_matrix: np.ndarray,
                                     target_vertex_indices: List[int], source_vertex_indices: List[int],
                                     initial_transformation: np.ndarray = None):
        """
        3점 정합 결과를 시각화
        
        Args:
            target_mesh: 타겟 메시
            source_mesh: 소스 메시
            target_regions: 타겟 영역
            source_regions: 소스 영역
            transformation_matrix: 변환 행렬
            target_vertex_indices: 타겟 정점 인덱스들
            source_vertex_indices: 소스 정점 인덱스들
        """
        if not PYVISTA_AVAILABLE:
            print("   PyVista가 설치되지 않아 시각화를 건너뜁니다.")
            return
        
        try:
            # PyVista 플로터 생성
            plotter = pv.Plotter(window_size=(1200, 800))
            plotter.set_background('white')
            
            # 1. 원본 타겟 메시 (반투명 파란색)
            target_pv = self._mesh_to_pyvista(target_mesh)
            plotter.add_mesh(target_pv, color='lightblue', opacity=0.3, 
                           label='Target Mesh (Original)')
            
            # # 2. 원본 소스 메시 (반투명 빨간색)
            # source_pv = self._mesh_to_pyvista(source_mesh)
            # plotter.add_mesh(source_pv, color='lightcoral', opacity=0.3, 
            #                label='Source Mesh (Original)')
            
            # # 3. Kabsch 정렬된 소스 메시 (중간 단계, 연한 주황색)
            # if initial_transformation is not None:
            #     kabsch_source_mesh = self._apply_transformation(source_mesh, initial_transformation)
            #     kabsch_source_pv = self._mesh_to_pyvista(kabsch_source_mesh)
            #     plotter.add_mesh(kabsch_source_pv, color='orange', opacity=0.5,
            #                    label='Source Mesh (Kabsch Aligned)')
            
            # 4. 최종 변환된 소스 메시 (진한 빨간색)
            transformed_source_mesh = self._apply_transformation(source_mesh, transformation_matrix)
            transformed_source_pv = self._mesh_to_pyvista(transformed_source_mesh)
            plotter.add_mesh(transformed_source_pv, color='red', opacity=0.7,
                           label='Source Mesh (Final)')
            
            # 5. 타겟 영역 (진한 파란색)
            target_regions_pv = self._mesh_to_pyvista(target_regions)
            plotter.add_mesh(target_regions_pv, color='blue', opacity=0.8,
                           label='Target Regions')
            
            # 6. 소스 영역 (진한 보라색)
            source_regions_pv = self._mesh_to_pyvista(source_regions)
            plotter.add_mesh(source_regions_pv, color='purple', opacity=0.8,
                           label='Source Regions')
            
            # 7. 선택된 점들 표시
            # 타겟 점들 (파란색 구)
            target_points = target_mesh.vertices[target_vertex_indices]
            plotter.add_points(target_points, color='darkblue', point_size=15,
                             render_points_as_spheres=True, label='Target Points')
            
            # 소스 점들 (빨간색 구)
            source_points = source_mesh.vertices[source_vertex_indices]
            plotter.add_points(source_points, color='darkred', point_size=15,
                             render_points_as_spheres=True, label='Source Points (Original)')
            
            # Kabsch 정렬된 소스 점들 (주황색 구)
            if initial_transformation is not None:
                kabsch_source_points = self._transform_points(source_points, initial_transformation)
                plotter.add_points(kabsch_source_points, color='darkorange', point_size=15,
                                 render_points_as_spheres=True, label='Source Points (Kabsch)')
            
            # 8. 최종 변환된 소스 점들 (녹색 구)
            transformed_source_points = self._transform_points(source_points, transformation_matrix)
            plotter.add_points(transformed_source_points, color='green', point_size=15,
                             render_points_as_spheres=True, label='Source Points (Final)')
            
            # 9. 대응점들 연결선
            for i in range(len(target_points)):
                line = pv.Line(target_points[i], transformed_source_points[i])
                plotter.add_mesh(line, color='yellow', line_width=3)
            
            # 범례 및 제목 추가
            plotter.add_legend(size=(0.3, 0.3), loc='upper right')
            plotter.add_title('Three-Point Registration Result', font_size=16)
            plotter.add_axes()
            
            # 카메라 설정
            plotter.camera_position = 'iso'
            
            print("   시각화 창이 열렸습니다. 창을 닫으면 계속 진행됩니다.")
            plotter.show()
            
        except Exception as e:
            print(f"   시각화 중 오류 발생: {e}")
    
    def _mesh_to_pyvista(self, mesh: Mesh) -> pv.PolyData:
        """
        Mesh 객체를 PyVista PolyData로 변환
        
        Args:
            mesh: 변환할 메시
            
        Returns:
            pv.PolyData: PyVista 메시
        """
        vertices = mesh.vertices
        faces = mesh.faces
        
        # PyVista 형식으로 면 데이터 변환
        pv_faces = []
        for face in faces:
            pv_faces.append(len(face))  # 면의 정점 개수
            pv_faces.extend(face)       # 정점 인덱스들
        
        return pv.PolyData(vertices, pv_faces)
    
    def _apply_transformation(self, mesh: Mesh, transformation_matrix: np.ndarray) -> Mesh:
        """
        메시에 변환 행렬 적용
        
        Args:
            mesh: 원본 메시
            transformation_matrix: 4x4 변환 행렬
            
        Returns:
            Mesh: 변환된 메시
        """
        transformed_mesh = copy.deepcopy(mesh)
        transformed_vertices = self._transform_points(mesh.vertices, transformation_matrix)
        transformed_mesh.vertices = transformed_vertices
        return transformed_mesh
    
    def _transform_points(self, points: np.ndarray, transformation_matrix: np.ndarray) -> np.ndarray:
        """
        점들에 변환 행렬 적용
        
        Args:
            points: 변환할 점들 (N x 3)
            transformation_matrix: 4x4 변환 행렬
            
        Returns:
            np.ndarray: 변환된 점들
        """
        # 동차 좌표로 변환
        homogeneous_points = np.hstack([points, np.ones((points.shape[0], 1))])
        
        # 변환 적용
        transformed_homogeneous = np.dot(homogeneous_points, transformation_matrix.T)
        
        # 3D 좌표로 변환
        return transformed_homogeneous[:, :3] / transformed_homogeneous[:, 3:4]
