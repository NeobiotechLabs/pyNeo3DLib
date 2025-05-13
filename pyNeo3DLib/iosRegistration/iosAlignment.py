import numpy as np
from pyNeo3DLib.fileLoader.mesh import Mesh
import pyvista as pv
import numpy.linalg as LA
from scipy.spatial import ConvexHull

class IosAlignment:
    def __init__(self, ios_path):
        self.ios_path = ios_path
        self.__load_models()

    def __load_models(self):
        print(f"loading model from {self.ios_path}")
        self.ios_mesh = Mesh()
        self.ios_mesh = self.ios_mesh.from_file(self.ios_path)
        print(self.ios_mesh.faces)    
        
    def run_analysis(self):
        print("ios_analysis")
        axe_aligned_mesh, transform_matrix = self.align_mesh_to_axes()
        aligned_z_mesh, rotation_matrix_z = self.orient_mesh_z_axis(axe_aligned_mesh)
        result = self.find_molars_and_align_to_y(aligned_z_mesh)
        y_aligned_mesh, was_rotated, left_molar, right_molar, center = result
        
        # 모든 변환 행렬 계산
        # 1. z축 회전 행렬과 OBB 정렬 행렬 결합
        combined_matrix = np.matmul(rotation_matrix_z, transform_matrix)
        
        # 2. 어금니 기반 y축 정렬이 필요한 경우 해당 변환 행렬 추가
        if was_rotated:
            # 180도 회전 변환 행렬
            y_rotation_matrix = np.array([
                [-1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            combined_matrix = np.matmul(y_rotation_matrix, combined_matrix)
        
        # 최종 메시와 변환 행렬 반환
        return y_aligned_mesh, combined_matrix

    def calculate_centroid(self, mesh):
        """메시의 무게중심 계산"""
        return np.mean(mesh.vertices, axis=0)
    
    def calculate_obb(self, mesh):
        """
        메시의 OBB(Oriented Bounding Box) 계산
        
        Returns:
            center (np.ndarray): OBB의 중심
            axes (np.ndarray): OBB의 주축 방향 (3x3 행렬)
            extents (np.ndarray): 각 축 방향으로의 반길이
        """
        # 무게중심 계산
        centroid = self.calculate_centroid(mesh)
        
        # 무게중심 중심으로 이동
        centered_vertices = mesh.vertices - centroid
        
        # 공분산 행렬 계산
        covariance_matrix = np.cov(centered_vertices, rowvar=False)
        
        # 고유값과 고유벡터 계산
        eigenvalues, eigenvectors = LA.eigh(covariance_matrix)
        
        # 고유벡터가 정규화되어 있는지 확인
        for i in range(3):
            eigenvectors[:, i] = eigenvectors[:, i] / LA.norm(eigenvectors[:, i])
        
        # 각 축 방향으로 투영
        projected = np.dot(centered_vertices, eigenvectors)
        
        # 경계 계산
        min_projected = np.min(projected, axis=0)
        max_projected = np.max(projected, axis=0)
        
        # 축 방향 길이의 절반
        extents = (max_projected - min_projected) / 2
        
        # OBB 중심은 원래 좌표계에서 각 축의 중간점
        obb_center = centroid + np.dot(eigenvectors, (min_projected + max_projected) / 2)
        
        return obb_center, eigenvectors, extents
    
    def visualize_mesh_features(self):
        """메시의 무게중심, OBB, OBB 중심을 시각화"""
        # PyVista 플로터 생성
        plotter = pv.Plotter()
        
        # 메시를 PyVista 메시로 변환
        mesh_points = self.ios_mesh.vertices
        
        # 면 정보 변환 (PyVista 형식)
        faces = []
        for face in self.ios_mesh.faces:
            faces.append(len(face))  # 면의 정점 수 (보통 3)
            faces.extend(face)  # 면을 구성하는 정점 인덱스
        
        pv_mesh = pv.PolyData(mesh_points, faces)
        
        # 무게중심 계산
        centroid = self.calculate_centroid(self.ios_mesh)
        
        # OBB 계산
        obb_center, obb_axes, obb_extents = self.calculate_obb(self.ios_mesh)
        
        # 메시 추가
        plotter.add_mesh(pv_mesh, color='lightblue', opacity=0.7, show_edges=True)
        
        # 무게중심 표시 (빨간색 구)
        plotter.add_mesh(pv.Sphere(radius=max(obb_extents)/20, center=centroid), 
                       color='red', render_points_as_spheres=True, 
                       label='Center of Mass')
        
        # OBB 중심 표시 (녹색 구)
        plotter.add_mesh(pv.Sphere(radius=max(obb_extents)/20, center=obb_center), 
                       color='green', render_points_as_spheres=True,
                       label='OBB Center')
        
        # OBB 표시
        corners = []
        for i in [-1, 1]:
            for j in [-1, 1]:
                for k in [-1, 1]:
                    # OBB 모서리 계산
                    corner = obb_center + i * obb_extents[0] * obb_axes[:, 0] + \
                             j * obb_extents[1] * obb_axes[:, 1] + \
                             k * obb_extents[2] * obb_axes[:, 2]
                    corners.append(corner)
        
        # OBB의 모서리를 선으로 연결
        edges = [
            # 아래쪽 사각형
            [0, 1], [1, 3], [3, 2], [2, 0],
            # 위쪽 사각형
            [4, 5], [5, 7], [7, 6], [6, 4],
            # 연결 선
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        
        # 모서리 연결해서 OBB 시각화
        for edge in edges:
            line = pv.Line(corners[edge[0]], corners[edge[1]])
            plotter.add_mesh(line, color='yellow', line_width=3)
        
        # 축 인디케이터 추가
        plotter.add_axes()
        
        # 범례 추가
        plotter.add_legend()
        
        # 제목 설정
        plotter.add_title("Mesh Features Visualization: Center of Mass and OBB", font_size=16)
        
        # 시각화
        plotter.show()
        
        # 자원 해제
        plotter.close()

    def align_mesh_to_axes(self):
        """
        메시를 좌표축에 정렬합니다:
        - OBB의 가장 짧은 축을 z축과 정렬
        - OBB의 가장 긴 축을 x축과 정렬
        
        Returns:
            pv.PolyData: 정렬된 메시
        """
        # OBB 계산
        obb_center, obb_axes, obb_extents = self.calculate_obb(self.ios_mesh)
        
        # 축 길이에 따라 인덱스 정렬 (오름차순)
        idx_by_length = np.argsort(obb_extents)
        
        # 가장 짧은 축과 가장 긴 축 식별
        shortest_axis_idx = idx_by_length[0]
        longest_axis_idx = idx_by_length[2]
        middle_axis_idx = idx_by_length[1]
        
        # 각 축 방향 벡터 추출
        z_axis = obb_axes[:, shortest_axis_idx]  # 가장 짧은 축 -> z축
        x_axis = obb_axes[:, longest_axis_idx]   # 가장 긴 축 -> x축
        
        # 축 방향이 음수인 경우 반전 (선택 사항)
        if z_axis[2] < 0:  # z축이 글로벌 -z 방향을 향하면 반전
            z_axis = -z_axis
        
        if x_axis[0] < 0:  # x축이 글로벌 -x 방향을 향하면 반전
            x_axis = -x_axis
        
        # 직교 좌표계 생성
        # 우선 z축을 정확히 정의
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        # x축과 z축이 직교하도록 조정
        x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # y축은 x축과 z축의 외적으로 계산 (오른손 좌표계)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # 회전 행렬 구성
        rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
        
        # 회전 행렬이 직교행렬인지 확인
        if not np.allclose(np.dot(rotation_matrix.T, rotation_matrix), np.eye(3), rtol=1e-5, atol=1e-5):
            print("경고: 계산된 회전 행렬이 직교행렬이 아닙니다.")
        
        # 회전 행렬을 표준 좌표계로 변환
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix.T  # 표준 좌표계로 변환하기 위해 전치
        transform_matrix[:3, 3] = -np.dot(rotation_matrix.T, obb_center)  # 중심점 이동
        
        # 메시를 표준 좌표계로 변환
        mesh_points = self.ios_mesh.vertices
        faces = []
        for face in self.ios_mesh.faces:
            faces.append(len(face))
            faces.extend(face)
        
        original_pv_mesh = pv.PolyData(mesh_points, faces)
        
        # pyvista의 transform_matrix를 사용하여 메시 변환
        aligned_mesh = original_pv_mesh.copy()
        aligned_mesh.transform(transform_matrix)
        
        return aligned_mesh, transform_matrix

    def visualize_aligned_mesh(self):
        """정렬된 메시와 기존 메시를 함께 시각화합니다"""
        # 정렬된 메시 계산
        aligned_mesh, transform_matrix = self.align_mesh_to_axes()
        
        # OBB 계산 (원본 메시)
        original_obb_center, original_obb_axes, original_obb_extents = self.calculate_obb(self.ios_mesh)
        
        # 정렬된 메시용 좌표축 기반 OBB 생성
        aligned_obb_extents = original_obb_extents.copy()
        
        # 새로운 축 방향 (표준 좌표계)
        aligned_box_axes = np.eye(3)
        
        # 표준 좌표계에서 OBB 중심 계산
        aligned_center = np.zeros(3)  # 원점으로 설정 또는 정렬된 메시 중심으로 설정
        
        # 플로터 생성
        plotter = pv.Plotter()
        
        # 원본 메시 추가 (반투명 회색)
        mesh_points = self.ios_mesh.vertices
        faces = []
        for face in self.ios_mesh.faces:
            faces.append(len(face))
            faces.extend(face)
        original_pv_mesh = pv.PolyData(mesh_points, faces)
        plotter.add_mesh(original_pv_mesh, color='gray', opacity=0.3, show_edges=False)
        
        # 정렬된 메시 추가 (파란색)
        plotter.add_mesh(aligned_mesh, color='lightblue', opacity=0.7, show_edges=True)
        
        # 좌표축 시각화
        plotter.add_axes()
        
        # 원본 OBB 표시 (노란색)
        self._add_obb_to_plotter(plotter, original_obb_center, original_obb_axes, original_obb_extents, color='yellow')
        
        # 정렬된 OBB 표시 (빨간색)
        # 정렬된 메시의 AABB 계산 (정렬 후이므로 좌표축 방향 AABB가 OBB와 같음)
        aligned_points = np.array(aligned_mesh.points)
        min_bounds = np.min(aligned_points, axis=0)
        max_bounds = np.max(aligned_points, axis=0)
        
        # AABB 중심 계산
        aabb_center = (min_bounds + max_bounds) / 2
        
        # AABB 크기 계산
        aabb_extents = (max_bounds - min_bounds) / 2
        
        # 정렬된 OBB는 AABB와 같음
        aligned_obb_center = aabb_center
        aligned_obb_extents = aabb_extents
        
        # 정렬된 OBB 표시
        self._add_obb_to_plotter(plotter, aligned_obb_center, aligned_box_axes, aligned_obb_extents, color='red')
        
        # 제목 설정
        plotter.add_title("Aligned Mesh with OBB", font_size=16)
        
        # 시각화
        plotter.show()
        
        # 자원을 명시적으로 해제
        plotter.close()

        return aligned_mesh.copy(deep=True)

    def _add_obb_to_plotter(self, plotter, center, axes, extents, color='yellow'):
        """OBB를 플로터에 추가합니다"""
        corners = []
        for i in [-1, 1]:
            for j in [-1, 1]:
                for k in [-1, 1]:
                    corner = center + i * extents[0] * axes[:, 0] + j * extents[1] * axes[:, 1] + k * extents[2] * axes[:, 2]
                    corners.append(corner)
        
        edges = [
            [0, 1], [1, 3], [3, 2], [2, 0],  # 아래쪽 사각형
            [4, 5], [5, 7], [7, 6], [6, 4],  # 위쪽 사각형
            [0, 4], [1, 5], [2, 6], [3, 7]   # 연결 선
        ]
        
        for edge in edges:
            line = pv.Line(corners[edge[0]], corners[edge[1]])
            plotter.add_mesh(line, color=color, line_width=3)

    def determine_flat_side(self, mesh_points, mesh_faces):
        """
        메시의 평평한 부분을 찾아 z축 방향으로 정렬합니다.
        치아 인상 모델의 특성을 고려한 개선된 방법을 사용합니다.
        
        Args:
            mesh_points: 메시의 정점 좌표
            mesh_faces: 메시의 면 정보
            
        Returns:
            bool: 뒤집어야 하면 True, 그대로 두면 False
        """
        # PyVista 메시로 변환
        faces_list = []
        for face in mesh_faces:
            faces_list.append(len(face))
            faces_list.extend(face)
        
        pv_mesh = pv.PolyData(mesh_points, faces_list)
        
        # 면 법선 벡터 계산
        pv_mesh.compute_normals(inplace=True)
        normals = pv_mesh.point_data['Normals']
        
        # 메시의 z값 범위 분석
        z_values = mesh_points[:, 2]
        z_min, z_max = np.min(z_values), np.max(z_values)
        z_range = z_max - z_min
        
        # 하단부와 상단부 정의 (z값 기준으로 상위/하위 15%)
        lower_threshold = z_min + 0.15 * z_range
        upper_threshold = z_max - 0.15 * z_range
        
        lower_mask = z_values <= lower_threshold
        upper_mask = z_values >= upper_threshold
        
        lower_region = mesh_points[lower_mask]
        upper_region = mesh_points[upper_mask]
        
        # 1. 변동성 분석: z 값의 표준편차 계산
        lower_z_std = np.std(lower_region[:, 2]) if len(lower_region) > 0 else float('inf')
        upper_z_std = np.std(upper_region[:, 2]) if len(upper_region) > 0 else float('inf')
        
        print(f"하단부 z 표준편차: {lower_z_std:.6f}")
        print(f"상단부 z 표준편차: {upper_z_std:.6f}")
        
        # 2. 복잡도 분석: 각 영역 내 법선 벡터의 다양성 계산
        lower_normal_diversity = self.calculate_normal_diversity(normals[lower_mask]) if np.any(lower_mask) else float('inf')
        upper_normal_diversity = self.calculate_normal_diversity(normals[upper_mask]) if np.any(upper_mask) else float('inf')
        
        print(f"하단부 법선 다양성: {lower_normal_diversity:.6f}")
        print(f"상단부 법선 다양성: {upper_normal_diversity:.6f}")
        
        # 3. 수직 법선 분석: z 방향으로 향하는 법선의 비율
        lower_vertical_ratio = np.mean(np.abs(normals[lower_mask][:, 2])) if np.any(lower_mask) else 0
        upper_vertical_ratio = np.mean(np.abs(normals[upper_mask][:, 2])) if np.any(upper_mask) else 0
        
        print(f"하단부 수직 법선 비율: {lower_vertical_ratio:.6f}")
        print(f"상단부 수직 법선 비율: {upper_vertical_ratio:.6f}")
        
        # 4. 높이 분포 분석: 각 영역 내 z 값의 분포 형태
        lower_height_distribution = self.analyze_height_distribution(lower_region[:, 2]) if len(lower_region) > 0 else 0
        upper_height_distribution = self.analyze_height_distribution(upper_region[:, 2]) if len(upper_region) > 0 else 0
        
        print(f"하단부 높이 분포 평탄도: {lower_height_distribution:.6f}")
        print(f"상단부 높이 분포 평탄도: {upper_height_distribution:.6f}")
        
        # 5. 치아 인상 특성 기반 점수 계산 (바닥이 평평하고 상단이 복잡함)
        # - 낮은 z 표준편차: 평평한 면일 가능성 높음
        # - 낮은 법선 다양성: 평평한 면일 가능성 높음
        # - 높은 수직 법선 비율: 바닥면일 가능성 높음
        # - 높은 높이 분포 평탄도: 평평한 면일 가능성 높음
        
        # 각 특성을 0~1 범위로 정규화 (값이 작을수록 1에 가까움)
        z_std_ratio = min(upper_z_std, lower_z_std) / (max(upper_z_std, lower_z_std) + 1e-6)
        normal_div_ratio = min(upper_normal_diversity, lower_normal_diversity) / (max(upper_normal_diversity, lower_normal_diversity) + 1e-6)
        
        # 바닥을 식별하는 점수 (값이 높을수록 바닥일 가능성이 높음)
        lower_base_score = (
            (1.0 - (lower_z_std / (upper_z_std + lower_z_std + 1e-6))) * 0.35 +
            (1.0 - (lower_normal_diversity / (upper_normal_diversity + lower_normal_diversity + 1e-6))) * 0.25 +
            (lower_vertical_ratio / (lower_vertical_ratio + upper_vertical_ratio + 1e-6)) * 0.20 +
            (lower_height_distribution / (lower_height_distribution + upper_height_distribution + 1e-6)) * 0.20
        )
        
        upper_base_score = (
            (1.0 - (upper_z_std / (upper_z_std + lower_z_std + 1e-6))) * 0.35 +
            (1.0 - (upper_normal_diversity / (upper_normal_diversity + lower_normal_diversity + 1e-6))) * 0.25 +
            (upper_vertical_ratio / (lower_vertical_ratio + upper_vertical_ratio + 1e-6)) * 0.20 +
            (upper_height_distribution / (lower_height_distribution + upper_height_distribution + 1e-6)) * 0.20
        )
        
        print(f"하단부 바닥 점수: {lower_base_score:.4f}")
        print(f"상단부 바닥 점수: {upper_base_score:.4f}")
        
        # 6. 직접적인 z축 법선 분석 (바닥면은 +z 또는 -z 방향 법선이 많음)
        lower_positive_z = np.sum(normals[lower_mask][:, 2] > 0.9) if np.any(lower_mask) else 0
        lower_negative_z = np.sum(normals[lower_mask][:, 2] < -0.9) if np.any(lower_mask) else 0
        upper_positive_z = np.sum(normals[upper_mask][:, 2] > 0.9) if np.any(upper_mask) else 0
        upper_negative_z = np.sum(normals[upper_mask][:, 2] < -0.9) if np.any(upper_mask) else 0
        
        print(f"하단부 +z 법선 수: {lower_positive_z}, -z 법선 수: {lower_negative_z}")
        print(f"상단부 +z 법선 수: {upper_positive_z}, -z 법선 수: {upper_negative_z}")
        
        # z축 방향 법선이 많은 쪽이 바닥일 가능성 높음
        lower_z_alignment = max(lower_positive_z, lower_negative_z) / (len(lower_region) + 1e-6)
        upper_z_alignment = max(upper_positive_z, upper_negative_z) / (len(upper_region) + 1e-6)
        
        print(f"하단부 z축 정렬도: {lower_z_alignment:.4f}")
        print(f"상단부 z축 정렬도: {upper_z_alignment:.4f}")
        
        # 최종 판단: 바닥 점수와 z축 정렬도를 모두 고려
        # 이 경우는 결과를 반전시킵니다 (현재 구현이 틀리다는 피드백에 기반)
        should_flip = not (lower_base_score > upper_base_score or lower_z_alignment > upper_z_alignment * 1.5)
        
        print(f"뒤집기 판단: {should_flip}")
        
        return should_flip

    def calculate_normal_diversity(self, normals):
        """법선 벡터의 다양성을 계산합니다 (값이 낮을수록 일관됨)"""
        if len(normals) < 3:
            return float('inf')
        
        # 법선 벡터의 공분산 행렬 계산
        cov_matrix = np.cov(normals, rowvar=False)
        
        # 고유값 계산
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        
        # 고유값의 합을 다양성 지표로 사용 (값이 높을수록 다양함)
        return np.sum(eigenvalues)

    def analyze_height_distribution(self, heights):
        """높이 분포의 평탄도를 분석합니다 (값이 높을수록 평탄함)"""
        if len(heights) < 3:
            return 0
        
        # 높이의 사분위수 계산
        q1 = np.percentile(heights, 25)
        q2 = np.percentile(heights, 50)
        q3 = np.percentile(heights, 75)
        
        # 사분위 범위
        iqr = q3 - q1
        
        # 평균값 주변에 데이터가 얼마나 집중되어 있는지 계산
        if iqr > 0:
            concentration = 1.0 - (iqr / (np.max(heights) - np.min(heights)))
            return concentration
        
        return 1.0  # 모든 높이가 같으면 완벽히 평탄

    def orient_mesh_z_axis(self, mesh):
        """
        메시를 z축 방향으로 올바르게 정렬합니다.
        평평한 부분이 아래로 향하도록 합니다.
        
        Args:
            mesh: 정렬할 메시
            
        Returns:
            pv.PolyData: 방향이 조정된 메시
        """
        mesh_points = np.array(mesh.points)
        
        # faces 정보 추출
        faces = []
        face_count = 0
        face_data = mesh.faces
        i = 0
        while i < len(face_data):
            n_vertices = face_data[i]
            faces.append(face_data[i+1:i+1+n_vertices])
            i += n_vertices + 1
        
        # z축 방향 결정
        should_flip = self.determine_flat_side(mesh_points, faces)
        
        if should_flip:
            print("메시를 z축 방향으로 뒤집습니다.")
            # z축 방향으로 뒤집기 (180도 회전)
            rotation_matrix = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ])
            
            flipped_mesh = mesh.copy()
            flipped_mesh.transform(rotation_matrix)
            return flipped_mesh, rotation_matrix
        else:
            print("메시의 z축 방향이 이미 올바릅니다.")
            return mesh, np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

    def align_and_orient_mesh(self):
        """
        메시를 좌표축에 정렬하고 올바른 방향으로 조정합니다.
        1. OBB 축을 좌표축에 정렬
        2. z축 방향 조정 (평평한 부분이 아래로)
        
        Returns:
            pv.PolyData: 정렬되고 방향이 조정된 메시
            np.ndarray: 결합된 변환 행렬
        """
        # 1. 축 정렬
        aligned_mesh, transform_matrix = self.align_mesh_to_axes()
        
        # 2. z축 방향 조정
        oriented_mesh, rotation_matrix_z = self.orient_mesh_z_axis(aligned_mesh)
        
        # 변환 행렬 결합
        combined_matrix = np.matmul(rotation_matrix_z, transform_matrix)
        
        return oriented_mesh, combined_matrix

    def visualize_oriented_mesh(self):
        """정렬되고 방향이 조정된 메시를 시각화합니다"""
        # 정렬 및 방향 조정
        oriented_mesh, combined_matrix = self.align_and_orient_mesh()
        
        # 플로터 생성
        plotter = pv.Plotter()
        
        # 정렬된 메시 추가
        plotter.add_mesh(oriented_mesh, color='lightblue', opacity=0.9, show_edges=True)
        
        # 좌표축 시각화
        plotter.add_axes()
        
        # 바닥 평면 추가 (시각적 참조)
        min_bounds = np.min(oriented_mesh.points, axis=0)
        max_bounds = np.max(oriented_mesh.points, axis=0)
        extent = max_bounds - min_bounds
        
        center = (min_bounds + max_bounds) / 2
        plane = pv.Plane(
            center=(center[0], center[1], min_bounds[2] - 0.05 * extent[2]),
            direction=(0, 0, 1),
            i_size=1.2 * extent[0],
            j_size=1.2 * extent[1]
        )
        plotter.add_mesh(plane, color='gray', opacity=0.5)
        
        # 제목 설정
        plotter.add_title("Oriented Mesh (Z-axis Aligned)", font_size=16)
        
        # 시각화
        plotter.show()
        
        # 자원 해제
        plotter.close()

    def find_molars_and_align_to_y(self, mesh):
        """
        어금니 두 개를 찾고, 그 방향이 -y를 향하도록 정렬합니다.
        이미 OBB로 정렬된 상태에서는 0도 또는 180도 회전만 필요합니다.
        
        Args:
            mesh: 정렬할 메시
            
        Returns:
            pv.PolyData: 전치부가 +y 방향으로 정렬된 메시
        """
        # 메시의 점 좌표 얻기
        points = np.array(mesh.points)
        
        # xy 평면에 투영된 점들
        projected_points = points[:, :2]  # xy 좌표만 추출
        
        # 중심점 계산
        center = np.mean(projected_points, axis=0)
        
        # 볼록 껍질로 아치의 바깥쪽 경계 찾기
        from scipy.spatial import ConvexHull
        hull = ConvexHull(projected_points)
        boundary_points = projected_points[hull.vertices]
        
        # 각 경계점에서 중심까지의 거리 계산
        distances = np.linalg.norm(boundary_points - center, axis=1)
        
        # 거리가 긴 상위 30% 점들 중에서 후보 선택
        distance_threshold = np.percentile(distances, 70)
        distant_mask = distances >= distance_threshold
        distant_points = boundary_points[distant_mask]
        
        # x값 기준으로 정렬하여 좌우 어금니 찾기
        x_sorted_idx = np.argsort(distant_points[:, 0])
        left_molar_candidates = distant_points[x_sorted_idx[:len(distant_points)//3]]
        right_molar_candidates = distant_points[x_sorted_idx[-len(distant_points)//3:]]
        
        # 각 후보군에서 y값이 제일 작은 점을 선택 (어금니는 일반적으로 아치의 뒤쪽에 위치)
        left_molar = left_molar_candidates[np.argmin(left_molar_candidates[:, 1])]
        right_molar = right_molar_candidates[np.argmin(right_molar_candidates[:, 1])]
        
        # 두 어금니의 중간 지점
        molar_midpoint = (left_molar + right_molar) / 2
        
        # 어금니 중간점과 중심점을 이은 벡터의 방향
        molar_direction = molar_midpoint - center
        molar_direction = molar_direction / np.linalg.norm(molar_direction)
        
        # 어금니 방향과 -y 방향의 내적
        neg_y_direction = np.array([0, -1])
        alignment = np.dot(molar_direction, neg_y_direction)
        
        print(f"어금니 방향과 -y축의 정렬도: {alignment}")
        
        # 내적이 양수면 방향이 비슷, 음수면 반대
        need_rotation = alignment < 0
        
        if need_rotation:
            print("어금니가 +y 방향을 향하고 있어 180도 회전 필요")
            # 180도 회전 변환 행렬
            rotation_matrix = np.array([
                [-1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            
            # 메시 회전
            aligned_mesh = mesh.copy()
            aligned_mesh.transform(rotation_matrix, inplace=True)
        else:
            print("어금니가 이미 -y 방향을 향하고 있어 회전 불필요")
            aligned_mesh = mesh.copy()
        
        # 어금니와 중심점 좌표 3D로 변환 (시각화용)
        avg_z = np.mean(points[:, 2])
        left_molar_3d = np.append(left_molar, avg_z)
        right_molar_3d = np.append(right_molar, avg_z)
        center_3d = np.append(center, avg_z)
        
        return aligned_mesh, need_rotation, left_molar_3d, right_molar_3d, center_3d

    def visualize_molar_alignment(self):
        """어금니 위치에 기반하여 전치부를 +y 방향으로 정렬한 메시를 시각화합니다"""
        # 먼저 메시를 축에 정렬하고 z축 방향 조정
        oriented_mesh, combined_matrix = self.align_and_orient_mesh()
        
        # 어금니 찾기 및 정렬
        result = self.find_molars_and_align_to_y(oriented_mesh)
        aligned_mesh, was_rotated, left_molar, right_molar, center = result
        
        # y축 회전이 적용된 경우 변환 행렬 업데이트
        if was_rotated:
            # 180도 회전 변환 행렬
            y_rotation_matrix = np.array([
                [-1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            combined_matrix = np.matmul(y_rotation_matrix, combined_matrix)
        
        # 플로터 생성
        plotter = pv.Plotter()
        
        # 정렬된 메시 추가
        plotter.add_mesh(aligned_mesh, color='lightblue', opacity=0.8, show_edges=True)
        
        # 좌표축 시각화
        plotter.add_axes()
        
        # 중심점 표시
        plotter.add_mesh(pv.Sphere(center=center, radius=np.max(aligned_mesh.bounds)/50), 
                        color='green', render_points_as_spheres=True, label='Center')
        
        # 어금니 위치 표시
        plotter.add_mesh(pv.Sphere(center=left_molar, radius=np.max(aligned_mesh.bounds)/50), 
                        color='red', render_points_as_spheres=True, label='Left Molar')
        plotter.add_mesh(pv.Sphere(center=right_molar, radius=np.max(aligned_mesh.bounds)/50), 
                        color='orange', render_points_as_spheres=True, label='Right Molar')
        
        # 정렬 방향 화살표 추가
        arrow_length = np.max(aligned_mesh.bounds) * 0.3
        arrow_start = center
        arrow_direction = np.array([0, 1, 0])  # +y 방향 (전치부 방향)
        arrow_end = arrow_start + arrow_direction * arrow_length
        
        # 화살표 생성
        arrow = pv.Line(arrow_start, arrow_end)
        plotter.add_mesh(arrow, color='blue', line_width=5, label='+Y (Anterior)')
        
        # 제목 설정
        rotation_text = "with 180° Rotation" if was_rotated else "without Rotation"
        plotter.add_title(f"Dental Model Aligned to +Y {rotation_text}", font_size=16)
        
        # 범례 추가
        plotter.add_legend()
        
        # 시각화
        plotter.show()
        
        # 자원을 명시적으로 해제
        plotter.close()
        
        # 결과 반환 (깊은 복사를 통해 참조 문제 방지)
        result_mesh = aligned_mesh.copy(deep=True)
        
        return result_mesh, combined_matrix, left_molar, right_molar, center


if __name__ == "__main__":
    try:
        # ios_laminate_registration = IosAnalysis("../../example/data/ios_with_smilearch.stl")
        ios_laminate_registration = IosAlignment("../../example/data/ios_with_smilearch_2.stl")
        
        # 전체 변환 행렬을 얻기 위해 run_analysis 실행
        aligned_mesh, transform_matrix = ios_laminate_registration.run_analysis()
        
        # 변환 행렬 출력
        print("========== run_analysis 전체 변환 행렬 ==========")
        print(transform_matrix)
        print("=================================================")
        
        # 메시 시각화 순서대로 실행
        ios_laminate_registration.visualize_mesh_features()
        ios_laminate_registration.visualize_aligned_mesh()
        
        # visualize_molar_alignment 호출 및 결과 사용
        final_mesh, final_transform, left_molar, right_molar, center = ios_laminate_registration.visualize_molar_alignment()
        
        print("\n========== 최종 전체 변환 행렬 ==========")
        print(final_transform)
        print("=========================================")
        
        print("\n원본 메시에 변환 행렬 적용 결과는 final_mesh와 동일해야 합니다.")
        
        # 메모리에서 명시적으로 해제
        del final_mesh
        del aligned_mesh
        
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        # 명시적으로 가비지 컬렉션 호출 (선택사항)
        import gc
        gc.collect()
    