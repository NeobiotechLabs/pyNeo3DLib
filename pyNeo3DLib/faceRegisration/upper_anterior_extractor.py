"""
상악전치부 메쉬 추출 모듈

이 모듈은 입술 메쉬에서 상악전치부 메쉬를 추출하는 기능을 담당합니다.
단일 책임 원칙(SRP)에 따라 메쉬 추출 로직만을 캡슐화합니다.
"""
from dataclasses import dataclass, field
from typing import Tuple, Optional
import numpy as np
import pyvista as pv

from pyNeo3DLib.fileLoader.mesh import Mesh
from pyNeo3DLib.faceRegisration.rotationAxisCalculator import RotationAxisCalculator
from pyNeo3DLib.faceRegisration.ios_local_coordinate_builder import IOSLocalCoordinateSystemBuilder
from pyNeo3DLib.faceRegisration.ray_caster import RayCaster
from pyNeo3DLib.faceRegisration.constants import IncisorAlignmentConstants

@dataclass
class UpperAnteriorExtractionResult:
    """
    상악전치부 추출 결과를 담는 데이터 클래스.
    
    Attributes:
        upper_anterior_mesh: 추출된 상악전치부 Mesh 객체
    """
    upper_anterior_mesh: Mesh


class UpperAnteriorExtractor:
    """
    입술 메쉬에서 상악전치부 메쉬를 추출하는 클래스.
    
    단일 책임: 상악전치부 메쉬 추출 파이프라인 실행
    
    이 클래스는 다음 단계를 수행합니다:
    1. 주축 및 중심점 계산
    2. Mesh를 PyVista PolyData로 변환
    3. 좌표계 벡터 찾기
    4. 노이즈 제거 및 경계 탐지
    5. 양방향 레이캐스팅으로 교차점 찾기
    6. 상악전치부 메쉬 슬라이싱
    """
    
    def __init__(self, noise_angle_threshold: float = 70.0, 
                 boundary_scan_span: float = 3.0,
                 boundary_num_samples: int = 100):
        """
        Args:
            noise_angle_threshold: 노이즈 제거를 위한 각도 임계값 (기본값: 70도)
            boundary_scan_span: 경계 탐지를 위한 스캔 범위 (기본값: 3)
            boundary_num_samples: 경계 탐지를 위한 샘플링 수 (기본값: 100)
        """
        self._noise_angle_threshold = noise_angle_threshold
        self._boundary_scan_span = boundary_scan_span
        self._boundary_num_samples = boundary_num_samples
        self._raycaster = RayCaster()
        self._rotation_axis_calculator = RotationAxisCalculator()
        self._coordinate_system_builder = IOSLocalCoordinateSystemBuilder()
    
    def extract(self, lip_mesh: Mesh) -> UpperAnteriorExtractionResult:
        """
        입술 메쉬에서 상악전치부 메쉬를 추출합니다.
        
        Args:
            lip_mesh: 입술 영역 Mesh 객체
            
        Returns:
            UpperAnteriorExtractionResult: 추출 결과 (상악전치부 메쉬, 방향 벡터들, 경계점 등)
        """

        

        # Mesh를 PyVista PolyData로 변환
        pv_lip_mesh = self._convert_mesh_to_pyvista(lip_mesh)
        
        # 노말벡터 기반 노이즈 제거
        cleaned_mesh = self._remove_noise_by_normal_angle(
            pv_mesh=pv_lip_mesh,
            reference_vector=np.array([0, 0, 1]),
            angle_threshold_degrees=self._noise_angle_threshold
        )

        clipped_mesh = self._clip_mesh_by_axis_range(
            mesh=cleaned_mesh,
            axis=0,  # x축
            min_value=IncisorAlignmentConstants.X_AXIS_CLIP_MIN,
            max_value=IncisorAlignmentConstants.X_AXIS_CLIP_MAX,
            extract_largest=False
        )

        # 분리된 메쉬 덩어리 중 z값이 가장 높은 도심점을 가진 메쉬만 추출
        clipped_mesh = self._extract_body_with_highest_z_centroid(clipped_mesh)

        #  중절치 중심점 찾기
        incisor_center_point = self._find_central_incisor_center_point(
            mesh=clipped_mesh
        )


        upper_anterior_mesh = self._create_slicing_mesh(
            pv_mesh=cleaned_mesh,
            origin=incisor_center_point,
            normal=np.array([0, -1, 0])
        )


        
        return UpperAnteriorExtractionResult(
            upper_anterior_mesh=upper_anterior_mesh,
        )
    
    def _compute_principal_axes_and_centroid(self, mesh: Mesh) -> Tuple[np.ndarray, np.ndarray]:
        """
        메쉬의 주축(Principal Axes)과 중심점을 계산합니다.
        
        Args:
            mesh: 분석할 Mesh 객체
            
        Returns:
            tuple: (principal_axes, centroid)
                - principal_axes: 주축 벡터들 (shape (3, 3))
                - centroid: 중심점 (shape (3,))
        """
        result = self._rotation_axis_calculator.compute_principal_axes(mesh.vertices)
        print("principal_axes shape: ", result.principal_axes.shape)
        return result.principal_axes, result.centroid
    
    def _convert_mesh_to_pyvista(self, mesh) -> pv.PolyData:
        """
        Mesh 객체를 PyVista PolyData로 변환합니다.
        
        Args:
            mesh: 변환할 Mesh 객체 또는 PyVista PolyData
            
        Returns:
            pv.PolyData: PyVista PolyData 객체
        """
        if isinstance(mesh, pv.PolyData):
            return mesh
        
        faces_pv = np.hstack([[3, *face] for face in mesh.faces])
        return pv.PolyData(mesh.vertices, faces_pv)
    
    def _find_coordinate_system_vectors(self, pv_mesh: pv.PolyData, centroid: np.ndarray, 
                                         principal_axes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        메쉬와 주축을 기반으로 좌표계 벡터(zero/one intersection vector)를 찾습니다.
        
        Args:
            pv_mesh: PyVista PolyData 메쉬
            centroid: 중심점
            principal_axes: 주축 벡터들 (shape (3, 3))
            
        Returns:
            tuple: (zero_intersection_vector, one_intersection_vector)
        """
        zero_intersection_vector, one_intersection_vector = self._coordinate_system_builder._find_zero_and_one_intersection_vector(
            pv_mesh=pv_mesh,
            centroid=centroid,
            axis_vectors=[principal_axes[0], principal_axes[1], principal_axes[2]]
        )
        return zero_intersection_vector, one_intersection_vector
    
    def _clean_mesh_and_find_boundary(self, pv_mesh: pv.PolyData, centroid: np.ndarray,
                                       zero_intersection_vector: np.ndarray,
                                       one_intersection_vector: np.ndarray) -> Tuple[pv.PolyData, np.ndarray, np.ndarray]:
        """
        노말벡터 기반으로 노이즈를 제거하고, 상악/하악 경계를 탐지합니다.
        
        Args:
            pv_mesh: PyVista PolyData 메쉬
            centroid: 중심점
            zero_intersection_vector: 스캔 방향 벡터
            one_intersection_vector: 레이 방향 벡터
            
        Returns:
            tuple: (cleaned_mesh, upper_jaw_direction, boundary_point)
        """
        # 노말벡터 기반 노이즈 제거
        cleaned_mesh = self._remove_noise_by_normal_angle(
            pv_mesh=pv_mesh,
            reference_vector=one_intersection_vector,
            angle_threshold_degrees=self._noise_angle_threshold
        )
        
        # 레이캐스팅으로 상악/하악 경계 탐지
        distances_array, sample_points_array = self._find_upper_lower_boundary_by_raycasting(
            pv_mesh=cleaned_mesh,
            centroid=centroid,
            scan_direction_vector=zero_intersection_vector,
            ray_direction_vector=one_intersection_vector
        )
        
        # 상악/하악 경계점 결정
        boundary_point = self._determine_upper_lower_boundary_point(
            distances_array=distances_array,
            sample_points_array=sample_points_array
        )

        # upper_jaw_direction 결정
        reference_direction = np.array([0, -1, 0])
        inner_product = np.dot(zero_intersection_vector, reference_direction)
        if inner_product > 0:
            upper_jaw_direction = zero_intersection_vector
        else:
            upper_jaw_direction = -zero_intersection_vector
        
        return cleaned_mesh, upper_jaw_direction, boundary_point
    
    def _remove_noise_by_normal_angle(self, pv_mesh: pv.PolyData, reference_vector: np.ndarray,
                                       angle_threshold_degrees: float = 70) -> pv.PolyData:
        """
        노말벡터와 기준벡터 간의 각도를 기반으로 노이즈를 제거합니다.
        """
        normal_vectors = np.asarray(pv_mesh.point_normals)
        normal_vectors_dot_product = np.dot(normal_vectors, reference_vector)
        normal_vectors_angle = np.arccos(np.abs(normal_vectors_dot_product))
        normal_vectors_angle_mask = normal_vectors_angle > (angle_threshold_degrees * np.pi / 180)
        indices_to_remove = np.where(normal_vectors_angle_mask)[0]
        
        cleaned_mesh, _ = pv_mesh.remove_points(indices_to_remove)
        cleaned_mesh = cleaned_mesh.connectivity(extraction_mode='largest')
        
        return cleaned_mesh
    
    def _find_upper_lower_boundary_by_raycasting(self, pv_mesh: pv.PolyData, centroid: np.ndarray,
                                                  scan_direction_vector: np.ndarray,
                                                  ray_direction_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        레이캐스팅을 통해 상악과 하악의 경계를 찾습니다.
        """
        raycast_start_point = centroid.reshape(1, 3) + self._boundary_scan_span * scan_direction_vector.reshape(1, 3)
        raycast_end_point = centroid.reshape(1, 3) - self._boundary_scan_span * scan_direction_vector.reshape(1, 3)
        
        linspace_points = np.linspace(raycast_start_point, raycast_end_point, self._boundary_num_samples)
        
        distances_array = []
        sample_points_array = []
        
        for point in linspace_points:
            intersections_plus = self._raycaster.ray_casting(pv_mesh, point.reshape(1, 3), ray_direction_vector.reshape(1, 3))
            intersections_minus = self._raycaster.ray_casting(pv_mesh, point.reshape(1, 3), -ray_direction_vector.reshape(1, 3))
            intersections = np.concatenate([intersections_plus, intersections_minus])
            
            if len(intersections) > 0:
                distances = np.linalg.norm(intersections - point.reshape(1, 3), axis=1)
                min_distance = np.min(distances)
                distances_array.append(min_distance)
                sample_points_array.append(point.flatten())
        
        return np.array(distances_array), np.array(sample_points_array)
    
    def _determine_upper_lower_boundary_point(self, distances_array: np.ndarray,
                                               sample_points_array: np.ndarray) -> np.ndarray:
        """
        거리 변화량을 분석하여 경계점 3D 좌표를 결정합니다.
        """
        distance_changes = np.diff(distances_array)
        max_distance_change_index = np.argmax(distance_changes)
        max_distance_change_point = sample_points_array[max_distance_change_index]
        return max_distance_change_point
    
    def _find_closest_intersection_bidirectional(self, mesh, centroid: np.ndarray,
                                                  direction_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        양방향 레이캐스팅을 수행하여 가장 가까운 교차점과 방향 벡터를 찾습니다.
        """
        pv_mesh = self._convert_mesh_to_pyvista(mesh)
        
        intersections_plus = self._raycaster.ray_casting(
            pv_mesh, centroid.reshape(1, 3), direction_vector.reshape(1, 3)
        )
        intersections_minus = self._raycaster.ray_casting(
            pv_mesh, centroid.reshape(1, 3), -direction_vector.reshape(1, 3)
        )
        
        bidirectional_intersections = np.concatenate([intersections_plus, intersections_minus], axis=0)
        print("bidirectional_intersections: ", bidirectional_intersections)
        
        if len(bidirectional_intersections) > 0:
            distances = np.linalg.norm(bidirectional_intersections - centroid, axis=1)
            closest_idx = np.argmin(distances)
            intersection_point = bidirectional_intersections[closest_idx]
            
            intersection_vector = (intersection_point - centroid).reshape(1, 3)
            intersection_vector = intersection_vector / np.linalg.norm(intersection_vector)
        else:
            print("Warning: No intersection found in bidirectional raycasting")
            intersection_vector = direction_vector.reshape(1, 3)
            intersection_point = centroid
            
        return intersection_vector, intersection_point
    
    def _pyvista_to_mesh(self, pv_mesh: pv.PolyData) -> Mesh:
        """
        PyVista PolyData를 Mesh 객체로 변환합니다.
        
        Args:
            pv_mesh: PyVista PolyData 객체
            
        Returns:
            Mesh: 변환된 Mesh 객체
        """
        mesh = Mesh()
        mesh.vertices = np.array(pv_mesh.points)
        
        faces_pv = pv_mesh.faces.reshape(-1, 4)[:, 1:4]
        mesh.faces = np.array(faces_pv)
        
        pv_mesh.compute_normals(inplace=True)
        if pv_mesh.point_normals is not None:
            mesh.normals = np.array(pv_mesh.point_normals)
        
        return mesh
    
    def _extract_body_with_highest_z_centroid(self, mesh: Mesh) -> Mesh:
        """
        메쉬가 여러 분리된 덩어리로 구성된 경우, 도심점의 z값이 가장 큰 덩어리만 추출합니다.
        
        메쉬를 PyVista PolyData로 변환한 후 split_bodies()를 통해 분리된 덩어리들을 확인하고,
        각 덩어리의 도심점(centroid) z값을 비교하여 가장 큰 값을 가진 메쉬만 반환합니다.
        
        Args:
            mesh: 처리할 Mesh 객체
            
        Returns:
            Mesh: z값이 가장 높은 도심점을 가진 메쉬 덩어리 (단일 덩어리일 경우 그대로 반환)
        """
        # Mesh를 pv.PolyData로 변환
        pv_faces = np.hstack([
            np.full((len(mesh.faces), 1), 3),
            mesh.faces
        ]).flatten().astype(np.int32)
        pv_mesh = pv.PolyData(mesh.vertices, pv_faces)

        # 메쉬가 여러 덩어리로 분리되어 있는지 확인
        bodies = pv_mesh.split_bodies()
        if len(bodies) > 1:
            max_centroid_z = -np.inf
            max_mesh = None
            for body in bodies:
                centroid = body.center
                if centroid[2] > max_centroid_z:
                    max_centroid_z = centroid[2]
                    max_mesh = body
            # UnstructuredGrid를 PolyData로 변환
            pv_mesh = max_mesh.extract_surface()

        # pv.PolyData를 Mesh로 다시 변환
        result_mesh = Mesh()
        result_mesh.vertices = np.array(pv_mesh.points)
        result_mesh.faces = np.array(pv_mesh.faces.reshape(-1, 4)[:, 1:4])
        
        return result_mesh
    
    def _create_slicing_mesh(self, pv_mesh: pv.PolyData, origin: np.ndarray, 
                              normal: np.ndarray) -> Mesh:
        """
        시작점과 법선벡터가 주어지면 슬라이싱 면을 만들어서 메쉬를 잘라서
        법선벡터 방향에 있는 메쉬를 반환합니다.
        """
        normal = np.array(normal).flatten()
        normal = normal / np.linalg.norm(normal)
        origin = np.array(origin).flatten()
        
        clipped_pv_mesh = pv_mesh.clip(normal=normal, origin=origin, invert=False)
        clipped_pv_mesh = clipped_pv_mesh.connectivity(extraction_mode='largest')
        
        clipped_mesh = Mesh()
        clipped_mesh.vertices = np.array(clipped_pv_mesh.points)
        
        faces_pv = clipped_pv_mesh.faces.reshape(-1, 4)[:, 1:4]
        clipped_mesh.faces = np.array(faces_pv)
        
        clipped_pv_mesh.compute_normals(inplace=True)
        if clipped_pv_mesh.point_normals is not None:
            clipped_mesh.normals = np.array(clipped_pv_mesh.point_normals)
        
        return clipped_mesh



    def _find_central_incisor_center_point(
        self,
        mesh: Mesh,
    ) -> np.ndarray:
        """
        입술 메쉬에서 중절치 중심점을 추정합니다.
        
        메쉬에서 y값이 가장 큰 정점을 찾고, x값을 0으로 설정하여
        중절치 중심점을 추정합니다.
        """

        vertices = mesh.vertices
        
        # y값이 가장 큰 정점 찾기 (전치부 가장 아래쪽)
        max_y_idx = np.argmax(vertices[:, 1])
        center_point = vertices[max_y_idx].copy()
    
        # x값을 0으로 설정 (중앙 정렬)
        center_point[0] = 0
        
        return center_point

    def _clip_mesh_by_axis_range(
        self,
        mesh: Mesh,
        axis: int,
        min_value: float,
        max_value: float,
        extract_largest: bool = True
    ) -> Mesh:
        """
        메쉬를 특정 축 기준으로 지정된 범위만 남기고 클립합니다.
        
        Args:
            mesh: 클립할 Mesh 객체
            axis: 클립할 축 (0=x, 1=y, 2=z)
            min_value: 유지할 최소값
            max_value: 유지할 최대값
            extract_largest: True이면 가장 큰 연결된 덩어리만 반환
            
        Returns:
            Mesh: 클립된 Mesh 객체
        """
        # 입력이 pv.PolyData인지 Mesh인지 확인
        if isinstance(mesh, pv.PolyData):
            pv_mesh = mesh
        else:
            # Mesh 객체를 PyVista PolyData로 변환
            vertices = mesh.vertices
            faces = mesh.faces
            
            # PyVista용 faces 배열 생성 (각 면 앞에 vertex 수 추가)
            pv_faces = np.hstack([
                np.full((len(faces), 1), 3),  # 각 면은 3개의 vertex를 가짐
                faces
            ]).flatten().astype(np.int32)
            
            pv_mesh = pv.PolyData(vertices, pv_faces)
        
        # 축에 따른 법선 벡터 설정
        normal_positive = [0, 0, 0]
        normal_negative = [0, 0, 0]
        normal_positive[axis] = 1
        normal_negative[axis] = -1
        
        # max_value 평면으로 클립 (axis > max_value 부분 제거)
        origin_max = [0, 0, 0]
        origin_max[axis] = max_value
        clipped = pv_mesh.clip(normal=normal_positive, origin=origin_max, invert=True)
        
        # min_value 평면으로 클립 (axis < min_value 부분 제거)
        origin_min = [0, 0, 0]
        origin_min[axis] = min_value
        clipped = clipped.clip(normal=normal_negative, origin=origin_min, invert=True)
        
        # 가장 큰 연결된 덩어리만 추출
        if extract_largest:
            clipped = clipped.connectivity(extraction_mode='largest')
        
        # PyVista 메쉬를 Mesh 객체로 변환
        result_mesh = Mesh()
        result_mesh.vertices = np.array(clipped.points)
        
        # faces 변환 (PyVista는 [n, v0, v1, v2, ...] 형식, Mesh는 [v0, v1, v2] 형식)
        if len(clipped.faces) > 0:
            faces_pv = clipped.faces.reshape(-1, 4)[:, 1:4]
            result_mesh.faces = np.array(faces_pv)
        else:
            result_mesh.faces = np.array([]).reshape(0, 3)
        
        # 노말 계산
        clipped.compute_normals(inplace=True)
        if clipped.point_normals is not None:
            result_mesh.normals = np.array(clipped.point_normals)
        
        return result_mesh

