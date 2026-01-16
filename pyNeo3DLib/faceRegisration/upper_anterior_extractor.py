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
from pyNeo3DLib.faceRegisration.constants import IncisorAlignmentConstants
from pyNeo3DLib.faceRegisration.mesh_cleaner import MeshCleaner

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
        
        # 노말벡터 기반 노이즈 제거(기준이되는 벡터와 노말벡터의 각도가 임계값 이상이면 제거, 페이스스캔의 치아메쉬에는 치아 주변에 노이즈가 있기 때문에 제거해야함)
        cleaned_mesh = self._remove_noise_by_normal_angle(
            pv_mesh=pv_lip_mesh,
            reference_vector=np.array([0, 0, 1]),  #기준이되는 벡터
            angle_threshold_degrees=self._noise_angle_threshold
        )

        # 상악 중절치 중심점 찾기 위해 x축 범위 기준으로 자르기(중절치 부근 메쉬만 나오도록 자름)
        mesh_cleaner = MeshCleaner()
        clipped_mesh = mesh_cleaner.clip_mesh_by_axis_range(
            mesh=cleaned_mesh,
            axis=0,  # x축
            min_value=IncisorAlignmentConstants.X_AXIS_CLIP_MIN,
            max_value=IncisorAlignmentConstants.X_AXIS_CLIP_MAX,
            extract_largest=False
        )

        # 분리된 메쉬 덩어리 중 z값이 가장 높은 도심점을 가진 메쉬만 추출(상악 전치부가 하악 전치부보다 앞으로 튀어나와있기 때문에 가장 앞쪽에 있는 덩어리가 상악전치부)
        clipped_mesh = self._extract_body_with_highest_z_centroid(clipped_mesh)

        #  중절치 중심점 찾기
        incisor_center_point = self._find_central_incisor_center_point_for_extraction_upper_anterior(
            mesh=clipped_mesh
        )

        # 중절치 중심점 으로부터 법선벡터 방향으로 슬라이싱 면을 만들어서 메쉬를 잘라서 법선벡터 방향에 있는 메쉬를 반환합니다.
        upper_anterior_mesh = self._create_slicing_mesh(
            pv_mesh=cleaned_mesh,
            origin=incisor_center_point,
            normal=np.array([0, -1, 0])  # 슬라이싱 면의 법선벡터
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

        # 메쉬가 여러 덩어리로 분리되어 있는지 확인(상하악 동시에 있는 메쉬면 상악전치부, 하악전치부가 나올텐데 그걸 제거하고 상악전치부만 나오도록 자름)
        bodies = pv_mesh.split_bodies()
        if len(bodies) > 1:
            max_centroid_z = -np.inf
            max_mesh = None
            for body in bodies:
                centroid = body.center
                # z값이 가장 높은 도심점을 가진 덩어리만 추출(상악 전치부가 하악 전치부보다 앞으로 튀어나와있기 때문에 가장 앞쪽에 있는 덩어리가 상악전치부)
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



    def _find_central_incisor_center_point_for_extraction_upper_anterior(
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
