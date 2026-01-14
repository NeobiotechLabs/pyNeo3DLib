"""
페이스 라미네이트 정합 모듈

이 모듈은 페이스 스캔과 라미네이트 메쉬 간의 정합을 오케스트레이션합니다.
단일 책임 원칙(SRP)에 따라 전체 워크플로우 조정만을 담당합니다.
"""
import numpy as np
import copy

from pyNeo3DLib.fileLoader.mesh import Mesh
from pyNeo3DLib.visualization.neovis import visualize_meshes

# 분리된 모듈들 import
from pyNeo3DLib.faceRegisration.mesh_transformer import MeshTransformer
from pyNeo3DLib.faceRegisration.mesh_converter import MeshConverter
from pyNeo3DLib.faceRegisration.mesh_cleaner import MeshCleaner
from pyNeo3DLib.faceRegisration.icp_registrator import ICPRegistrator
from pyNeo3DLib.faceRegisration.texture_mesh_extractor import TextureMeshExtractor
from pyNeo3DLib.faceRegisration.incisor_aligner import IncisorAligner
from pyNeo3DLib.faceRegisration.mouthEraserForface import MouthEraserForFace
from pyNeo3DLib.faceRegisration.upper_anterior_extractor import UpperAnteriorExtractor


class FaceLaminateRegistration:
    """
    페이스 스캔과 라미네이트 메쉬 간의 정합을 수행하는 클래스.
    
    단일 책임: 정합 워크플로우 오케스트레이션
    
    이 클래스는 다음 컴포넌트들을 조합하여 정합을 수행합니다:
    - MeshTransformer: 메쉬 변환
    - ICPRegistrator: ICP 정합
    - TextureMeshExtractor: 텍스처 기반 메쉬 추출
    - MeshCleaner: 메쉬 클리닝
    - IncisorAligner: 중절치 정렬
    - UpperAnteriorExtractor: 상악전치부 추출
    """
    
    def __init__(self, face_path: str, laminate_path: str, visualization: bool = False):
        """
        FaceLaminateRegistration 초기화
        
        Args:
            face_path: 페이스 스캔 파일 경로
            laminate_path: 라미네이트 메쉬 파일 경로
            visualization: 시각화 활성화 여부
        """
        self.face_smile_path = face_path
        self.laminate_path = laminate_path
        self.visualization = visualization
        
        # 변환 행렬 초기화
        self.transform_matrix = np.eye(4)
        
        # 컴포넌트 초기화
        self._mesh_transformer = MeshTransformer()
        self._mesh_cleaner = MeshCleaner()
        self._icp_registrator = ICPRegistrator(visualization=visualization)
        self._texture_extractor = TextureMeshExtractor()
        self._incisor_aligner = IncisorAligner()
        self._upper_anterior_extractor = UpperAnteriorExtractor()
        
        # 메쉬 로드
        self._load_models()
    
    def _load_models(self):
        """메쉬 파일들을 로드합니다."""
        self.face_smile_mesh = Mesh.from_file(self.face_smile_path)
        self.laminate_mesh = Mesh.from_file(self.laminate_path)
    
    def run_registration(self):
        """
        전체 정합 프로세스를 실행합니다.
        
        Returns:
            tuple: (최종 변환 행렬, 변환된 페이스 메쉬)
        """
        # 1. 초기 시각화
        if self.visualization:
            visualize_meshes(
                [self.face_smile_mesh, self.laminate_mesh], 
                ["Face", "Laminate"], 
                title="Initial Meshes"
            )
        
        # 2. Y축 정렬 (Z축 중심 180도 회전)
        self._align_y_axis()
        if self.visualization:
            visualize_meshes(
                [self.face_smile_mesh, self.laminate_mesh], 
                ["Face", "Laminate"], 
                title="After Y-axis Alignment"
            )
        print("Y-axis alignment transformation matrix:")
        print(self.transform_matrix)
        
        # 3. 입술 영역 메쉬 추출
        lip_mesh = self._extract_lip_mesh()
        if lip_mesh is None:
            return None, None
        
        if self.visualization:
            visualize_meshes(
                [lip_mesh, self.face_smile_mesh], 
                ["Lip", "FaceScan"], 
                title="Lip Mesh Extracted"
            )
        
        # 4. 상악전치부 추출 및 좌표계 계산
        lip_mesh, local_coordinate_system = self._extract_upper_anterior(lip_mesh)
        
        if self.visualization:
            visualize_meshes(
                [lip_mesh, self.laminate_mesh], 
                ["Lip", "Laminate"], 
                title="Upper Anterior with Coordinate System"
            )
        
        # 5. 글로벌 좌표계로 변환
        lip_mesh, rotated_face_mesh = self._transform_to_global(lip_mesh, local_coordinate_system)
        
        # 6. 첫 번째 중절치 정렬
        lip_mesh, rotated_face_mesh = self._apply_incisor_alignment(
            lip_mesh, rotated_face_mesh
        )
        
        if self.visualization:
            visualize_meshes(
                [lip_mesh, rotated_face_mesh, self.laminate_mesh], 
                ["Lip", "Rotated FaceScan", "Laminate"], 
                title="After Incisor Alignment"
            )
        
        # 7. 첫 번째 ICP 정합
        lip_mesh, rotated_face_mesh = self._apply_icp(lip_mesh, rotated_face_mesh)

        if self.visualization:
            visualize_meshes(
                [lip_mesh, rotated_face_mesh, self.laminate_mesh], 
                ["Lip", "Rotated FaceScan", "Laminate"], 
                title="After first ICP"
            )
        
        # 8. ICP 후 노이즈 제거
        lip_mesh = self._mesh_cleaner.remove_noise_by_normal_angle(lip_mesh)
        
        # 9. 두 번째 중절치 정렬
        lip_mesh, rotated_face_mesh = self._apply_incisor_alignment(
            lip_mesh, rotated_face_mesh
        )
        
        # 10. 두 번째 ICP 정합
        lip_mesh, rotated_face_mesh = self._apply_icp(lip_mesh, rotated_face_mesh)
        
        # 11. 최종 시각화
        if self.visualization:
            visualize_meshes(
                [lip_mesh, rotated_face_mesh, self.laminate_mesh], 
                ["Lip", "Rotated FaceScan", "Laminate"], 
                title="After second ICP"
            )
        
        print("Final accumulated transformation matrix:")
        print(self.transform_matrix)
        
        return self.transform_matrix, rotated_face_mesh
    
    def _align_y_axis(self):
        """Z축 중심 180도 회전을 적용합니다."""
        rotation_matrix = MeshTransformer.create_rotation_matrix_z(np.pi)
        
        # 변환 행렬 누적
        self.transform_matrix = np.dot(rotation_matrix, self.transform_matrix)
        
        # 메쉬에 변환 적용
        MeshTransformer.apply_transformation_inplace(self.face_smile_mesh, rotation_matrix)
    
    def _extract_lip_mesh(self) -> Mesh:
        """입술 영역 메쉬를 추출합니다."""
        # 입술 텍스처 생성
        mouth_eraser = MouthEraserForFace()
        texture_with_transparent_mouth = mouth_eraser.erase_mouth(self.face_smile_path)
        
        if texture_with_transparent_mouth is None:
            print("Face with transparent mouth generation failed")
            return None
        
        # 투명 영역 메쉬 추출
        lip_mesh = self._texture_extractor.extract_transparent_region(
            mesh=self.face_smile_mesh,
            texture_image=texture_with_transparent_mouth
        )
        
        if lip_mesh is None:
            print("Lip mesh extraction failed")
            return None
        
        # 메쉬 클리닝
        lip_mesh = self._mesh_cleaner.clean_mesh(lip_mesh)
        
        print("Lip mesh extraction completed")
        return lip_mesh
    
    def _extract_upper_anterior(self, lip_mesh: Mesh):
        """상악전치부를 추출하고 로컬 좌표계를 반환합니다."""
        extraction_result = self._upper_anterior_extractor.extract(lip_mesh)
        upper_anterior_mesh = extraction_result.upper_anterior_mesh
        
        # 로컬 좌표계 정의 (고정값)
        local_coordinate_system = np.array([
            [1, 0, 0],   # X축
            [0, 0, 1],   # Y축
            [0, -1, 0]   # Z축
        ])
        
        return upper_anterior_mesh, local_coordinate_system
    
    def _transform_to_global(self, lip_mesh: Mesh, local_coordinate_system: np.ndarray):
        """로컬 좌표계에서 글로벌 좌표계로 변환합니다."""
        # 변환 행렬 계산
        rotation_matrix_3x3 = np.linalg.inv(local_coordinate_system)
        global_transform = np.eye(4)
        global_transform[:3, :3] = rotation_matrix_3x3.T
        
        # 입술 메쉬 변환
        lip_mesh.vertices = np.dot(
            lip_mesh.vertices, 
            global_transform[:3, :3].T
        ) + global_transform[:3, 3]
        
        # 페이스 메쉬 복사 및 변환
        rotated_face_mesh = copy.deepcopy(self.face_smile_mesh)
        rotated_face_mesh.vertices = np.dot(
            rotated_face_mesh.vertices, 
            global_transform[:3, :3].T
        ) + global_transform[:3, 3]
        
        # 변환 행렬 누적
        self.transform_matrix = np.dot(global_transform, self.transform_matrix)
        
        return lip_mesh, rotated_face_mesh
    
    def _apply_incisor_alignment(self, lip_mesh: Mesh, rotated_face_mesh: Mesh):
        """중절치 정렬을 적용합니다."""
        alignment_result = self._incisor_aligner.calculate_alignment_translation(
            target_mesh=self.laminate_mesh,
            source_mesh=lip_mesh
        )
        
        # 이동 적용
        MeshTransformer.translate_mesh_inplace(lip_mesh, alignment_result.translation_vector)
        MeshTransformer.translate_mesh_inplace(rotated_face_mesh, alignment_result.translation_vector)
        
        # 변환 행렬 누적
        self.transform_matrix = np.dot(
            alignment_result.translation_matrix, 
            self.transform_matrix
        )
        
        return lip_mesh, rotated_face_mesh
    
    def _apply_icp(self, lip_mesh: Mesh, rotated_face_mesh: Mesh):
        """첫 번째 ICP 정합을 적용합니다."""
        icp_result = self._icp_registrator.register(lip_mesh, self.laminate_mesh)
        
        # 변환 행렬 누적
        self.transform_matrix = np.dot(
            icp_result.transformation_matrix, 
            self.transform_matrix
        )
        
        # 메쉬에 변환 적용
        MeshTransformer.apply_rotation_and_translation_inplace(
            rotated_face_mesh,
            icp_result.transformation_matrix[:3, :3],
            icp_result.transformation_matrix[:3, 3]
        )
        
        return icp_result.transformed_mesh, rotated_face_mesh
    



if __name__ == "__main__":
    face_laminate_registration = FaceLaminateRegistration(
        "../../example/data/ahn/Smile/Smile_Scan.ply",
        "../../example/data/smile_arch_half.stl", 
        visualization=True
    )
    final_transform, moved_mesh = face_laminate_registration.run_registration()
    print(final_transform)
