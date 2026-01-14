"""
페이스 정합(Face Registration) 모듈

이 패키지는 페이스 스캔 데이터의 정합 및 처리를 위한 클래스들을 제공합니다.

주요 모듈:
- faceLaminateRegistration: 페이스-라미네이트 정합 오케스트레이션
- facesRegistration: 여러 페이스 스캔 정합
- facePhotoRegistration: 페이스 사진 정합
- faceAlign: 페이스 정렬

유틸리티 모듈:
- mesh_transformer: 메쉬 변환 기능
- mesh_converter: 메쉬 형식 변환
- mesh_cleaner: 메쉬 클리닝 및 노이즈 제거
- icp_registrator: ICP 정합
- texture_mesh_extractor: 텍스처 기반 메쉬 추출
- incisor_aligner: 중절치 정렬
- upper_anterior_extractor: 상악전치부 추출
"""

# 메인 정합 클래스들
from .facesRegistration import *
from .faceLaminateRegistration import *
from .facePhotoRegistration import *
from .faceAlign import *

# 유틸리티 클래스들
from .upper_anterior_extractor import UpperAnteriorExtractor, UpperAnteriorExtractionResult
from .mesh_transformer import MeshTransformer
from .mesh_converter import MeshConverter
from .mesh_cleaner import MeshCleaner
from .icp_registrator import ICPRegistrator, ICPResult
from .texture_mesh_extractor import TextureMeshExtractor
from .incisor_aligner import IncisorAligner, IncisorAlignmentResult

__all__ = [
    # 메인 정합 클래스
    "facesRegistration", 
    "faceLaminateRegistration", 
    "facePhotoRegistration", 
    "faceAlign",
    
    # 상악전치부 추출
    "UpperAnteriorExtractor",
    "UpperAnteriorExtractionResult",
    
    # 메쉬 변환
    "MeshTransformer",
    
    # 메쉬 형식 변환
    "MeshConverter",
    
    # 메쉬 클리닝
    "MeshCleaner",
    
    # ICP 정합
    "ICPRegistrator",
    "ICPResult",
    
    # 텍스처 메쉬 추출
    "TextureMeshExtractor",
    
    # 중절치 정렬
    "IncisorAligner",
    "IncisorAlignmentResult",
]
