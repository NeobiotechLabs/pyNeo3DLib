"""
CBCT 데이터 처리 모듈

CBCT 볼륨에서 표면 추출, 코 중심 추정, 좌표계 변환을 담당합니다.

단일 책임: CBCT 데이터의 전처리 및 특징점 추출
"""
import numpy as np
import open3d as o3d
from typing import Tuple, Optional

from ..config import AlignmentConfig
from ..utils import np_to_pcd, pcd_to_np, apply_transform, compute_translation_matrix
from ..types import CBCTExtractionResult, CoordinateTransformResult
from .depth_map.dicom_loader import CBCTDicomLoader
from .depth_map.surface_extractor import CBCTSurfaceExtractor
from .depth_map.extractor import CBCTDepthMapExtractor


class CBCTProcessor:
    """
    CBCT 데이터 처리 클래스
    
    담당 기능:
    - CBCT 표면 추출
    - 코 중심 추정
    - 코 주변 영역 추출
    - RAI → 표준 좌표계 변환
    """
    
    def __init__(self, config: Optional[AlignmentConfig] = None):
        """
        Args:
            config: 정합 설정 (None일 경우 기본값 사용)
        """
        self.config = config if config is not None else AlignmentConfig()
        self.rai_to_standard_matrix = self.config.coordinate_transform.get_rai_to_standard_matrix()
    
    def extract_surface(
        self,
        dicom_folder: str,
        verbose: bool = True
    ) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
        """
        CBCT에서 피부 표면 포인트 클라우드 추출 (RAI 좌표계)
        
        Args:
            dicom_folder: DICOM 폴더 경로
            verbose: 상세 출력 여부
        
        Returns:
            Tuple[pcd_cropped, pcd_full]:
                - pcd_cropped: Z축 Crop된 표면 포인트 클라우드 (얼굴 영역)
                - pcd_full: 전체 표면 포인트 클라우드
        """
        if verbose:
            print("\n[CBCT 표면 추출] (RAI 좌표계)")
            print("-" * 50)
        
        # DICOM 로드
        loader = CBCTDicomLoader(dicom_folder)
        loader.load(orientation="RAI", verbose=verbose)
        hu_volume = loader.get_volume()
        
        # 표면 추출
        extractor = CBCTSurfaceExtractor(loader)
        cfg = self.config.cbct_extraction
        pts_cropped, pts_full = extractor.extract_surface_points(
            hu_volume=hu_volume,
            z_crop_top_ratio=cfg.z_crop_top_ratio,
            z_crop_bottom_ratio=cfg.z_crop_bottom_ratio,
            downsample_factor=cfg.downsample_factor,
            verbose=verbose,
        )
        
        # numpy → o3d.PointCloud 변환
        pcd_cropped = np_to_pcd(pts_cropped)
        pcd_full = np_to_pcd(pts_full)
        
        if verbose:
            print(f"\n결과:")
            print(f"  Cropped 포인트 수: {len(pcd_cropped.points):,}")
            print(f"  Full 포인트 수: {len(pcd_full.points):,}")
        
        return pcd_cropped, pcd_full
    
    def estimate_nose_center(
        self,
        pcd: o3d.geometry.PointCloud,
        verbose: bool = True
    ) -> np.ndarray:
        """
        포인트 클라우드에서 코 중심 좌표 추정
        
        Args:
            pcd: 얼굴 표면 포인트 클라우드
            verbose: 상세 출력 여부
        
        Returns:
            np.ndarray: 코 중심 좌표 (3,)
        """
        if verbose:
            print("\n[코 중심 추정]")
            print("-" * 50)
        
        pts = pcd_to_np(pcd)
        
        nose_cfg = self.config.nose_estimation
        nose_center = CBCTDepthMapExtractor.estimate_nose_center(
            pts,
            x_center_ratio_start=nose_cfg.x_center_ratio_start,
            x_center_ratio_end=nose_cfg.x_center_ratio_end,
        )
        
        if verbose:
            print(f"추정된 코 중심: {nose_center}")
        
        return nose_center
    
    def extract_nose_region(
        self,
        pcd: o3d.geometry.PointCloud,
        nose_center: np.ndarray,
        verbose: bool = True
    ) -> o3d.geometry.PointCloud:
        """
        코 중심 기준으로 Depth Map 레이캐스팅을 통해 코 주변 영역 추출
        
        Args:
            pcd: 얼굴 표면 포인트 클라우드
            nose_center: 코 중심 좌표
            verbose: 상세 출력 여부
        
        Returns:
            o3d.geometry.PointCloud: 코 주변 표면 포인트 클라우드
        """
        if verbose:
            print("\n[코 주변 영역 추출] (Depth Map)")
            print("-" * 50)
        
        pts = pcd_to_np(pcd)
        
        # Depth Map 추출
        depth_cfg = self.config.depth_map
        depth_extractor = CBCTDepthMapExtractor(
            pts_face=pts,
            grid_center=nose_center,
            grid_width_mm=depth_cfg.grid_width_mm,
            grid_height_mm=depth_cfg.grid_height_mm,
            grid_resolution=depth_cfg.grid_resolution,
            ray_direction=list(depth_cfg.ray_direction),
            ray_start_offset_mm=depth_cfg.ray_start_offset_mm,
            search_radius_mm=depth_cfg.search_radius_mm,
        )
        
        result = depth_extractor.extract(verbose=verbose)
        hit_points = result["hit_points_array"]
        
        # numpy → o3d.PointCloud 변환
        pcd_nose_region = np_to_pcd(hit_points)
        
        if verbose:
            print(f"\n결과: {len(pcd_nose_region.points):,} 포인트")
        
        return pcd_nose_region
    
    def transform_to_standard_coordinate(
        self,
        pcd: o3d.geometry.PointCloud,
        nose_center: np.ndarray,
        verbose: bool = True
    ) -> CoordinateTransformResult:
        """
        RAI 좌표계 포인트 클라우드를 표준 좌표계로 변환 (코 중심을 원점으로)
        
        변환 순서:
        1. RAI → 표준 좌표계 회전
        2. 코 중심을 원점으로 평행이동
        
        Args:
            pcd: 입력 포인트 클라우드 (RAI 좌표계)
            nose_center: 코 중심 좌표 (RAI 좌표계)
            verbose: 상세 출력 여부
        
        Returns:
            CoordinateTransformResult: 변환 결과
        """
        if verbose:
            print("\n[RAI → 표준 좌표계 변환]")
            print("-" * 50)
        
        # 1. 코 중심을 표준 좌표계로 변환
        nose_center_h = np.append(nose_center, 1)  # homogeneous
        nose_center_std = (self.rai_to_standard_matrix @ nose_center_h)[:3]
        
        if verbose:
            print(f"코 중심 (RAI): {nose_center}")
            print(f"코 중심 (표준): {nose_center_std}")
        
        # 2. 원점 이동 변환 행렬
        translation_matrix = compute_translation_matrix(-nose_center_std)
        
        # 3. 결합 변환 행렬: 원점이동 @ RAI변환
        combined_transform = translation_matrix @ self.rai_to_standard_matrix
        
        # 4. 변환 적용
        pcd_transformed = apply_transform(pcd, combined_transform)
        
        if verbose:
            print(f"\n결합 변환 행렬 (RAI→표준+원점이동):")
            print(combined_transform)
            print(f"\n변환 후 포인트 수: {len(pcd_transformed.points):,}")
        
        return CoordinateTransformResult(
            pcd_standard=pcd_transformed,
            transform_matrix=combined_transform,
            nose_center_standard=np.zeros(3)  # 원점으로 이동됨
        )
    
    def process(
        self,
        dicom_folder: str,
        verbose: bool = True
    ) -> Tuple[CBCTExtractionResult, CoordinateTransformResult, o3d.geometry.PointCloud]:
        """
        CBCT 데이터 전체 처리 파이프라인
        
        Args:
            dicom_folder: DICOM 폴더 경로
            verbose: 상세 출력 여부
        
        Returns:
            Tuple[extraction_result, transform_result, pcd_full_std]:
                - extraction_result: 추출 결과
                - transform_result: 좌표계 변환 결과
                - pcd_full_std: 전체 볼륨 (표준 좌표계)
        """
        # 1. 표면 추출
        pcd_cropped, pcd_full = self.extract_surface(dicom_folder, verbose)
        
        # 2. 코 중심 추정
        nose_center = self.estimate_nose_center(pcd_cropped, verbose)
        
        # 3. 코 주변 영역 추출
        nose_region = self.extract_nose_region(pcd_cropped, nose_center, verbose)
        
        extraction_result = CBCTExtractionResult(
            surface_cropped=pcd_cropped,
            surface_full=pcd_full,
            nose_center=nose_center,
            nose_region=nose_region
        )
        
        # 4. 좌표계 변환
        transform_result = self.transform_to_standard_coordinate(
            nose_region, nose_center, verbose
        )

        transform_result_matrix = transform_result.transform_matrix
        nose_center_standard = transform_result.nose_center_standard
        pcd_standard = transform_result.pcd_standard
        
        # 5. 전체 볼륨도 동일 변환 적용
        pcd_full_std = apply_transform(pcd_full, transform_result_matrix)
        
        if verbose:
            print(f"\n전체 볼륨 변환 완료: {len(pcd_full_std.points):,} 포인트")
        
        return nose_center_standard, pcd_standard, pcd_full_std, transform_result_matrix

