"""
CBCT 데이터 처리 모듈

CBCT 볼륨에서 표면 추출, 코 중심 추정, 좌표계 변환을 담당합니다.

단일 책임: CBCT 데이터의 전처리 및 특징점 추출
"""
import numpy as np
import open3d as o3d
from typing import Tuple, Optional

try:
    from skimage import measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

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
    - LPS → 표준 좌표계 변환
    """
    
    def __init__(self, config: Optional[AlignmentConfig] = None):
        """
        Args:
            config: 정합 설정 (None일 경우 기본값 사용)
        """
        self.config = config if config is not None else AlignmentConfig()
        self.lps_to_standard_matrix = self.config.coordinate_transform.get_lps_to_standard_matrix()
    
    def extract_surface(
        self,
        dicom_folder: str,
        verbose: bool = True
    ) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
        """
        CBCT에서 피부 표면 포인트 클라우드 추출 (LPS 좌표계)
        
        Args:
            dicom_folder: DICOM 폴더 경로
            verbose: 상세 출력 여부
        
        Returns:
            Tuple[pcd_cropped, pcd_full]:
                - pcd_cropped: Z축 Crop된 표면 포인트 클라우드 (얼굴 영역)
                - pcd_full: 전체 표면 포인트 클라우드
        """
        if verbose:
            print("\n[CBCT 표면 추출] (LPS 좌표계)")
            print("-" * 50)
        
        # DICOM 로드
        loader = CBCTDicomLoader(dicom_folder)
        loader.load(orientation="LPS", verbose=verbose)
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
        LPS 좌표계 포인트 클라우드를 표준 좌표계로 변환 (코 중심을 원점으로)
        
        변환 순서:
        1. LPS → 표준 좌표계 회전
        2. 코 중심을 원점으로 평행이동
        
        Args:
            pcd: 입력 포인트 클라우드 (LPS 좌표계)
            nose_center: 코 중심 좌표 (LPS 좌표계)
            verbose: 상세 출력 여부
        
        Returns:
            CoordinateTransformResult: 변환 결과
        """
        if verbose:
            print("\n[LPS → 표준 좌표계 변환]")
            print("-" * 50)
        
        # 1. 코 중심을 표준 좌표계로 변환
        nose_center_h = np.append(nose_center, 1)  # homogeneous
        nose_center_std = (self.lps_to_standard_matrix @ nose_center_h)[:3]
        
        if verbose:
            print(f"코 중심 (LPS): {nose_center}")
            print(f"코 중심 (표준): {nose_center_std}")
        
        # 2. 원점 이동 변환 행렬
        translation_matrix = compute_translation_matrix(-nose_center_std)
        
        # 3. 결합 변환 행렬: 원점이동 @ LPS변환
        combined_transform = translation_matrix @ self.lps_to_standard_matrix
        
        # 4. 변환 적용
        pcd_transformed = apply_transform(pcd, combined_transform)
        
        if verbose:
            print(f"\n결합 변환 행렬 (LPS→표준+원점이동):")
            print(combined_transform)
            print(f"\n변환 후 포인트 수: {len(pcd_transformed.points):,}")
        
        return CoordinateTransformResult(
            pcd_standard=pcd_transformed,
            transform_matrix=combined_transform,
            nose_center_standard=np.zeros(3)  # 원점으로 이동됨
        )
    
    def transform_to_standard_coordinate_simple(
        self,
        pcd: o3d.geometry.PointCloud,
        verbose: bool = True
    ) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
        """
        LPS 좌표계 포인트 클라우드를 표준 좌표계로 회전 변환만 적용 (원점 이동 없음)
        
        Args:
            pcd: 입력 포인트 클라우드 (LPS 좌표계)
            verbose: 상세 출력 여부
        
        Returns:
            Tuple[pcd_standard, lps_to_standard_matrix]:
                - pcd_standard: 표준 좌표계로 변환된 포인트 클라우드
                - lps_to_standard_matrix: LPS→표준 변환 행렬
        """
        if verbose:
            print("\n[LPS → 표준 좌표계 회전 변환]")
            print("-" * 50)
        
        pcd_standard = apply_transform(pcd, self.lps_to_standard_matrix)
        
        if verbose:
            print(f"변환 후 포인트 수: {len(pcd_standard.points):,}")
        
        return pcd_standard, self.lps_to_standard_matrix
    
    def generate_mesh_from_volume(
        self,
        loader: CBCTDicomLoader,
        hu_threshold: float = -200.0,
        step_size: int = 1,
        target_triangles: Optional[int] = None,
        verbose: bool = True
    ) -> o3d.geometry.TriangleMesh:
        """
        마칭큐브(Marching Cubes) 알고리즘을 사용하여 CBCT 볼륨에서 메쉬 생성
        
        Args:
            loader: DICOM 로더 (이미 로드된 상태)
            hu_threshold: HU 임계값 (기본값: -200, 피부 표면)
            step_size: 마칭큐브 스텝 사이즈 (클수록 빠르지만 해상도 낮음)
            target_triangles: 목표 삼각형 수 (None이면 다운샘플링 안함, 예: 500000)
            verbose: 상세 출력 여부
        
        Returns:
            o3d.geometry.TriangleMesh: 생성된 메쉬 (LPS 좌표계)
        
        Raises:
            ImportError: scikit-image가 설치되지 않은 경우
            ValueError: 메쉬 생성 실패 시
        """
        if not SKIMAGE_AVAILABLE:
            raise ImportError(
                "마칭큐브 메쉬 생성을 위해 scikit-image가 필요합니다. "
                "pip install scikit-image"
            )
        
        if verbose:
            print("\n[마칭큐브 메쉬 생성]")
            print("-" * 50)
            print(f"  HU 임계값: {hu_threshold}")
            print(f"  스텝 사이즈: {step_size}")
        
        # 볼륨 및 spacing 가져오기
        hu_volume = loader.get_volume()  # (Z, Y, X)
        spacing_xyz = loader.get_spacing()  # (X, Y, Z)
        
        # spacing을 볼륨 순서 (Z, Y, X)로 변환
        spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
        
        if verbose:
            print(f"  볼륨 크기: {hu_volume.shape}")
            print(f"  Spacing (Z, Y, X): {spacing_zyx}")
        
        # 마칭큐브 실행
        try:
            verts, faces, normals, values = measure.marching_cubes(
                hu_volume,
                level=hu_threshold,
                spacing=spacing_zyx,
                step_size=step_size
            )
        except Exception as e:
            raise ValueError(f"마칭큐브 실행 실패: {e}") from e
        
        if verbose:
            print(f"  정점 수: {len(verts):,}")
            print(f"  면 수: {len(faces):,}")
        
        # verts는 (Z, Y, X) 순서로 반환됨 → (X, Y, Z)로 변환하여 LPS 좌표계로
        # marching_cubes의 verts는 이미 spacing이 적용된 물리 좌표
        # 하지만 순서가 (Z, Y, X)이므로 (X, Y, Z)로 변환 필요
        verts_xyz = verts[:, [2, 1, 0]]  # (Z, Y, X) → (X, Y, Z)
        
        # Open3D 메쉬 생성
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts_xyz)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        
        # 노말 계산 (마칭큐브에서 제공된 노말 대신 재계산)
        mesh.compute_vertex_normals()
        
        # 메쉬 다운샘플링 (target_triangles 지정 시)
        if target_triangles is not None and len(faces) > target_triangles:
            if verbose:
                print(f"  메쉬 다운샘플링: {len(faces):,} → {target_triangles:,} 삼각형")
            mesh = mesh.simplify_quadric_decimation(target_triangles)
            mesh.compute_vertex_normals()  # 다운샘플링 후 노말 재계산
            if verbose:
                print(f"  다운샘플링 후 정점 수: {len(mesh.vertices):,}")
                print(f"  다운샘플링 후 면 수: {len(mesh.triangles):,}")
        
        if verbose:
            print(f"  메쉬 생성 완료")
            bbox = mesh.get_axis_aligned_bounding_box()
            print(f"  바운딩 박스: {bbox.min_bound} ~ {bbox.max_bound}")
        
        return mesh

    def process(
        self,
        dicom_folder: str,
        verbose: bool = True,
        generate_mesh: bool = False,
        mesh_hu_threshold: float = -200.0,
        mesh_step_size: int = 4
    ) -> Tuple[np.ndarray, o3d.geometry.PointCloud, o3d.geometry.PointCloud, np.ndarray, Optional[o3d.geometry.TriangleMesh]]:
        """
        CBCT 데이터 전체 처리 파이프라인 (LPS 좌표계 데이터 반환)
        
        처리 순서:
        1. 표면 추출 (LPS 좌표계)
        2. 좌표계 변환 (LPS → 표준) - 코 중심 추정용
        3. 코 중심 추정 (표준 좌표계에서)
        4. 코 주변 영역 추출 (표준 좌표계에서)
        5. LPS 좌표계로 역변환하여 반환
        6. (옵션) 마칭큐브로 메쉬 생성
        
        Args:
            dicom_folder: DICOM 폴더 경로
            verbose: 상세 출력 여부
            generate_mesh: 마칭큐브 메쉬 생성 여부
            mesh_hu_threshold: 메쉬 생성 HU 임계값 (기본값: -200)
            mesh_step_size: 마칭큐브 스텝 사이즈 (기본값: 4)
        
        Returns:
            Tuple[nose_center_lps, pcd_nose_region_lps, pcd_full_lps, lps_to_standard_transform, mesh_lps]:
                - nose_center_lps: 코 중심 좌표 (LPS 좌표계)
                - pcd_nose_region_lps: 코 주변 영역 (LPS 좌표계)
                - pcd_full_lps: 전체 볼륨 (LPS 좌표계)
                - lps_to_standard_transform: LPS → 표준 변환 행렬 (코 중심 원점이동 포함)
                - mesh_lps: 마칭큐브 메쉬 (LPS 좌표계, generate_mesh=False면 None)
        """
        # DICOM 로드
        if verbose:
            print("\n[CBCT 표면 추출] (LPS 좌표계)")
            print("-" * 50)
        
        loader = CBCTDicomLoader(dicom_folder)
        loader.load(orientation="LPS", verbose=verbose)
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
        pcd_cropped_lps = np_to_pcd(pts_cropped)
        pcd_full_lps = np_to_pcd(pts_full)
        
        if verbose:
            print(f"\n결과:")
            print(f"  Cropped 포인트 수: {len(pcd_cropped_lps.points):,}")
            print(f"  Full 포인트 수: {len(pcd_full_lps.points):,}")
        
        # 2. 좌표계 변환 (LPS → 표준) - 코 중심 추정용
        pcd_cropped_std, lps_matrix = self.transform_to_standard_coordinate_simple(pcd_cropped_lps, verbose)
        
        # 3. 코 중심 추정 (표준 좌표계에서)
        nose_center_std = self.estimate_nose_center(pcd_cropped_std, verbose)

        # 4. 코 주변 영역 추출 (표준 좌표계에서)
        nose_region_std = self.extract_nose_region(pcd_cropped_std, nose_center_std, verbose)
        
        # 5. 코 중심을 원점으로 이동하는 변환 행렬 생성
        translation_matrix = compute_translation_matrix(-nose_center_std)
        combined_transform = translation_matrix @ lps_matrix
        
        # 6. 코 중심을 LPS 좌표계로 역변환
        lps_matrix_inv = np.linalg.inv(lps_matrix)
        nose_center_lps_h = lps_matrix_inv @ np.append(nose_center_std, 1)
        nose_center_lps = nose_center_lps_h[:3]
        
        # 7. 코 주변 영역을 LPS 좌표계로 역변환
        pcd_nose_region_lps = apply_transform(nose_region_std, lps_matrix_inv)
        
        if verbose:
            print(f"\n[LPS 좌표계 데이터 반환]")
            print("-" * 50)
            print(f"코 중심 (표준): {nose_center_std}")
            print(f"코 중심 (LPS): {nose_center_lps}")
            print(f"코 주변 영역 포인트 수: {len(pcd_nose_region_lps.points):,}")
            print(f"전체 볼륨 포인트 수: {len(pcd_full_lps.points):,}")
            print(f"\nLPS → 표준 변환 행렬 (코 중심 원점이동 포함):")
            print(combined_transform)
        
        # 8. 마칭큐브 메쉬 생성 (옵션)
        mesh_lps = None
        if generate_mesh:
            mesh_lps = self.generate_mesh_from_volume(
                loader=loader,
                hu_threshold=mesh_hu_threshold,
                step_size=mesh_step_size,
                verbose=verbose
            )
        
        return nose_center_lps, pcd_nose_region_lps, pcd_full_lps, combined_transform, mesh_lps

