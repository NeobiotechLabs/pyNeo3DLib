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
    
    def load_dicom(
        self,
        dicom_folder: str,
        verbose: bool = True
    ) -> CBCTDicomLoader:
        """
        DICOM 폴더에서 CBCT 데이터 로드
        
        Args:
            dicom_folder: DICOM 폴더 경로
            verbose: 상세 출력 여부
        
        Returns:
            CBCTDicomLoader: 로드된 DICOM 로더
        """
        if verbose:
            print("\n[DICOM 로드]")
            print("-" * 50)
        
        loader = CBCTDicomLoader(dicom_folder)
        loader.load(orientation="LPS", verbose=verbose)
        return loader
    
    def extract_full_surface_from_loader(
        self,
        loader: CBCTDicomLoader,
        verbose: bool = True
    ) -> o3d.geometry.PointCloud:
        """
        로드된 DICOM에서 전체 표면 포인트 클라우드 추출 (LPS 좌표계)
        
        단일 책임: 전체 표면 추출
        영역 필터링이 필요하면 crop_surface_to_face_region() 별도 호출
        
        Args:
            loader: 로드된 DICOM 로더
            verbose: 상세 출력 여부
        
        Returns:
            o3d.geometry.PointCloud: 전체 표면 포인트 클라우드
        """
        if verbose:
            print("\n[CBCT 전체 표면 추출] (LPS 좌표계)")
            print("-" * 50)
        
        hu_volume = loader.get_volume()
        
        # 표면 추출
        extractor = CBCTSurfaceExtractor(loader)
        cfg = self.config.cbct_extraction
        pts_full = extractor.extract_full_surface_points(
            hu_volume=hu_volume,
            downsample_factor=cfg.downsample_factor,
            verbose=verbose,
        )
        
        # numpy → o3d.PointCloud 변환
        pcd_full = np_to_pcd(pts_full)
        
        if verbose:
            print(f"\n결과: Full 포인트 수: {len(pcd_full.points):,}")
        
        return pcd_full
    
    def crop_surface_to_face_region(
        self,
        pcd: o3d.geometry.PointCloud,
        verbose: bool = True
    ) -> o3d.geometry.PointCloud:
        """
        포인트 클라우드에서 얼굴 영역만 필터링
        
        단일 책임: 영역 필터링 (crop)
        
        Args:
            pcd: 전체 표면 포인트 클라우드
            verbose: 상세 출력 여부
        
        Returns:
            o3d.geometry.PointCloud: 얼굴 영역만 포함된 포인트 클라우드
        """
        if verbose:
            print("\n[얼굴 영역 필터링]")
            print("-" * 50)
        
        pts = pcd_to_np(pcd)
        cfg = self.config.cbct_extraction
        
        # 물리 좌표 범위 계산
        x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
        z_min, z_max = pts[:, 2].min(), pts[:, 2].max()
        
        x_range = x_max - x_min
        z_range = z_max - z_min
        
        # crop 경계 계산
        z_crop_min = z_min + z_range * cfg.z_crop_bottom_ratio
        z_crop_max = z_max - z_range * cfg.z_crop_top_ratio
        x_crop_min = x_min + x_range * cfg.x_crop_ratio_start
        x_crop_max = x_min + x_range * cfg.x_crop_ratio_end
        
        # 마스크 적용
        mask = (
            (pts[:, 2] >= z_crop_min) & (pts[:, 2] <= z_crop_max) &
            (pts[:, 0] >= x_crop_min) & (pts[:, 0] <= x_crop_max)
        )
        
        pts_cropped = pts[mask]
        pcd_cropped = np_to_pcd(pts_cropped)
        
        if verbose:
            print(f"  Z crop: {z_crop_min:.1f}~{z_crop_max:.1f} mm")
            print(f"  X crop: {x_crop_min:.1f}~{x_crop_max:.1f} mm")
            print(f"  결과: {len(pts):,} → {len(pts_cropped):,}개 포인트")
        
        return pcd_cropped
    
    def extract_full_surface(
        self,
        dicom_folder: str,
        verbose: bool = True
    ) -> o3d.geometry.PointCloud:
        """
        CBCT에서 전체 표면 포인트 클라우드 추출 (LPS 좌표계)
        
        편의 메서드: load_dicom + extract_full_surface_from_loader 조합
        
        Args:
            dicom_folder: DICOM 폴더 경로
            verbose: 상세 출력 여부
        
        Returns:
            o3d.geometry.PointCloud: 전체 표면 포인트 클라우드
        """
        loader = self.load_dicom(dicom_folder, verbose)
        return self.extract_full_surface_from_loader(loader, verbose)
    
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
        
        pts_face = pcd_to_np(pcd)

        # y값이 큰 순서대로 정렬하고 상위 5% 포인트 추출
        sorted_indices = pts_face[:, 1].argsort()[::-1]  # y값 내림차순 정렬
        top_5_percent_count = max(1, int(len(pts_face) * 0.05))
        top_points = pts_face[sorted_indices[:top_5_percent_count]]
        
        # 상위 5% 포인트의 중앙값 위치 계산 후 y값만 최댓값으로 대체
        # 중앙값은 이상치(outlier)에 강건하여 더 안정적인 중심 추정 가능
        nose_center = np.median(top_points, axis=0)
        nose_center[1] = pts_face[:, 1].max()
        
        if verbose:
            print(f"estimated nose center: {nose_center}")
        
        return nose_center
    
    def extract_nose_region(
        self,
        pcd: o3d.geometry.PointCloud,
        verbose: bool = True
    ) -> o3d.geometry.PointCloud:
        """
        포인트 클라우드의 실제 범위를 기반으로 Depth Map 레이캐스팅을 통해 표면 영역 추출
        
        격자 중심은 입력 포인트 클라우드의 범위에서 자동 계산:
        - X, Z: 범위의 중심
        - Y: 최대값 (가장 앞쪽, 레이캐스팅 시작점)
        
        Args:
            pcd: 얼굴 표면 포인트 클라우드
            verbose: 상세 출력 여부
        
        Returns:
            o3d.geometry.PointCloud: 표면 포인트 클라우드
        """
        if verbose:
            print("\n[표면 영역 추출] (Depth Map)")
            print("-" * 50)
        
        pts = pcd_to_np(pcd)
        
        # pts의 실제 X, Y, Z 범위 계산
        x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
        y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
        z_min, z_max = pts[:, 2].min(), pts[:, 2].max()
        
        # 격자 크기: pts의 실제 범위 사용
        grid_width_mm = x_max - x_min
        grid_height_mm = z_max - z_min
        
        # 격자 중심: X, Z는 범위의 중심, Y는 최대값 (가장 앞쪽)
        grid_center = np.array([
            (x_min + x_max) / 2,
            y_max,  # 가장 앞쪽 (레이캐스팅 시작점)
            (z_min + z_max) / 2
        ])
        
        if verbose:
            print(f"pts 범위 - X: [{x_min:.1f}, {x_max:.1f}], Y: [{y_min:.1f}, {y_max:.1f}], Z: [{z_min:.1f}, {z_max:.1f}]")
            print(f"격자 크기: {grid_width_mm:.1f}mm x {grid_height_mm:.1f}mm")
            print(f"격자 중심: {grid_center}")
        
        # Depth Map 추출
        depth_cfg = self.config.depth_map
        depth_extractor = CBCTDepthMapExtractor(
            pts_face=pts,
            grid_center=grid_center,
            grid_width_mm=grid_width_mm,
            grid_height_mm=grid_height_mm,
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
    ) -> o3d.geometry.PointCloud:
        """
        LPS 좌표계 포인트 클라우드를 표준 좌표계로 회전 변환만 적용 (원점 이동 없음)
        
        변환 행렬은 get_lps_to_standard_matrix() 메서드로 별도 접근 가능
        
        Args:
            pcd: 입력 포인트 클라우드 (LPS 좌표계)
            verbose: 상세 출력 여부
        
        Returns:
            o3d.geometry.PointCloud: 표준 좌표계로 변환된 포인트 클라우드
        """
        if verbose:
            print("\n[LPS → 표준 좌표계 회전 변환]")
            print("-" * 50)
        
        pcd_standard = apply_transform(pcd, self.lps_to_standard_matrix)
        
        if verbose:
            print(f"변환 후 포인트 수: {len(pcd_standard.points):,}")
        
        return pcd_standard
    
    def get_lps_to_standard_matrix(self) -> np.ndarray:
        """
        LPS → 표준 좌표계 변환 행렬 반환
        
        Returns:
            np.ndarray: 4x4 변환 행렬
        """
        return self.lps_to_standard_matrix
    
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
