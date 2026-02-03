"""
CBCT Surface Point Cloud Extractor Module

CBCT 볼륨에서 피부/연조직 표면 포인트클라우드를 추출하는 모듈
"""

from __future__ import annotations

import os
from typing import Optional, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:
    from scipy.signal import find_peaks
    from scipy.ndimage import gaussian_filter1d
    from scipy import ndimage
except ImportError as e:
    raise ImportError("scipy가 필요합니다. pip install scipy") from e

try:
    from sklearn.cluster import DBSCAN
except ImportError as e:
    raise ImportError("scikit-learn이 필요합니다. pip install scikit-learn") from e

try:
    import open3d as o3d
except ImportError as e:
    raise ImportError("open3d가 필요합니다. pip install open3d") from e

from .dicom_loader import CBCTDicomLoader


class CBCTSurfaceExtractor:
    """
    CBCT 볼륨에서 피부/연조직 표면 포인트클라우드 추출
    
    주요 기능:
    - HU 히스토그램 분석으로 연조직 피크 자동 추정
    - Threshold 기반 마스크 생성
    - Morphology 연산으로 노이즈 제거
    - 3D erosion으로 진짜 표면만 추출
    - 코 중심 추정 (PCA + DBSCAN)
    - ROI 크롭
    
    사용 예제:
    ```python
    from cbct.dicom_loader import CBCTDicomLoader
    from cbct.surface_extractor import CBCTSurfaceExtractor
    
    # DICOM 로드
    loader = CBCTDicomLoader("path/to/dicom")
    loader.load(orientation="LPS")
    
    # 1. 원본 CBCT 전체 볼륨 추출
    extractor = CBCTSurfaceExtractor(loader)
    full_volume_raw = extractor.extract_full_volume_points(
        threshold_offset=-50.0
    )
    
    # 2. 표면 추출 및 전체 볼륨 Crop/Downsample
    surface_pts, full_volume_pts = extractor.extract_surface_points(
        threshold_offset=-50.0,
        z_crop_top_ratio=0.4,
        z_crop_bottom_ratio=0.3,
        full_volume_pts=full_volume_raw,
    )
    ```
    """
    
    def __init__(self, dicom_loader: CBCTDicomLoader):
        """
        Parameters:
        -----------
        dicom_loader : CBCTDicomLoader
            로드된 DICOM 로더 객체
        """
        self.loader = dicom_loader
        
        # 캐시
        self._skin_hu_cache: Optional[float] = None
        self._peak_info_cache: Optional[dict] = None
    
    # ----------------------------
    # HU 분석
    # ----------------------------
    def find_peaks_in_histogram(self, bins: int = 256) -> dict:
        """
        HU 히스토그램에서 피크를 찾아 연조직/뼈 분리 힌트 제공
        
        mean 이상 구간에서 피크를 찾아:
        - 첫 번째 피크: 연조직(피부)
        - 두 번째 피크: 뼈
        
        Parameters:
        -----------
        bins : int
            히스토그램 빈 개수
            
        Returns:
        --------
        dict
            - mean_hu: 평균 HU
            - peaks: 피크 리스트 [{"hu": float, "count": int}, ...]
            - soft_tissue_peak: 연조직 피크 HU
            - soft_tissue_range: 연조직 범위 (min, max)
            - bone_peak: 뼈 피크 HU (있으면)
            - bone_range: 뼈 범위 (min, max)
        """
        hu = self.loader.get_volume()
        stats = self.loader.get_statistics()
        mean_hu = stats["mean"]
        
        hu_min, hu_max = float(hu.min()), float(hu.max())
        counts, bin_edges = np.histogram(hu.flatten(), bins=bins, range=(hu_min, hu_max))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        
        # 스무딩
        counts_smooth = gaussian_filter1d(counts.astype(float), sigma=3)
        mean_idx = int(np.searchsorted(bin_centers, mean_hu))
        
        # mean 이상 구간에서 피크 찾기
        peaks, _ = find_peaks(
            counts_smooth[mean_idx:],
            prominence=float(counts_smooth[mean_idx:].max()) * 0.05,
            distance=bins // 20,
        )
        peaks = peaks + mean_idx
        
        result = {
            "mean_hu": float(mean_hu),
            "peaks": [],
            "soft_tissue_range": None,
            "bone_range": None,
        }
        
        if len(peaks) == 0:
            # 피크 없으면 기본값
            result["soft_tissue_range"] = (-200.0, 200.0)
            result["bone_range"] = (200.0, float(hu_max))
            self._peak_info_cache = result
            return result
        
        # 피크 정보 저장
        for p in peaks:
            result["peaks"].append({"hu": float(bin_centers[p]), "count": int(counts[p])})
        
        peak_hus = sorted([float(bin_centers[p]) for p in peaks])
        
        # 첫 번째 피크 = 연조직
        soft_peak = peak_hus[0]
        soft_min = max(float(mean_hu), soft_peak - 200.0)
        soft_max = soft_peak + 200.0
        result["soft_tissue_peak"] = float(soft_peak)
        result["soft_tissue_range"] = (float(soft_min), float(soft_max))
        
        # 두 번째 피크 = 뼈
        if len(peak_hus) >= 2:
            bone_peak = peak_hus[-1]
            result["bone_peak"] = float(bone_peak)
            bone_min = float(soft_max)
            bone_max = min(float(hu_max), bone_peak + 500.0)
            result["bone_range"] = (float(bone_min), float(bone_max))
        else:
            result["bone_range"] = (float(soft_max), float(hu_max))
        
        self._peak_info_cache = result
        return result
    
    def get_skin_hu(self, bins: int = 256) -> float:
        """
        연조직(피부) HU 값 추정
        
        mean 이상 구간의 첫 번째 피크를 연조직으로 가정
        
        Parameters:
        -----------
        bins : int
            히스토그램 빈 개수
            
        Returns:
        --------
        float
            연조직 HU 값
        """
        if self._skin_hu_cache is not None:
            return self._skin_hu_cache
        
        info = self.find_peaks_in_histogram(bins=bins)
        skin_hu = float(info.get("soft_tissue_peak", 50.0))
        self._skin_hu_cache = skin_hu
        return skin_hu
    
    # ----------------------------
    # 표면 추출 하위 메서드 (SRP)
    # ----------------------------
    def _compute_threshold(
        self,
        skin_hu: Optional[float],
        threshold_offset: float,
        bins_for_skin: int,
    ) -> float:
        """Threshold 값 계산"""
        if skin_hu is None:
            skin_hu = self.get_skin_hu(bins=bins_for_skin)
        return float(skin_hu + threshold_offset)
        
    def _crop_and_downsample(
        self,
        hu: np.ndarray,
        z_crop_top_ratio: float,
        z_crop_bottom_ratio: float,
        downsample_factor: int,
        verbose: bool,
    ) -> Tuple[np.ndarray, int]:
        """
        볼륨 crop 및 다운샘플링
        
        Parameters:
        -----------
        hu : np.ndarray
            원본 볼륨
        z_crop_top_ratio : float
            Z축 상부 제거 비율
        z_crop_bottom_ratio : float
            Z축 하부 제거 비율
        downsample_factor : int
            다운샘플링 배율
        verbose : bool
            진행 상황 출력 여부
            
        Returns:
        --------
        Tuple[np.ndarray, int]
            - 처리된 볼륨 (crop + downsample)
            - z_start (원본 좌표 복원용)
        """
        Z = hu.shape[0]
        
        z_start = int(Z * z_crop_top_ratio)
        z_end = int(Z * (1.0 - z_crop_bottom_ratio))
        
        if z_start >= z_end:
            raise ValueError(
                f"Z축 crop 범위 오류: z_crop_top_ratio({z_crop_top_ratio}) + "
                f"z_crop_bottom_ratio({z_crop_bottom_ratio})가 1.0 이상입니다."
            )
        
        hu_cropped = hu[z_start:z_end, :, :]
        
        if verbose:
            print(f"  Z축 crop: {z_start}~{z_end} / {Z} "
                  f"(상부 {z_crop_top_ratio*100:.0f}% 제거, 하부 {z_crop_bottom_ratio*100:.0f}% 제거)")
            print(f"  crop 후 볼륨: shape={hu_cropped.shape}")
        
        if downsample_factor > 1:
            hu_processed = hu_cropped[::downsample_factor, ::downsample_factor, ::downsample_factor]
            if verbose:
                print(f"  다운샘플링: {downsample_factor}배 ({hu_cropped.shape} → {hu_processed.shape})")
        else:
            hu_processed = hu_cropped
            if verbose:
                print(f"  다운샘플링: 미적용 (downsample_factor={downsample_factor})")
        
        return hu_processed, z_start
    
    def _create_binary_mask(self, hu: np.ndarray, threshold: float) -> np.ndarray:
        """Threshold 기반 바이너리 마스크 생성"""
        return hu > threshold
    
    def _apply_closing(
        self,
        binary: np.ndarray,
        closing_iter: int,
        use_fast_mode: bool,
        verbose: bool,
    ) -> np.ndarray:
        """Morphology closing 적용"""
        if closing_iter <= 0:
            return binary
        
        if use_fast_mode:
            if verbose:
                print(f"  Closing (병렬): {closing_iter}회")
            return self._parallel_morphology(binary, "closing", closing_iter)
        else:
            if verbose:
                print(f"  Closing: {closing_iter}회")
            return ndimage.binary_closing(binary, iterations=closing_iter)
    
    def _extract_surface_mask(
        self,
        binary: np.ndarray,
        erosion_iter: int,
        verbose: bool,
    ) -> np.ndarray:
        """바이너리 마스크에서 표면 마스크 추출"""
        if verbose:
            print(f"  표면 추출 (erosion={erosion_iter}회)")
        return self._surface_from_binary(binary, erosion_iter=erosion_iter)
    
    def _extract_and_transform_points(
        self,
        surface: np.ndarray,
        z_start: int,
        downsample_factor: int,
    ) -> np.ndarray:
        """
        표면 마스크에서 포인트 추출 및 물리 좌표 변환
        
        Raises:
        -------
        RuntimeError
            추출된 포인트가 너무 적을 때
        """
        idx = np.argwhere(surface)  # (z, y, x) in cropped & downsampled
        
        if idx.shape[0] < 2000:
            raise RuntimeError(
                "표면 포인트가 너무 적습니다. threshold_offset / z_crop_ratio / closing_iter를 조정하세요."
            )
        
        # 원본 스케일로 복원
        if downsample_factor > 1:
            idx = idx * downsample_factor
        
        # Z축 오프셋 복원
        idx[:, 0] += z_start
        
        # 물리 좌표로 변환
        return self.loader.index_to_physical(idx)
    
    def _limit_point_count(
        self,
        pts: np.ndarray,
        max_points: int,
        verbose: bool,
    ) -> np.ndarray:
        """포인트 개수 제한 (랜덤 샘플링)"""
        if pts.shape[0] <= max_points:
            return pts
        
        if verbose:
            print(f"  포인트 다운샘플링: {pts.shape[0]:,} -> {max_points:,}")
        
        sel = np.random.choice(pts.shape[0], size=max_points, replace=False)
        return pts[sel]
    
    def _surface_from_binary(self, binary: np.ndarray, erosion_iter: int = 1) -> np.ndarray:
        """
        바이너리 마스크에서 표면만 추출
        
        surface = mask & ~erode(mask)
        
        중요: 3D erosion을 사용하여 진짜 표면만 추출
        (2D 슬라이스별 처리는 Z축 방향 내부가 남아있음)
        
        Parameters:
        -----------
        binary : np.ndarray
            바이너리 마스크
        erosion_iter : int
            Erosion 반복 횟수
            
        Returns:
        --------
        np.ndarray
            표면 마스크
        """
        er = ndimage.binary_erosion(binary, iterations=erosion_iter)
        return binary & (~er)
    
    def _parallel_morphology(
        self,
        binary: np.ndarray,
        operation: str,
        iterations: int
    ) -> np.ndarray:
        """
        슬라이스별 병렬 morphology 연산
        
        Parameters:
        -----------
        binary : np.ndarray
            바이너리 마스크
        operation : str
            연산 종류 ("closing", "erosion", "dilation")
        iterations : int
            반복 횟수
            
        Returns:
        --------
        np.ndarray
            연산 결과
        """
        if binary.shape[0] < 4:  # 작은 볼륨은 순차 처리
            if operation == "closing":
                return ndimage.binary_closing(binary, iterations=iterations)
            elif operation == "erosion":
                return ndimage.binary_erosion(binary, iterations=iterations)
            elif operation == "dilation":
                return ndimage.binary_dilation(binary, iterations=iterations)
        
        def process_slice(z_slice: np.ndarray) -> np.ndarray:
            if operation == "closing":
                return ndimage.binary_closing(z_slice, iterations=iterations)
            elif operation == "erosion":
                return ndimage.binary_erosion(z_slice, iterations=iterations)
            elif operation == "dilation":
                return ndimage.binary_dilation(z_slice, iterations=iterations)
            return z_slice
        
        result_slices = []
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as executor:
            futures = [executor.submit(process_slice, binary[z]) for z in range(binary.shape[0])]
            for future in futures:  # 순서 보장
                result_slices.append(future.result())
        
        return np.stack(result_slices, axis=0)



    # ----------------------------
    # 표면 추출
    # ----------------------------
    def extract_surface_points(
        self,
        hu_volume: np.ndarray,
        z_crop_top_ratio: float = 0.0,
        z_crop_bottom_ratio: float = 0.0,
        x_crop_ratio_start: float = 0.0,
        x_crop_ratio_end: float = 1.0,
        downsample_factor: int = 2,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        피부/연조직 표면 포인트클라우드 추출
        
        Morphology 연산(closing/erosion)을 전체 볼륨에서 수행한 후,
        Z crop과 X crop을 포인트 레벨에서 적용하여 경계 품질 향상
        
        Parameters:
        -----------
        hu_volume : np.ndarray
            원본 HU 볼륨
        z_crop_top_ratio : float
            Z축 상부 제거 비율 (0.0~1.0)
        z_crop_bottom_ratio : float
            Z축 하부 제거 비율 (0.0~1.0)
        x_crop_ratio_start : float
            X축 좌측 시작 비율 (0.0~1.0), 이 비율 이전 제거
        x_crop_ratio_end : float
            X축 우측 종료 비율 (0.0~1.0), 이 비율 이후 제거
        downsample_factor : int
            다운샘플링 배율 (1=원본, 2=각 축 1/2, 연산량 1/8)
        verbose : bool
            진행 상황 출력 여부
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            - pts_cropped: Z/X crop 적용된 표면 포인트클라우드 (N, 3) - (X, Y, Z) in mm
            - pts_full: crop 전 전체 표면 포인트클라우드 (M, 3) - (X, Y, Z) in mm
        """
        # 내부 기본 설정
        THRESHOLD_OFFSET = -50.0
        CLOSING_ITER = 1
        EROSION_ITER = 1
        MAX_POINTS = 300_000
        BINS_FOR_SKIN = 256
        USE_FAST_MODE = True
        
        # 1. Threshold 계산
        threshold = self._compute_threshold(None, THRESHOLD_OFFSET, BINS_FOR_SKIN)
        
        if verbose:
            print(f"\n[표면 추출 시작]")
            print(f"  Threshold: {threshold:.1f} HU")
        
        # 2. 다운샘플링만 먼저 (Z crop은 나중에 포인트 레벨에서)
        if downsample_factor > 1:
            hu_use = hu_volume[::downsample_factor, ::downsample_factor, ::downsample_factor]
            if verbose:
                print(f"  다운샘플링: {downsample_factor}배 ({hu_volume.shape} → {hu_use.shape})")
        else:
            hu_use = hu_volume
            if verbose:
                print(f"  다운샘플링: 미적용 (downsample_factor={downsample_factor})")
        
        # 3. Binary 마스크 생성
        binary = self._create_binary_mask(hu_use, threshold)
        
        # 4. Morphology closing (전체 볼륨에서 수행)
        binary = self._apply_closing(binary, CLOSING_ITER, USE_FAST_MODE, verbose)
        
        # 5. 표면 추출
        surface = self._extract_surface_mask(binary, EROSION_ITER, verbose)
        
        # 6. 포인트 추출
        idx = np.argwhere(surface)  # (z, y, x) in downsampled coords
        
        if idx.shape[0] < 2000:
            raise RuntimeError(
                "표면 포인트가 너무 적습니다. threshold_offset을 조정하세요."
            )
        
        # 7. 원본 스케일 복원 (crop 전에 수행)
        if downsample_factor > 1:
            idx = idx * downsample_factor
        
        # 8. 전체 포인트 (crop 전) 물리 좌표 변환
        pts_full = self.loader.index_to_physical(idx)
        pts_full = self._limit_point_count(pts_full, MAX_POINTS, verbose)
        
        if verbose:
            print(f"  전체 표면: {pts_full.shape[0]:,}개 포인트")
        
        # 9. Z crop + X crop (포인트 레벨에서 적용)
        Z_orig = hu_volume.shape[0]  # 원본 Z 크기
        X_orig = hu_volume.shape[2]  # 원본 X 크기 (볼륨은 Z, Y, X 순서)
        
        z_min = int(Z_orig * z_crop_bottom_ratio)
        z_max = int(Z_orig * (1.0 - z_crop_top_ratio))
        x_min = int(X_orig * x_crop_ratio_start)
        x_max = int(X_orig * x_crop_ratio_end)
        
        do_z_crop = z_crop_top_ratio > 0 or z_crop_bottom_ratio > 0
        do_x_crop = x_crop_ratio_start > 0 or x_crop_ratio_end < 1.0
        
        if do_z_crop or do_x_crop:
            # idx는 원본 스케일로 복원된 상태 (z, y, x 순서)
            mask = np.ones(idx.shape[0], dtype=bool)
            
            if do_z_crop:
                mask &= (idx[:, 0] >= z_min) & (idx[:, 0] < z_max)
                if verbose:
                    print(f"  Z crop (포인트): {z_min}~{z_max} / {Z_orig} "
                          f"(상부 {z_crop_top_ratio*100:.0f}% 제거, 하부 {z_crop_bottom_ratio*100:.0f}% 제거)")
            
            if do_x_crop:
                mask &= (idx[:, 2] >= x_min) & (idx[:, 2] < x_max)
                if verbose:
                    print(f"  X crop (포인트): {x_min}~{x_max} / {X_orig} "
                          f"(좌측 {x_crop_ratio_start*100:.0f}%~우측 {x_crop_ratio_end*100:.0f}% 유지)")
            
            idx_cropped = idx[mask]
            
            # crop된 포인트 물리 좌표 변환
            pts_cropped = self.loader.index_to_physical(idx_cropped)
            pts_cropped = self._limit_point_count(pts_cropped, MAX_POINTS, verbose=False)
        else:
            # crop 없으면 동일
            pts_cropped = pts_full.copy()
        
        if verbose:
            print(f"[표면 추출 완료] crop 후: {pts_cropped.shape[0]:,}개, 전체: {pts_full.shape[0]:,}개 포인트")

        return pts_cropped, pts_full
    
    def extract_full_surface_points(
        self,
        hu_volume: np.ndarray,
        downsample_factor: int = 2,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        전체 표면 포인트클라우드 추출 (crop 없음)
        
        단일 책임: 전체 표면 포인트 추출
        crop이 필요하면 crop_points_to_region() 메서드를 별도로 호출
        
        Parameters:
        -----------
        hu_volume : np.ndarray
            원본 HU 볼륨
        downsample_factor : int
            다운샘플링 배율
        verbose : bool
            진행 상황 출력 여부
            
        Returns:
        --------
        np.ndarray
            전체 표면 포인트클라우드 (N, 3) - (X, Y, Z) in mm
        """
        # 내부 기본 설정
        THRESHOLD_OFFSET = -50.0
        CLOSING_ITER = 1
        EROSION_ITER = 1
        MAX_POINTS = 300_000
        BINS_FOR_SKIN = 256
        USE_FAST_MODE = True
        
        # 1. Threshold 계산
        threshold = self._compute_threshold(None, THRESHOLD_OFFSET, BINS_FOR_SKIN)
        
        if verbose:
            print(f"\n[전체 표면 추출 시작]")
            print(f"  Threshold: {threshold:.1f} HU")
        
        # 2. 다운샘플링
        if downsample_factor > 1:
            hu_use = hu_volume[::downsample_factor, ::downsample_factor, ::downsample_factor]
            if verbose:
                print(f"  다운샘플링: {downsample_factor}배 ({hu_volume.shape} → {hu_use.shape})")
        else:
            hu_use = hu_volume
            if verbose:
                print(f"  다운샘플링: 미적용 (downsample_factor={downsample_factor})")
        
        # 3. Binary 마스크 생성
        binary = self._create_binary_mask(hu_use, threshold)
        
        # 4. Morphology closing
        binary = self._apply_closing(binary, CLOSING_ITER, USE_FAST_MODE, verbose)
        
        # 5. 표면 추출
        surface = self._extract_surface_mask(binary, EROSION_ITER, verbose)
        
        # 6. 포인트 추출
        idx = np.argwhere(surface)
        
        if idx.shape[0] < 2000:
            raise RuntimeError(
                "표면 포인트가 너무 적습니다. threshold_offset을 조정하세요."
            )
        
        # 7. 원본 스케일 복원
        if downsample_factor > 1:
            idx = idx * downsample_factor
        
        # 8. 물리 좌표 변환
        pts_full = self.loader.index_to_physical(idx)
        pts_full = self._limit_point_count(pts_full, MAX_POINTS, verbose)
        
        if verbose:
            print(f"[전체 표면 추출 완료] {pts_full.shape[0]:,}개 포인트")

        return pts_full
    
    def crop_points_to_region(
        self,
        pts: np.ndarray,
        hu_volume_shape: Tuple[int, int, int],
        z_crop_top_ratio: float = 0.0,
        z_crop_bottom_ratio: float = 0.0,
        x_crop_ratio_start: float = 0.0,
        x_crop_ratio_end: float = 1.0,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        이미 추출된 포인트에서 특정 영역만 필터링
        
        단일 책임: 포인트 영역 필터링
        
        Parameters:
        -----------
        pts : np.ndarray
            입력 포인트클라우드 (N, 3) - (X, Y, Z) in mm
        hu_volume_shape : Tuple[int, int, int]
            원본 볼륨 shape (Z, Y, X)
        z_crop_top_ratio : float
            Z축 상부 제거 비율 (0.0~1.0)
        z_crop_bottom_ratio : float
            Z축 하부 제거 비율 (0.0~1.0)
        x_crop_ratio_start : float
            X축 좌측 시작 비율 (0.0~1.0)
        x_crop_ratio_end : float
            X축 우측 종료 비율 (0.0~1.0)
        verbose : bool
            진행 상황 출력 여부
            
        Returns:
        --------
        np.ndarray
            필터링된 포인트클라우드 (M, 3) - (X, Y, Z) in mm
        """
        if pts.shape[0] == 0:
            return pts
        
        # 물리 좌표 범위 계산
        x_min_phys, x_max_phys = pts[:, 0].min(), pts[:, 0].max()
        z_min_phys, z_max_phys = pts[:, 2].min(), pts[:, 2].max()
        
        x_range = x_max_phys - x_min_phys
        z_range = z_max_phys - z_min_phys
        
        # crop 경계 계산 (물리 좌표)
        z_crop_min = z_min_phys + z_range * z_crop_bottom_ratio
        z_crop_max = z_max_phys - z_range * z_crop_top_ratio
        x_crop_min = x_min_phys + x_range * x_crop_ratio_start
        x_crop_max = x_min_phys + x_range * x_crop_ratio_end
        
        do_z_crop = z_crop_top_ratio > 0 or z_crop_bottom_ratio > 0
        do_x_crop = x_crop_ratio_start > 0 or x_crop_ratio_end < 1.0
        
        if not do_z_crop and not do_x_crop:
            return pts
        
        mask = np.ones(pts.shape[0], dtype=bool)
        
        if do_z_crop:
            mask &= (pts[:, 2] >= z_crop_min) & (pts[:, 2] <= z_crop_max)
            if verbose:
                print(f"  Z crop: {z_crop_min:.1f}~{z_crop_max:.1f} mm "
                      f"(상부 {z_crop_top_ratio*100:.0f}% 제거, 하부 {z_crop_bottom_ratio*100:.0f}% 제거)")
        
        if do_x_crop:
            mask &= (pts[:, 0] >= x_crop_min) & (pts[:, 0] <= x_crop_max)
            if verbose:
                print(f"  X crop: {x_crop_min:.1f}~{x_crop_max:.1f} mm "
                      f"(좌측 {x_crop_ratio_start*100:.0f}%~우측 {x_crop_ratio_end*100:.0f}% 유지)")
        
        pts_cropped = pts[mask]
        
        if verbose:
            print(f"  Crop 결과: {pts.shape[0]:,} → {pts_cropped.shape[0]:,}개 포인트")
        
        return pts_cropped
    
    


