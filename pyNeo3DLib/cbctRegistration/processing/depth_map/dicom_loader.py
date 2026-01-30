"""
CBCT DICOM Loader Module

DICOM 시리즈를 로드하고 RAI 좌표계로 변환하는 모듈
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np

try:
    import SimpleITK as sitk
except ImportError as e:
    raise ImportError("SimpleITK가 필요합니다. pip install SimpleITK") from e


class CBCTDicomLoader:
    """
    DICOM 시리즈 로더 및 좌표계 변환
    
    주요 기능:
    - SimpleITK로 DICOM 시리즈 자동 로드
    - RAI (Right-Anterior-Inferior) 좌표계로 정렬
    - 물리 좌표계 정보 (origin, spacing, direction) 관리
    
    사용 예제:
    ```python
    loader = CBCTDicomLoader("path/to/dicom")
    loader.load(orientation="RAI")
    
    # NumPy 배열 접근
    hu_volume = loader.get_volume()  # (Z, Y, X)
    
    # 메타데이터 접근
    spacing = loader.get_spacing()  # (X, Y, Z)
    origin = loader.get_origin()    # (X, Y, Z)
    
    # 인덱스 -> 물리 좌표 변환
    pts_mm = loader.index_to_physical(idx_zyx)
    ```
    """
    
    def __init__(self, dicom_folder: str | Path):
        """
        Parameters:
        -----------
        dicom_folder : str | Path
            DICOM 파일이 있는 폴더 경로
        """
        self.dicom_folder = Path(dicom_folder)
        if not self.dicom_folder.exists():
            raise FileNotFoundError(f"폴더를 찾을 수 없습니다: {self.dicom_folder}")
        
        # SimpleITK 이미지 (글로벌 좌표계 정보 포함)
        self._sitk_image: Optional[sitk.Image] = None
        
        # NumPy 배열 (Z, Y, X)
        self._volume: Optional[np.ndarray] = None
        
        # 메타데이터
        self._metadata: Dict[str, Any] = {}
    
    def load(self, orientation: str = "RAI", verbose: bool = True) -> np.ndarray:
        """
        DICOM 시리즈를 로드하고 지정된 좌표계로 정렬
        
        Parameters:
        -----------
        orientation : str
            목표 좌표계 방향 (기본값: "RAI")
            - R (Right): X축 양의 방향이 환자의 오른쪽
            - A (Anterior): Y축 양의 방향이 환자의 앞쪽
            - I (Inferior): Z축 양의 방향이 환자의 아래쪽
        verbose : bool
            진행 상황 출력 여부
            
        Returns:
        --------
        np.ndarray
            HU 볼륨 (Z, Y, X)
        """
        if verbose:
            print(f"[SimpleITK] DICOM 시리즈 로딩 중...")
        
        # SimpleITK로 DICOM 로드
        img, series_id = self._load_dicom_series(self.dicom_folder)
        
        if verbose:
            print(f"  시리즈 ID: {series_id}")
            print(f"  원본 크기: {img.GetSize()}")  # (X, Y, Z)
            print(f"  원본 spacing: {img.GetSpacing()}")  # (X, Y, Z)
            print(f"  원본 origin: {img.GetOrigin()}")
            print(f"  원본 direction: {img.GetDirection()}")
        
        # ⭐ 핵심: 환자 기준 canonical orientation으로 정렬
        if verbose:
            print(f"\n[좌표계 정렬] {orientation} 좌표계로 변환 중...")
        
        img = sitk.DICOMOrient(img, orientation)
        
        if verbose:
            print(f"  정렬 후 크기: {img.GetSize()}")
            print(f"  정렬 후 spacing: {img.GetSpacing()}")
            print(f"  정렬 후 origin: {img.GetOrigin()}")
            print(f"  정렬 후 direction: {img.GetDirection()}")
        
        # SimpleITK 이미지 저장
        self._sitk_image = img
        
        # NumPy 배열로 변환 (Z, Y, X)
        vol = sitk.GetArrayFromImage(img)
        
        if verbose:
            print(f"  NumPy 배열 shape: {vol.shape} (Z, Y, X)")
        
        # 메타데이터 저장
        spacing = img.GetSpacing()  # (X, Y, Z)
        self._metadata = {
            "patient_name": "Unknown",
            "study_date": "Unknown",
            "modality": "CT",
            "rows": int(img.GetSize()[1]),      # Y
            "columns": int(img.GetSize()[0]),   # X
            "pixel_spacing": [spacing[1], spacing[0]],  # [row(Y), col(X)]
            "slice_thickness": float(spacing[2]),       # Z
            "spacing_xyz": spacing,  # (X, Y, Z)
            "origin_xyz": img.GetOrigin(),  # (X, Y, Z)
            "direction": img.GetDirection(),
            "orientation": orientation,
            "rescale_slope": 1.0,
            "rescale_intercept": 0.0,
        }
        
        # HU 변환 (SimpleITK는 이미 HU로 변환됨)
        self._volume = vol.astype(np.float32)
        
        if verbose:
            print(f"\n[완료] 볼륨 로딩 및 좌표계 정렬 완료")
            print(f"  HU 범위: {self._volume.min():.1f} ~ {self._volume.max():.1f}")
        
        return self._volume
    
    def _load_dicom_series(self, dicom_dir: Path) -> Tuple[sitk.Image, str]:
        """
        SimpleITK로 DICOM 시리즈 로딩
        
        Returns:
        --------
        img : sitk.Image
            로드된 DICOM 이미지
        series_id : str
            시리즈 UID
        """
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
        
        if not series_ids:
            raise ValueError(f"DICOM 시리즈를 찾을 수 없습니다: {dicom_dir}")
        
        # 첫 번째 시리즈 사용 (여러 시리즈가 있을 경우)
        series_id = series_ids[0]
        dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_id)
        
        if not dicom_names:
            raise ValueError(f"시리즈 {series_id}에 파일이 없습니다")
        
        reader.SetFileNames(dicom_names)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        
        img = reader.Execute()
        
        return img, series_id
    
    # ----------------------------
    # Getters
    # ----------------------------
    def get_volume(self) -> np.ndarray:
        """HU 볼륨 반환 (Z, Y, X)"""
        if self._volume is None:
            raise RuntimeError("load()를 먼저 실행하세요")
        return self._volume
    
    def get_sitk_image(self) -> sitk.Image:
        """SimpleITK 이미지 객체 반환"""
        if self._sitk_image is None:
            raise RuntimeError("load()를 먼저 실행하세요")
        return self._sitk_image
    
    def get_metadata(self) -> Dict[str, Any]:
        """메타데이터 딕셔너리 반환"""
        if not self._metadata:
            raise RuntimeError("load()를 먼저 실행하세요")
        return self._metadata
    
    def get_spacing(self) -> Tuple[float, float, float]:
        """
        Spacing 반환 (X, Y, Z) in mm
        
        Returns:
        --------
        Tuple[float, float, float]
            (X spacing, Y spacing, Z spacing)
        """
        return self._metadata["spacing_xyz"]
    
    def get_origin(self) -> Tuple[float, float, float]:
        """
        Origin 반환 (X, Y, Z) in mm
        
        Returns:
        --------
        Tuple[float, float, float]
            (X origin, Y origin, Z origin)
        """
        return self._metadata["origin_xyz"]
    
    def get_orientation(self) -> str:
        """좌표계 방향 반환 (예: "RAI")"""
        return self._metadata.get("orientation", "Unknown")
    
    # ----------------------------
    # 좌표 변환
    # ----------------------------
    def index_to_physical(self, idx_zyx: np.ndarray) -> np.ndarray:
        """
        볼륨 인덱스 (Z, Y, X)를 물리 좌표 (X, Y, Z) mm로 변환
        
        Parameters:
        -----------
        idx_zyx : np.ndarray (N, 3)
            볼륨 인덱스 배열 [(z, y, x), ...]
            
        Returns:
        --------
        np.ndarray (N, 3)
            물리 좌표 배열 [(x, y, z), ...] in mm
        """
        if self._volume is None:
            raise RuntimeError("load()를 먼저 실행하세요")
        
        spacing = self._metadata["spacing_xyz"]
        origin = self._metadata["origin_xyz"]
        
        x_spacing = float(spacing[0])
        y_spacing = float(spacing[1])
        z_spacing = float(spacing[2])
        
        # NumPy 배열은 (Z, Y, X) 순서이므로 인덱스 변환
        z = idx_zyx[:, 0].astype(np.float32) * z_spacing
        y = idx_zyx[:, 1].astype(np.float32) * y_spacing
        x = idx_zyx[:, 2].astype(np.float32) * x_spacing
        
        # 원점 오프셋 적용 (글로벌 좌표계)
        x += origin[0]
        y += origin[1]
        z += origin[2]
        
        return np.stack([x, y, z], axis=1)
    
    def physical_to_index(self, pts_xyz: np.ndarray) -> np.ndarray:
        """
        물리 좌표 (X, Y, Z) mm를 볼륨 인덱스 (Z, Y, X)로 변환
        
        Parameters:
        -----------
        pts_xyz : np.ndarray (N, 3)
            물리 좌표 배열 [(x, y, z), ...] in mm
            
        Returns:
        --------
        np.ndarray (N, 3)
            볼륨 인덱스 배열 [(z, y, x), ...]
        """
        if self._volume is None:
            raise RuntimeError("load()를 먼저 실행하세요")
        
        spacing = self._metadata["spacing_xyz"]
        origin = self._metadata["origin_xyz"]
        
        x_spacing = float(spacing[0])
        y_spacing = float(spacing[1])
        z_spacing = float(spacing[2])
        
        # 원점 오프셋 제거
        x = (pts_xyz[:, 0] - origin[0]) / x_spacing
        y = (pts_xyz[:, 1] - origin[1]) / y_spacing
        z = (pts_xyz[:, 2] - origin[2]) / z_spacing
        
        # (Z, Y, X) 순서로 반환
        return np.stack([z, y, x], axis=1)
    
    # ----------------------------
    # 통계
    # ----------------------------
    def get_statistics(self) -> Dict[str, float]:
        """
        HU 통계 반환
        
        Returns:
        --------
        Dict[str, float]
            min, max, mean, std, median
        """
        if self._volume is None:
            raise RuntimeError("load()를 먼저 실행하세요")
        
        hu = self._volume
        return {
            "min": float(hu.min()),
            "max": float(hu.max()),
            "mean": float(hu.mean()),
            "std": float(hu.std()),
            "median": float(np.median(hu)),
        }
    
    def __repr__(self) -> str:
        if self._volume is None:
            return f"CBCTDicomLoader('{self.dicom_folder}', not loaded)"
        
        shape = self._volume.shape
        orientation = self.get_orientation()
        return (
            f"CBCTDicomLoader('{self.dicom_folder}', "
            f"shape={shape}, orientation='{orientation}')"
        )


