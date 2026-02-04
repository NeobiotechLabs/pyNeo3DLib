"""
CBCT DICOM Loader Module

pydicom을 사용하여 DICOM 메타데이터를 직접 읽고,
ImagePositionPatient, ImageOrientationPatient를 올바르게 적용하여
vtk.js와 동일한 좌표계 결과를 얻습니다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import os

import numpy as np

try:
    import pydicom
except ImportError as e:
    raise ImportError("pydicom이 필요합니다. pip install pydicom") from e


class CBCTDicomLoader:
    """
    DICOM 시리즈 로더 (pydicom 기반)
    
    주요 기능:
    - pydicom으로 DICOM 메타데이터 직접 읽기
    - ImagePositionPatient, ImageOrientationPatient 올바르게 적용
    - vtk.js와 동일한 LPS 좌표계 결과
    
    사용 예제:
    ```python
    loader = CBCTDicomLoader("path/to/dicom")
    loader.load()
    
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
        
        # NumPy 배열 (Z, Y, X)
        self._volume: Optional[np.ndarray] = None
        
        # DICOM 좌표 변환 정보
        self._image_position: Optional[np.ndarray] = None  # 첫 슬라이스 원점 (X, Y, Z)
        self._row_direction: Optional[np.ndarray] = None   # 행 방향 벡터
        self._col_direction: Optional[np.ndarray] = None   # 열 방향 벡터
        self._slice_direction: Optional[np.ndarray] = None # 슬라이스 방향 벡터
        
        # 메타데이터
        self._metadata: Dict[str, Any] = {}
    
    def load(self, orientation: str = "LPS", verbose: bool = True) -> np.ndarray:
        """
        DICOM 시리즈를 로드 (pydicom 사용)
        
        Parameters:
        -----------
        orientation : str
            좌표계 방향 (LPS 기본, vtk.js와 동일)
        verbose : bool
            진행 상황 출력 여부
            
        Returns:
        --------
        np.ndarray
            HU 볼륨 (Z, Y, X)
        """
        if verbose:
            print(f"[pydicom] DICOM 시리즈 로딩 중...")
            print(f"  폴더: {self.dicom_folder}")
        
        # DICOM 파일 목록 가져오기
        dicom_files = self._get_dicom_files()
        
        if not dicom_files:
            raise ValueError(f"DICOM 파일을 찾을 수 없습니다: {self.dicom_folder}")
        
        if verbose:
            print(f"  파일 수: {len(dicom_files)}")
        
        # 첫 번째 파일에서 메타데이터 읽기
        first_ds = pydicom.dcmread(dicom_files[0])
        
        # 이미지 크기
        rows = int(first_ds.Rows)
        cols = int(first_ds.Columns)
        
        # 픽셀 간격 (row, column) -> (Y, X)
        pixel_spacing = [float(x) for x in first_ds.PixelSpacing]  # [row_spacing, col_spacing]
        
        # ImageOrientationPatient: [row_x, row_y, row_z, col_x, col_y, col_z]
        orientation_patient = [float(x) for x in first_ds.ImageOrientationPatient]
        self._row_direction = np.array(orientation_patient[0:3])  # 행 방향 (X 증가)
        self._col_direction = np.array(orientation_patient[3:6])  # 열 방향 (Y 증가)
        
        # 슬라이스 방향 = 행 × 열 (외적)
        self._slice_direction = np.cross(self._row_direction, self._col_direction)
        
        if verbose:
            print(f"\n[DICOM 좌표 정보]")
            print(f"  Row Direction (X): {self._row_direction}")
            print(f"  Col Direction (Y): {self._col_direction}")
            print(f"  Slice Direction (Z): {self._slice_direction}")
            print(f"  Pixel Spacing: {pixel_spacing}")
        
        # 모든 슬라이스 로드 및 정렬
        slices_data = []
        for f in dicom_files:
            ds = pydicom.dcmread(f)
            img_pos = [float(x) for x in ds.ImagePositionPatient]
            slices_data.append({
                'dataset': ds,
                'position': np.array(img_pos),
                'slice_location': np.dot(img_pos, self._slice_direction)  # 슬라이스 방향 투영
            })
        
        # 슬라이스 위치로 정렬
        slices_data.sort(key=lambda x: x['slice_location'])
        
        # 첫 번째 슬라이스 위치 (원점)
        self._image_position = slices_data[0]['position']
        
        # 슬라이스 간격 계산
        if len(slices_data) > 1:
            slice_spacing = abs(slices_data[1]['slice_location'] - slices_data[0]['slice_location'])
        else:
            slice_spacing = float(first_ds.get('SliceThickness', 1.0))
        
        if verbose:
            print(f"  Image Position (Origin): {self._image_position}")
            print(f"  Slice Spacing: {slice_spacing}")
        
        # 볼륨 배열 생성
        num_slices = len(slices_data)
        volume = np.zeros((num_slices, rows, cols), dtype=np.float32)
        
        # Rescale Slope/Intercept
        rescale_slope = float(first_ds.get('RescaleSlope', 1.0))
        rescale_intercept = float(first_ds.get('RescaleIntercept', 0.0))
        
        # 슬라이스 데이터 채우기
        for i, slice_info in enumerate(slices_data):
            ds = slice_info['dataset']
            pixel_array = ds.pixel_array.astype(np.float32)
            # HU 변환
            volume[i, :, :] = pixel_array * rescale_slope + rescale_intercept
        
        self._volume = volume
        
        # 메타데이터 저장
        # 주의: spacing은 (X, Y, Z) 순서 = (col_spacing, row_spacing, slice_spacing)
        self._metadata = {
            "patient_name": str(first_ds.get('PatientName', 'Unknown')),
            "study_date": str(first_ds.get('StudyDate', 'Unknown')),
            "modality": str(first_ds.get('Modality', 'CT')),
            "rows": rows,
            "columns": cols,
            "slices": num_slices,
            "pixel_spacing": pixel_spacing,  # [row, col] = [Y, X]
            "slice_thickness": slice_spacing,
            "spacing_xyz": (pixel_spacing[1], pixel_spacing[0], slice_spacing),  # (X, Y, Z)
            "origin_xyz": tuple(self._image_position),  # (X, Y, Z)
            "orientation": "LPS",
            "row_direction": self._row_direction,
            "col_direction": self._col_direction,
            "slice_direction": self._slice_direction,
            "rescale_slope": rescale_slope,
            "rescale_intercept": rescale_intercept,
        }
        
        if verbose:
            print(f"\n[완료] 볼륨 로딩 완료")
            print(f"  Shape: {volume.shape} (Z, Y, X)")
            print(f"  Spacing (X, Y, Z): {self._metadata['spacing_xyz']}")
            print(f"  Origin (X, Y, Z): {self._metadata['origin_xyz']}")
            print(f"  HU 범위: {volume.min():.1f} ~ {volume.max():.1f}")
        
        return self._volume
    
    def _get_dicom_files(self) -> List[Path]:
        """DICOM 파일 목록 가져오기"""
        dicom_files = []
        
        for f in self.dicom_folder.iterdir():
            if f.is_file():
                # .dcm 확장자 또는 확장자 없는 파일
                if f.suffix.lower() == '.dcm' or f.suffix == '':
                    try:
                        # DICOM 파일인지 확인
                        ds = pydicom.dcmread(f, stop_before_pixels=True)
                        if hasattr(ds, 'PixelData') or hasattr(ds, 'Rows'):
                            dicom_files.append(f)
                    except:
                        pass
        
        return sorted(dicom_files)
    
    # ----------------------------
    # Getters
    # ----------------------------
    def get_volume(self) -> np.ndarray:
        """HU 볼륨 반환 (Z, Y, X)"""
        if self._volume is None:
            raise RuntimeError("load()를 먼저 실행하세요")
        return self._volume
    
    def get_metadata(self) -> Dict[str, Any]:
        """메타데이터 딕셔너리 반환"""
        if not self._metadata:
            raise RuntimeError("load()를 먼저 실행하세요")
        return self._metadata
    
    def get_spacing(self) -> Tuple[float, float, float]:
        """
        Spacing 반환 (X, Y, Z) in mm
        """
        return self._metadata["spacing_xyz"]
    
    def get_origin(self) -> Tuple[float, float, float]:
        """
        Origin 반환 (X, Y, Z) in mm
        """
        return self._metadata["origin_xyz"]
    
    def get_orientation(self) -> str:
        """좌표계 방향 반환 (LPS)"""
        return self._metadata.get("orientation", "LPS")
    
    # ----------------------------
    # 좌표 변환 (vtk.js와 동일)
    # ----------------------------
    def index_to_physical(self, idx_zyx: np.ndarray, use_origin: bool = False) -> np.ndarray:
        """
        볼륨 인덱스 (Z, Y, X)를 물리 좌표 (X, Y, Z) mm로 변환
        
        NumPy 브로드캐스팅을 사용한 벡터화된 구현으로 빠른 처리 성능을 제공합니다.
        
        vtk.js 호환 (use_origin=False, 기본값):
        - Origin을 무시하고 (0,0,0)에서 시작
        - P = col*colSpacing*rowDir + row*rowSpacing*colDir + slice*sliceSpacing*sliceDir
        
        표준 DICOM (use_origin=True):
        - ImagePositionPatient(Origin) 적용
        - P = Origin + col*colSpacing*rowDir + row*rowSpacing*colDir + slice*sliceSpacing*sliceDir
        
        Parameters:
        -----------
        idx_zyx : np.ndarray (N, 3)
            볼륨 인덱스 배열 [(z, y, x), ...]  = [(slice, row, col), ...]
        use_origin : bool
            Origin 적용 여부 (vtk.js 호환을 위해 기본값 False)
            
        Returns:
        --------
        np.ndarray (N, 3)
            물리 좌표 배열 [(x, y, z), ...] in mm
        """
        if self._volume is None:
            raise RuntimeError("load()를 먼저 실행하세요")
        
        spacing = self._metadata["spacing_xyz"]  # (X, Y, Z) = (col, row, slice)
        col_spacing = spacing[0]
        row_spacing = spacing[1]
        slice_spacing = spacing[2]
        
        # 인덱스 분리 (브로드캐스팅을 위해 (N, 1) 형태로 유지)
        slice_idx = idx_zyx[:, 0:1].astype(np.float64)  # Z
        row_idx = idx_zyx[:, 1:2].astype(np.float64)    # Y
        col_idx = idx_zyx[:, 2:3].astype(np.float64)    # X
        
        # 방향 벡터들을 (1, 3) 형태로
        row_dir = self._row_direction.reshape(1, 3)
        col_dir = self._col_direction.reshape(1, 3)
        slice_dir = self._slice_direction.reshape(1, 3)
        
        # vtk.js 호환: Origin 무시 (0, 0, 0)에서 시작
        if use_origin:
            origin = self._image_position.reshape(1, 3)
        else:
            origin = np.zeros((1, 3))
        
        # 브로드캐스팅으로 벡터화된 계산
        physical_coords = (
            origin +
            (col_idx * col_spacing) * row_dir +
            (row_idx * row_spacing) * col_dir +
            (slice_idx * slice_spacing) * slice_dir
        )
        
        return physical_coords
    
    def physical_to_index(self, pts_xyz: np.ndarray) -> np.ndarray:
        """
        물리 좌표 (X, Y, Z) mm를 볼륨 인덱스 (Z, Y, X)로 변환
        """
        if self._volume is None:
            raise RuntimeError("load()를 먼저 실행하세요")
        
        spacing = self._metadata["spacing_xyz"]
        col_spacing = spacing[0]
        row_spacing = spacing[1]
        slice_spacing = spacing[2]
        
        # 원점에서 상대 위치
        relative = pts_xyz - self._image_position
        
        # 각 방향으로 투영
        col_idx = np.dot(relative, self._row_direction) / col_spacing
        row_idx = np.dot(relative, self._col_direction) / row_spacing
        slice_idx = np.dot(relative, self._slice_direction) / slice_spacing
        
        return np.stack([slice_idx, row_idx, col_idx], axis=1)
    
    # ----------------------------
    # 통계
    # ----------------------------
    def get_statistics(self) -> Dict[str, float]:
        """HU 통계 반환"""
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
