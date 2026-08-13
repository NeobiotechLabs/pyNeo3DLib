"""
CBCT DICOM 폴더 → NIfTI 변환 → 랜드마크 추론 통합 파이프라인.

pyNeo3DLib 통합용 연결(glue) 모듈. vendored 원본(cbctLandmark 내부 파일, dicomNifti)은 수정하지 않고
이 파일에서만 두 패키지를 조합한다.

흐름
----
DICOM 폴더
  → dicomNifti.dicom_folder_to_pre_inference_nifti  (.nii.gz 생성, Windows 비ASCII 경로 안전)
  → cbctLandmark.predict                            (ALI 에이전트 추론)
  → 랜드마크 좌표 dict (LPS mm)

좌표계
------
- 출력 랜드마크는 **LPS mm**. ``patient_origin=True``(기본)면 DICOM ImagePositionPatient 기준
  환자 좌표계로 나와 cbctRegistration 등 기존 모듈과 좌표계가 일치한다.
- pyNeo3DLib 표준 축 방향은 RAS 이므로, 다른 스캔(IOS/FaceScan)과 합칠 때는
  X·Y 부호 반전(LPS→RAS) 변환이 필요하다. (이번 단계에서는 미적용)

사용 예시
---------
from pyNeo3DLib.cbctLandmark.dicom_pipeline import predict_landmarks_from_dicom

results = predict_landmarks_from_dicom(
    dicom_folder=r"C:\\data\\patient_cbct",
    models_dir=r"C:\\Users\\jw.go\\projects\\dental-cbct-landmark\\models",
    landmarks=["Gn", "Pog", "B", "RCo", "LCo"],
)
"""

from __future__ import annotations

import logging
import os

from .pipeline import predict

logger = logging.getLogger(__name__)

__all__ = ["predict_landmarks_from_dicom"]


def predict_landmarks_from_dicom(
    dicom_folder: str,
    models_dir: str,
    landmarks: list[str],
    output_dir: str | None = None,
    work_dir: str | None = None,
    series_uid: str | None = None,
    patient_origin: bool = True,
    device: str | None = None,
    verbose: bool = True,
) -> dict[str, dict[str, float]]:
    """CBCT DICOM 폴더에서 랜드마크 좌표를 추론한다.

    Parameters
    ----------
    dicom_folder : str
        CBCT DICOM 파일들이 들어 있는 폴더.
    models_dir : str
        학습 가중치 루트. ``models/<팩>/<랜드마크>/<스케일(1, 0-3)>/*.pth`` 구조.
    landmarks : List[str]
        추론할 랜드마크 이름 목록 (예: ["Gn", "Pog", "B", "RCo", "LCo"]).
    output_dir : str, optional
        추론 산출물(.mrk.json 등) 저장 폴더. None이면 볼륨 파일 위치에 저장.
    work_dir : str, optional
        중간 산출물(.nii.gz) 저장 폴더. None이면 output_dir, 그것도 없으면 시스템 임시 폴더.
    series_uid : str, optional
        DICOM 시리즈 지정. None이면 자동 선택(최다 인스턴스 시리즈).
    patient_origin : bool
        True(기본): NIfTI 원점을 DICOM ImagePositionPatient로 → 환자 LPS 좌표.
        False: 원점 (0,0,0) → 볼륨 상대 좌표.
    device : str, optional
        추론 장치 ("cpu" / "cuda"). None이면 자동 선택.
    verbose : bool
        진행 로그 출력 여부.

    Returns
    -------
    Dict[str, Dict[str, float]]
        ``{"Gn": {"x": ..., "y": ..., "z": ...}, ...}`` — LPS mm.
        탐색에 실패한 랜드마크는 결과에서 제외된다.
    """
    from pyNeo3DLib.dicomNifti import dicom_folder_to_pre_inference_nifti

    if not os.path.isdir(dicom_folder):
        raise NotADirectoryError(f"DICOM 폴더를 찾을 수 없습니다: {dicom_folder}")

    base_dir = work_dir or output_dir
    if base_dir:
        os.makedirs(base_dir, exist_ok=True)
        nii_dir = base_dir
    else:
        import tempfile

        nii_dir = os.path.join(tempfile.gettempdir(), "pyNeo3DLandmark")
        os.makedirs(nii_dir, exist_ok=True)

    folder_name = os.path.basename(os.path.normpath(dicom_folder))
    nii_path = os.path.join(nii_dir, f"{folder_name}.nii.gz")

    if verbose:
        logger.info("[1/2] DICOM → NIfTI 변환: %s", dicom_folder)
    result = dicom_folder_to_pre_inference_nifti(
        dicom_folder,
        nii_path,
        series_uid=series_uid,
        patient_origin=patient_origin,
        progress=(print if verbose else None),
    )
    if verbose:
        logger.info("      변환 완료: %s", result.nifti_path)

    if verbose:
        logger.info("[2/2] 랜드마크 추론: %s", ", ".join(landmarks))
    coords = predict(
        volume=str(result.nifti_path),
        models_dir=models_dir,
        landmarks=landmarks,
        output_dir=output_dir,
        device=device,
        verbose=verbose,
    )
    return coords
