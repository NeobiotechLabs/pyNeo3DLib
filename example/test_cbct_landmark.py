"""CBCT 랜드마크 파이프라인 콘솔 검증 스크립트.

CBCT DICOM 폴더 → dcm2nii 변환(.nii.gz) → 랜드마크 추론 → 좌표 출력까지
전체 흐름을 콘솔에서 확인한다.

사용 예시
---------
venv\\Scripts\\python.exe example/test_cbct_landmark.py --dicom-folder "C:\\data\\patient_cbct"

옵션
----
--dicom-folder   CBCT DICOM 폴더 (필수)
--models-dir     가중치 루트 (기본: ../dental-cbct-landmark/models)
--landmarks      쉼표 구분 (기본: Gn,Pog,B,RCo,LCo)
--output-dir     산출물 폴더 (기본: <repo>/output/cbct_landmark)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

DEFAULT_MODELS_DIR = os.path.normpath(
    os.path.join(REPO_ROOT, "..", "dental-cbct-landmark", "models")
)
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "output", "cbct_landmark")
DEFAULT_LANDMARKS = "Gn,Pog,B,RCo,LCo"


def main() -> int:
    parser = argparse.ArgumentParser(description="CBCT 랜드마크 파이프라인 콘솔 검증")
    parser.add_argument("--dicom-folder", required=True, help="CBCT DICOM 폴더 경로")
    parser.add_argument(
        "--models-dir", default=DEFAULT_MODELS_DIR, help="가중치 루트 폴더"
    )
    parser.add_argument(
        "--landmarks", default=DEFAULT_LANDMARKS, help="랜드마크 목록 (쉼표 구분)"
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR, help="산출물 저장 폴더"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    landmarks = [s.strip() for s in args.landmarks.split(",") if s.strip()]

    print("── 입력 ──")
    print(f"  DICOM 폴더 : {args.dicom_folder}")
    print(f"  모델 폴더  : {args.models_dir}")
    print(f"  랜드마크   : {', '.join(landmarks)}")
    print(f"  출력 폴더  : {args.output_dir}")

    from pyNeo3DLib.cbctLandmark.dicom_pipeline import predict_landmarks_from_dicom

    started = time.time()
    results = predict_landmarks_from_dicom(
        dicom_folder=args.dicom_folder,
        models_dir=args.models_dir,
        landmarks=landmarks,
        output_dir=args.output_dir,
    )
    elapsed = time.time() - started

    print("\n── 추론 결과 (LPS mm) ──")
    for lm in landmarks:
        if lm in results:
            c = results[lm]
            print(f"  {lm:<10} x={c['x']:8.3f}  y={c['y']:8.3f}  z={c['z']:8.3f}")
        else:
            print(f"  {lm:<10} (탐색 실패 또는 결과 없음)")
    print(f"\n소요 시간: {elapsed:.1f}초")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
