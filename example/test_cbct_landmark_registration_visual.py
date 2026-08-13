"""CBCT 랜드마크 → 기존 정합 변환 행렬 적용 → 시각 검증 스크립트.

목적
----
cbctLandmark가 산출한 랜드마크(LPS mm)를 기존 FaceScan+CBCT 정합이 만드는
변환 행렬에 태운 뒤, 3D 창에서 해부학적 위치를 눈으로 확인한다.

좌표계·표시 정책 (진단으로 확정된 사실)
--------------------------------------
- T_cbct = CBCTFaceScanAlignmentPipeline.run() 반환 행렬. 랜드마크(LPS)에 그대로
  적용하면 CBCT 뼈 표면 위에 1~3mm 수준으로 앉는다 (중간 LPS→RAS 변환 불필요).
- FaceScan 은 이 환자 PLY 에 대해 aligner 의 T_face 가 제대로 동작하지 않아,
  기본(--face-transform fit)은 CBCT 피부 표면(씬)에 대한 경험적 강체 정합
  (FPFH RANSAC + ICP) 으로 배치한다. t_face/raw 는 디버그용.
- 표시용 CBCT 메쉬는 뼈/치아가 보이도록 bone window(기본 HU 200, --display-hu) 로
  추출하고, 정합 ICP 자체는 기존 그대로 피부 표면(-200, --hu-threshold) 사용.

데이터
------
- 루트 ``cbctdata/`` 와 ``example/data/FaceScan/이판임 - 페이스 스캔`` 은 같은 환자.
- 선행 참고자료(루트 PDF nnU-Net 파이프라인 문서)와 동일 방식: CBCT 정합
  변환행렬을 두부계측 랜드마크(RCo·LCo·B·Gn·Pog)에 적용해 시각화.

사용 예시
---------
venv\\Scripts\\python.exe example/test_cbct_landmark_registration_visual.py
venv\\Scripts\\python.exe example/test_cbct_landmark_registration_visual.py --no-cbct-surface
venv\\Scripts\\python.exe example/test_cbct_landmark_registration_visual.py --inference

의존성
------
mediapipe, open3d, pyvista, scikit-image, scipy, pydicom, scikit-learn, tensorflow.

수정사항
(26.8.12. SAM 매쉬 생성하도록 코드 추가)
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

try:
    import pyvista as pv

    PYVISTA_OK = True
except ImportError:
    PYVISTA_OK = False

DEFAULT_DICOM = os.path.join(REPO_ROOT, "cbctdata")
DEFAULT_MRK = os.path.join(
    REPO_ROOT, "output", "cbct_landmark", "cbctdata.nii", "cbctdata_merged.mrk.json"
)
DEFAULT_FACE = os.path.join(
    REPO_ROOT,
    "example",
    "data",
    "FaceScan",
    "이판임 - 페이스 스캔",
    "이판임_이판임_20251010131426747",
    "Smile.ply",
)
DEFAULT_SMILE_ARCH = os.path.join(REPO_ROOT, "example", "data", "ios", "smileArch.stl")
DEFAULT_MODELS = os.path.normpath(
    os.path.join(REPO_ROOT, "..", "dental-cbct-landmark", "models")
)
DEFAULT_LANDMARKS = "Gn,Pog,B,RCo,LCo"

# 정중선 랜드마크(빨강) — 나머지는 측방(파랑)
MIDLINE = {"Gn", "Pog", "B", "N", "ANS", "Me", "PNS"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CBCT 랜드마크 정합 시각 검증")
    p.add_argument("--dicom-folder", default=DEFAULT_DICOM, help="CBCT DICOM 폴더")
    p.add_argument(
        "--mrk", default=DEFAULT_MRK, help="기존 산출물 .mrk.json (--inference 시 무시)"
    )
    p.add_argument(
        "--inference", action="store_true", help="랜드마크 라이브 추론 (~215초)"
    )
    p.add_argument(
        "--models-dir", default=DEFAULT_MODELS, help="가중치 루트 (--inference 시)"
    )
    p.add_argument(
        "--landmarks", default=DEFAULT_LANDMARKS, help="랜드마크 목록 (--inference 시)"
    )
    p.add_argument("--facescan", default=DEFAULT_FACE, help="FaceScan(smile) 메시 경로")
    p.add_argument(
        "--smile-arch", default=DEFAULT_SMILE_ARCH, help="IOS smileArch 메시 경로"
    )
    # [수정 전] p.add_argument("--face-transform", choices=["fit", "t_face", "raw"], default="fit", ...)
    # [수정 후]
    p.add_argument(
        "--face-transform",
        choices=["t_face", "raw"],
        default="t_face",
        help="FaceScan 배치 방식: t_face=파이프라인 정규 행렬(4번 결과) 적용 (기본), raw=원본 프레임 디버그용",
    )
    p.add_argument("--no-facescan", action="store_true", help="FaceScan 중첩 생략")
    p.add_argument(
        "--no-cbct-surface", action="store_true", help="CBCT 뼈 메쉬 표시 생략(속도)"
    )
    p.add_argument(
        "--no-gui", action="store_true", help="창을 띄우지 않고 수치 요약만 출력"
    )
    p.add_argument(
        "--display-hu", type=float, default=200.0, help="표시용 뼈 메쉬 HU 임계값"
    )
    p.add_argument("--mesh-step", type=int, default=4, help="marching cubes step_size")
    p.add_argument(
        "--hu-threshold",
        type=float,
        default=-200.0,
        help="정합 ICP용 피부 표면 HU 임계값",
    )
    p.add_argument(
        "--sam-mesh",
        type=str,
        default=r"C:\Users\jw.go\projects\pyNeo3DLib\example\data\smilearch_bow.stl",
        help=r"C:\Users\jw.go\projects\pyNeo3DLib\example\data\smilearch_bow.stl"
    )
    return p.parse_args()


def stage_facescan(face_path: str) -> str:
    """FaceScan을 ASCII 임시 폴더로 복사해 새 경로를 반환.

    Windows에서 cv2.imread/open3d가 비ASCII(한글) 경로를 읽지 못하는
    문제를 우회한다 (windows_path_staging 과 같은 계열의 문제).
    """
    try:
        face_path.encode("ascii")
        return face_path
    except UnicodeEncodeError:
        pass
    stage = os.path.join(tempfile.gettempdir(), "neo_facescan_staging")
    os.makedirs(stage, exist_ok=True)
    src_dir = os.path.dirname(os.path.abspath(face_path))
    dst = os.path.join(stage, os.path.basename(face_path))
    shutil.copy2(face_path, dst)
    for name in os.listdir(src_dir):
        if name.lower().endswith((".png", ".mtl")):
            shutil.copy2(os.path.join(src_dir, name), os.path.join(stage, name))
    print(f"[stage] FaceScan을 ASCII 경로로 복사: {dst}")
    return dst


def compute_registration_matrices(
    args, face_path: str
) -> tuple[np.ndarray, np.ndarray]:
    """T_face(FaceScan→smileArch), T_cbct(CBCT LPS→씬 프레임) 반환.

    registration.py:433-437 이 호출하는 것과 동일한 함수를 직접 호출한다.
    """
    from pyNeo3DLib.cbctRegistration.core.alignment_pipeline import (
        CBCTFaceScanAlignmentPipeline,
    )
    from pyNeo3DLib.faceRegisration.faceSmileGuideAligner import FaceSmileGuideAligner

    t0 = time.time()
    print("\n[1/2] FaceSmileGuideAligner.align() ...")
    t_face = FaceSmileGuideAligner().align(
        face_scan_path=face_path,
        smile_arch_path=args.smile_arch,
        visualize=False,
    )
    print(f"      완료 ({time.time() - t0:.1f}s)")

    t0 = time.time()
    print("[2/2] CBCTFaceScanAlignmentPipeline.run() ...")
    pipeline = CBCTFaceScanAlignmentPipeline(
        config=None,
        random_seed=42,
        visualize=False,
        verbose=True,
        mesh_hu_threshold=args.hu_threshold,
        mesh_step_size=args.mesh_step,
    )
    t_cbct = pipeline.run(
        dicom_folder=args.dicom_folder,
        facescan_path=face_path,
        facescan_laminate_result=t_face,
    )
    print(f"      완료 ({time.time() - t0:.1f}s)")
    return t_face, t_cbct


def load_or_predict_landmarks(args) -> dict[str, np.ndarray]:
    """랜드마크 dict {이름: (3,) LPS mm} 반환."""
    if args.inference:
        from pyNeo3DLib.cbctLandmark.dicom_pipeline import predict_landmarks_from_dicom

        lm_list = [s.strip() for s in args.landmarks.split(",") if s.strip()]
        t0 = time.time()
        print("\n[landmark] predict_landmarks_from_dicom (~215s) ...")
        raw = predict_landmarks_from_dicom(
            dicom_folder=args.dicom_folder,
            models_dir=args.models_dir,
            landmarks=lm_list,
            output_dir=os.path.join(REPO_ROOT, "output", "cbct_landmark"),
        )
        print(f"           완료 ({time.time() - t0:.1f}s)")
        return {k: np.array([v["x"], v["y"], v["z"]]) for k, v in raw.items()}

    from pyNeo3DLib.cbctLandmark.markups import load_mrk_landmarks

    if not os.path.isfile(args.mrk):
        raise FileNotFoundError(
            f"mrk.json 없음: {args.mrk}\n"
            "  → --inference 플래그로 추론 실행 또는 --mrk 로 경로 지정"
        )
    return load_mrk_landmarks(args.mrk)


def extract_cbct_mesh_lps(args, hu_threshold: float, target: int):
    """CBCT 표면 메쉬를 환자 LPS 좌표계로 추출.

    marching cubes 산출물은 vox origin-0 공간이므로
    index_to_physical(use_origin=True) 로 LPS 변환한다.
    """
    from pyNeo3DLib.cbctRegistration.processing.cbct_processor import CBCTProcessor

    proc = CBCTProcessor()
    t0 = time.time()
    print(f"\n[CBCT] DICOM 로드 + 표면 메쉬 (HU {hu_threshold:g}) ...")
    loader = proc.load_dicom(args.dicom_folder, verbose=False)
    mesh = proc.generate_mesh_from_volume(
        loader,
        hu_threshold=hu_threshold,
        step_size=args.mesh_step,
        target_triangles=target,
        verbose=False,
    )
    v = np.asarray(mesh.vertices)
    cs, rs, ss = loader.get_spacing()
    idx_zyx = np.stack([v[:, 2] / ss, v[:, 1] / rs, v[:, 0] / cs], axis=1)
    lps = loader.index_to_physical(idx_zyx, use_origin=False)  # vox origin-0 → 환자 LPS

    import open3d as o3d

    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(lps)
    out.triangles = mesh.triangles
    print(
        f"       완료 ({time.time() - t0:.1f}s), "
        f"vertices={len(out.vertices):,}, faces={len(out.triangles):,}"
    )
    return out


def load_facescan(args, face_path: str, t_face) -> tuple[np.ndarray, np.ndarray]:
    """FaceScan (verts, faces) 반환. 
    파이프라인의 정규 행렬(t_face)을 적용하며, MediaPipe 실패 원인을 디버깅합니다."""
    if args.no_facescan:
        return None, None
    if not os.path.isfile(face_path):
        print(f"[warn] FaceScan 없음: {face_path}")
        return None, None

    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(face_path)
    verts = np.asarray(mesh.vertices)

    # [디버깅 1] 스케일 체크 (단위가 mm가 아닌 m로 들어왔는지 확인)
    bbox_size = verts.max(axis=0) - verts.min(axis=0)
    print(f"\n[디버그] FaceScan 원본 Bounding Box 크기: X={bbox_size[0]:.1f}, Y={bbox_size[1]:.1f}, Z={bbox_size[2]:.1f} mm")
    if bbox_size.max() < 10.0:
        print("🚨 [경고] FaceScan 메쉬 크기가 비정상적으로 작습니다! (m 단위일 가능성 높음)")

    # [디버깅 2] 정합 행렬 적용
    if args.face_transform == "t_face":
        print("\n[FaceScan] 파이프라인 정규 변환 행렬(4번: Smiling Face → Template) 적용 중...")
        f = np.asarray(t_face)
    else:
        f = np.eye(4)

    # 단일 좌표계 원칙에 따른 4x4 행렬 내적 연산 (단 한 번만 수행)
    verts_homo = np.hstack([verts, np.ones((len(verts), 1))])
    transformed_verts = (f @ verts_homo.T).T[:, :3]

    return transformed_verts, np.asarray(mesh.triangles)


def to_polydata(verts: np.ndarray, tris: np.ndarray):
    pv_faces = np.hstack([[3, *f] for f in tris]).ravel()
    return pv.PolyData(verts, pv_faces)


def print_summary(lm_common, face_verts, skin_verts):
    from scipy.spatial import cKDTree

    print("\n── 변환 후 랜드마크 (씬 프레임, mm) ──")
    for label, xyz in lm_common.items():
        print(f"  {label:<6} x={xyz[0]:8.3f}  y={xyz[1]:8.3f}  z={xyz[2]:8.3f}")

    if skin_verts is not None and len(skin_verts):
        d, _ = cKDTree(skin_verts).query(np.stack(list(lm_common.values())))
        print("\n── 주 검증: CBCT 피부 표면까지 거리 (1~3mm 수준이면 정상) ──")
        for label, x in zip(lm_common, d):
            print(f"  {label:<6} {x:7.2f} mm")
    if face_verts is not None and len(face_verts):
        d, _ = cKDTree(face_verts).query(np.stack(list(lm_common.values())))
        print("\n── 참고: FaceScan 표면까지 거리 ──")
        for label, x in zip(lm_common, d):
            print(f"  {label:<6} {x:7.2f} mm")


def visualize(lm_common, cbct_mesh_common, face_verts, face_tris, args) -> None:
    if not PYVISTA_OK:
        print("[skip] pyvista 없음. 수치 요약으로 대신합니다.")
        return
    if args.no_gui:
        print("[skip] --no-gui. 창을 띄우지 않습니다.")
        return

    plotter = pv.Plotter(window_size=(1400, 900))
    plotter.set_background("white")

    # 1. FaceScan (t_face가 적용되어 SmileArch 공간에 위치)
    if face_verts is not None:
        plotter.add_mesh(
            to_polydata(face_verts, face_tris),
            color="peachpuff",
            opacity=0.4,
            label="FaceScan (t_face applied)",
        )

    # 2. CBCT bones (t_cbct 단독 적용으로 FaceScan과 동일 공간에 위치)
    if cbct_mesh_common is not None:
        verts = np.asarray(cbct_mesh_common.vertices)
        tris = np.asarray(cbct_mesh_common.triangles)
        plotter.add_mesh(
            to_polydata(verts, tris),
            color="lightgray",
            opacity=0.6,
            label="CBCT bones (t_cbct applied)",
        )

    # 3. 랜드마크 표시
    for label, xyz in lm_common.items():
        pts = xyz.reshape(1, 3)
        color = "red" if label in MIDLINE else "blue"
        plotter.add_points(
            pts, color=color, point_size=18, render_points_as_spheres=True
        )
        plotter.add_point_labels(
            pts,
            [label],
            font_size=14,
            text_color="black",
            show_points=False,
            shape_opacity=0.6,
        )

    # 4. SAM Reference 메쉬 (기준 템플릿 공간)
    sam_path = args.sam_mesh
    if os.path.isfile(sam_path):
        sam_mesh = pv.read(sam_path)
        plotter.add_mesh(
            sam_mesh,
            color="gold",
            opacity=0.3,
            style="surface",
            label="SAM Bow (Reference)",
        )

    plotter.add_axes()
    plotter.add_legend(size=(0.25, 0.25), loc="upper right")
    plotter.add_title("CBCT Landmarks & FaceScan Registration Visualizer", font_size=14)
    plotter.camera_position = "iso"
    print("\npyvista 창이 열립니다. 창을 닫으면 종료됩니다.")
    plotter.show()


def main() -> int:
    args = parse_args()
    print("── 입력 ──")
    for key, val in vars(args).items():
        print(f"  {key:<16}: {val}")

    face_path = stage_facescan(args.facescan)
    t_face, t_cbct = compute_registration_matrices(args, face_path)

    landmarks_lps = load_or_predict_landmarks(args)
    print(f"\n로드된 랜드마크: {len(landmarks_lps)}개 ({list(landmarks_lps)})")

    from pyNeo3DLib.cbctRegistration.utils.common import (
        apply_transform,
        apply_transform_to_points,
    )

    # 💡 핵심 수정: t_cbct는 이미 완결된 변환 행렬이므로 t_face를 중복 곱하지 않고 t_cbct만 적용합니다.
    # CBCT 피부 표면 (-200 HU)
    skin_lps = extract_cbct_mesh_lps(args, args.hu_threshold, 30_000)
    skin_scene = apply_transform_to_points(np.asarray(skin_lps.vertices), t_cbct)

    # CBCT 뼈 표면 (200 HU)
    cbct_mesh_common = None
    if not args.no_cbct_surface:
        bone_lps = extract_cbct_mesh_lps(args, args.display_hu, 200_000)
        cbct_mesh_common = apply_transform(bone_lps, t_cbct)

    # 랜드마크
    labels = list(landmarks_lps.keys())
    pts_lps = np.stack([landmarks_lps[l] for l in labels])

    # 1. 메쉬 생성과 완벽히 동일한 방식으로 DICOM 공간 정보 로드
    from pyNeo3DLib.cbctRegistration.processing.cbct_processor import CBCTProcessor
    proc = CBCTProcessor()
    
    print("\n[Landmark] 실제 DICOM Origin을 구하기 위해 볼륨 정보를 로드합니다...")
    loader = proc.load_dicom(args.dicom_folder, verbose=False)
    
    # 2. 가장 안전한 Origin 추출 방법: 
    # Index [0, 0, 0] 에 해당하는 물리적 좌표를 use_origin=True로 변환하여 얻어냄
    true_origin = loader.index_to_physical(np.array([[0, 0, 0]]), use_origin=True)[0]
    
    # 3. 랜드마크 좌표에서 '진짜' Origin 오프셋 차감 (use_origin=False 공간과 일치시킴)
    pts_lps_no_origin = pts_lps - true_origin

    # 4. 변환 행렬 적용
    pts_common = apply_transform_to_points(pts_lps_no_origin, t_cbct)
    lm_common = {l: pts_common[i] for i, l in enumerate(labels)}

    # FaceScan (t_face 적용)
    face_verts, face_tris = load_facescan(args, face_path, t_face)

    print_summary(lm_common, face_verts, skin_scene)
    visualize(lm_common, cbct_mesh_common, face_verts, face_tris, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
