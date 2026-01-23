
import copy
import argparse
import numpy as np
import open3d as o3d
from pathlib import Path
import time

# 임포트 경로 문제 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from lip_preprocessing import visualize_lip_extraction

def align_face_scan_to_smile_arch(sample_id: int):
    # 1. Source 데이터 로드 (Face Scan -> Upper Teeth)
    print(f"Loading Face Scan Sample {sample_id}...")
    folder_path = f'./3dmodel/facescan/sample_{sample_id}'
    
    # 시각화 없이 데이터만 추출
    result = visualize_lip_extraction(folder_path, visualize=False)
    
    if not result or len(result['filtered_vertices']) == 0:
        print("Error: Failed to extract upper teeth vertices.")
        return

    # Source PointCloud 생성
    source_vertices = result['filtered_vertices']
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_vertices)
    source.paint_uniform_color([1.0, 1.0, 0.0])  # 노란색 (Source)
    
    print(f"Source (Upper Teeth): {len(source.points)} points")

    # 2. Target 데이터 로드 (Smile Arch)
    target_path = './3dmodel/smile_arch_half.stl'
    print(f"Loading Target Mesh: {target_path}")
    target_mesh = o3d.io.read_triangle_mesh(target_path)
    target_mesh.compute_vertex_normals()
    target_mesh.paint_uniform_color([0.7, 0.7, 0.7])  # 회색 (Target)
    
    # ICP를 위해 Target도 PointCloud로 변환 (샘플링)
    target_pcd = target_mesh.sample_points_poisson_disk(number_of_points=5000)
    
    # 3. 초기 정렬 (Initial Alignment)
    # Source 특징점 계산
    src_points = np.asarray(source.points)
    src_center = np.mean(src_points, axis=0)
    src_min = np.min(src_points, axis=0)
    src_max = np.max(src_points, axis=0)
    
    # Target 특징점 계산
    tgt_points = np.asarray(target_mesh.vertices)
    tgt_center = np.mean(tgt_points, axis=0)
    tgt_min = np.min(tgt_points, axis=0)
    tgt_max = np.max(tgt_points, axis=0)
    
    print("\nInitial Alignment Strategy:")
    print("  1. Center X of Source -> Center X of Target (0)")
    print("  2. Max Y of Source -> Max Y of Target")
    print(f"     Source Max Y: {src_max[1]:.2f} -> Target Max Y: {tgt_max[1]:.2f}")
    print("  3. Min Z of Source -> Min Z of Target")
    print(f"     Source Min Z: {src_min[2]:.2f} -> Target Min Z: {tgt_min[2]:.2f}")

    # 이동 벡터 계산
    translation = np.zeros(3)
    
    # X축: 입술 중점(데이터의 중심)을 Target의 0으로
    translation[0] = -src_center[0]  
    
    # Y축: 가장 앞쪽(+Y) 점 일치
    translation[1] = tgt_max[1] - src_max[1]
    
    # Z축: 가장 아래쪽(-Z) 점 일치 (치아의 절단연 부분)
    translation[2] = tgt_min[2] - src_min[2]

    # 전체 변환 행렬 누적용 (4x4 단위행렬)
    global_transform = np.eye(4)
    
    # 이동 행렬 생성 및 적용
    T = np.eye(4)
    T[:3, 3] = translation
    
    print(f"Applying Translation: {translation}")
    source.transform(T)
    global_transform = T @ global_transform  # 누적
    
    # 4. 시각화 및 ICP 애니메이션
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Face Scan Alignment (ICP Animation)", width=1200, height=800)
    
    vis.add_geometry(source)
    # Target은 Mesh 상태로 보여줌 (더 깔끔함)
    vis.add_geometry(target_mesh)
    
    # 카메라 설정: 정면 뷰
    ctr = vis.get_view_control()
    ctr.set_front([0, 1, 0])
    ctr.set_up([0, 0, 1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_zoom(0.6)
    
    # Point-to-Plane ICP를 위해 법선 계산 (Source)
    # 주변 30개 점을 참고하여 법선 추정
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30))
    
    # ICP 설정
    threshold = 1.0  # 초기 거리 임계값 (5mm)
    trans_init = np.identity(4)
    
    # Point-to-Plane 방식 (치아 아치 정렬에 유리) + Robust Kernel (안겹치는 영역 무시)
    # Welsch Loss 등과 유사하게 Outlier(비중첩 영역)에 가중치를 낮게 주는 TukeyLoss 사용
    # k=0.5: 0.5mm 이상 떨어진 점들은 가중치가 급격히 줄어듦 (강력한 Outlier 제거)
    loss = o3d.pipelines.registration.TukeyLoss(k=0.5)
    estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
    
    print("\nStarting ICP Animation (Robust Point-to-Plane with Tukey Loss)...")
    
    # 애니메이션 루프 (ICP를 1 iteration씩 수행)
    for i in range(60):
        # 단계적 Threshold 조절
        if i == 20:
             threshold = 0.5  # 중반: 2mm로 축소
             print("  [Step 2] Threshold reduced to 2.0mm")
        elif i == 40:
             threshold = 0.1  # 후반: 0.5mm로 정밀 정합
             print("  [Step 3] Threshold reduced to 0.5mm")
        
        reg_p2plane = o3d.pipelines.registration.registration_icp(
            source, target_pcd, threshold, np.identity(4),
            estimation_method,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1) # 1회만 수행
        )
        
        # 변환 적용
        source.transform(reg_p2plane.transformation)
        global_transform = reg_p2plane.transformation @ global_transform # 변환 누적
        
        source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30)) # 변환 후 법선 재계산 (안전장치)
        
        # 시각화 업데이트
        vis.update_geometry(source)
        vis.poll_events()
        vis.update_renderer()
        
        # 속도 조절
        time.sleep(0.05)

    print("Alignment Finished.")
    vis.run() # 사용자가 종료할 때까지 대기
    vis.destroy_window()
    
    # 5. 최종 결과 확인 (전체 얼굴 모델 적용)
    print("\nVisualizing Full Face Alignment...")
    full_face_mesh = result['aligned_mesh']
    full_face_mesh.transform(global_transform)
    
    # 얼굴은 살색(?) 비슷하게, 스마일 아치는 흰색으로
    full_face_mesh.paint_uniform_color([1.0, 0.8, 0.6]) # 살구색
    target_mesh.paint_uniform_color([1.0, 1.0, 1.0]) # 흰색 (치아 가이드)
    
    o3d.visualization.draw_geometries(
        [full_face_mesh, target_mesh],
        window_name="Final Alignment (Full Face)",
        width=1200,
        height=800,
        front=[0, 1, 0],
        lookat=[0, 0, 0],
        up=[0, 0, 1],
        zoom=0.6
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Align Face Scan to Smile Arch")
    parser.add_argument('--sample', '-s', type=int, default=1, help="Sample index")
    args = parser.parse_args()
    
    align_face_scan_to_smile_arch(args.sample)
