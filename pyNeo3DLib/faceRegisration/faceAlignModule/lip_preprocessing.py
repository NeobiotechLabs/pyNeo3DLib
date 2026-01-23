"""
Face Lip Preprocessing - 입술 영역 전처리 알고리즘 개발용

전체 3D 모델과 추출된 입술 부분을 시각화하여 비교
"""

import numpy as np
import argparse
from pathlib import Path
import open3d as o3d

from .face_lip_extractor import FaceLipExtractor



def load_mesh_from_folder(folder_path: str):
    """
    폴더에서 3D 메시 로드 (PLY 또는 OBJ)
    
    Returns:
        open3d.geometry.TriangleMesh
    """
    folder = Path(folder_path)
    
    ply_files = list(folder.glob('*.ply'))
    obj_files = list(folder.glob('*.obj'))
    
    if ply_files:
        mesh = o3d.io.read_triangle_mesh(str(ply_files[0]))
    elif obj_files:
        mesh = o3d.io.read_triangle_mesh(str(obj_files[0]))
    else:
        raise FileNotFoundError(f"No PLY or OBJ files found in {folder_path}")
    
    mesh.compute_vertex_normals()
    return mesh


def align_face_axis(
    mesh: o3d.geometry.TriangleMesh,
    extractor: 'FaceLipExtractor',
    folder_path: str
) -> tuple:
    """
    MediaPipe 랜드마크를 사용하여 얼굴 축정렬
    
    정렬 결과:
    - X축: 좌우 방향 (왼쪽 눈 → 오른쪽 눈)
    - Z축: 상하 방향 (눈 중심 → 코 끝 방향의 수직)
    - Y축: 전후 방향 (얼굴이 +Y 방향을 바라봄)
    - 원점: 코 끝
    
    Args:
        mesh: 정렬할 메시
        extractor: FaceLipExtractor 인스턴스
        folder_path: 스캔 데이터 폴더 경로
        
    Returns:
        tuple: (정렬된 메시, 변환 행렬)
    """
    # 텍스처 이미지에서 얼굴 랜드마크 추출
    texture_image = extractor.get_texture_image(folder_path)
    if texture_image is None:
        print("  Warning: No texture image for alignment")
        return mesh, np.eye(4)
    
    landmarks = extractor.detect_face_landmarks(texture_image)
    if landmarks is None:
        print("  Warning: No face detected for alignment")
        return mesh, np.eye(4)
    
    # MediaPipe 랜드마크 인덱스
    # 코 끝: 1
    # 왼쪽 눈 바깥쪽: 33, 안쪽: 133
    # 오른쪽 눈 바깥쪽: 263, 안쪽: 362
    # 입 중심: 13 (상순 중심), 14 (하순 중심)
    NOSE_TIP = 1
    LEFT_EYE_OUTER = 33
    LEFT_EYE_INNER = 133
    RIGHT_EYE_OUTER = 263
    RIGHT_EYE_INNER = 362
    UPPER_LIP_CENTER = 13
    LOWER_LIP_CENTER = 14
    
    # PLY/OBJ 파일에서 정점과 UV 좌표 추출 (Open3D 아닌 우리 파서 사용)
    folder = Path(folder_path)
    ply_files = list(folder.glob('*.ply'))
    obj_files = list(folder.glob('*.obj'))
    
    if ply_files:
        mesh_data = extractor.parse_ply(str(ply_files[0]))
        mesh_uvs = mesh_data['uvs']
        parser_vertices = mesh_data['vertices']
    else:
        mesh_data = extractor.parse_obj(str(obj_files[0]))
        mesh_uvs = mesh_data['vertex_uvs']
        parser_vertices = mesh_data['vertices']
    
    # 우리 파서의 정점 사용 (Open3D와 인덱스 일관성 유지)
    vertices = parser_vertices
    
    # 랜드마크 UV를 3D 좌표로 변환
    def uv_to_3d(landmark_idx):
        x, y, _ = landmarks[landmark_idx]
        uv = np.array([x, 1.0 - y])  # UV 좌표 변환
        distances = np.linalg.norm(mesh_uvs - uv, axis=1)
        nearest_idx = np.argmin(distances)
        return vertices[nearest_idx]
    
    # 기준점 추출
    nose_tip = uv_to_3d(NOSE_TIP)
    left_eye = (uv_to_3d(LEFT_EYE_OUTER) + uv_to_3d(LEFT_EYE_INNER)) / 2
    right_eye = (uv_to_3d(RIGHT_EYE_OUTER) + uv_to_3d(RIGHT_EYE_INNER)) / 2
    eye_center = (left_eye + right_eye) / 2
    mouth_center = (uv_to_3d(UPPER_LIP_CENTER) + uv_to_3d(LOWER_LIP_CENTER)) / 2
    
    print(f"  Reference points:")
    print(f"    Nose tip: {nose_tip}")
    print(f"    Left eye center: {left_eye}")
    print(f"    Right eye center: {right_eye}")
    print(f"    Eye center: {eye_center}")
    print(f"    Mouth center: {mouth_center}")
    
    # === 1단계: X축 정렬 (눈으로) ===
    # X축: 왼쪽 눈 → 오른쪽 눈 (좌우)
    x_axis = right_eye - left_eye
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # === 2단계: Z축 정렬 (눈과 입으로) ===
    # Z축: 입 중심 → 눈 중심 (상하, 위쪽이 +Z)
    z_axis_raw = eye_center - mouth_center
    
    # Z축을 X축에 수직하게 만들기 (Gram-Schmidt)
    z_axis = z_axis_raw - np.dot(z_axis_raw, x_axis) * x_axis
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    # === 3단계: Y축 계산 ===
    # Y축: Z × X (오른손 법칙)
    # X축이 눈 방향이므로, 이렇게 하면 양 눈이 동일한 Y값을 갖게 됨
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # Y축이 코 방향과 반대면 Y만 뒤집기 (Z는 유지 - 눈이 코보다 위)
    nose_direction = nose_tip - eye_center
    if np.dot(y_axis, nose_direction) < 0:
        y_axis = -y_axis
        # X축을 다시 계산하여 오른손 법칙 유지 (Y × Z = X)
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
    
    print(f"  Computed axes:")
    print(f"    X (right): {x_axis}")
    print(f"    Y (forward): {y_axis}")
    print(f"    Z (up): {z_axis}")
    
    # 회전 행렬 생성 (현재 축 → 표준 축)
    rotation_matrix = np.array([x_axis, y_axis, z_axis]).T
    
    # 코 끝을 원점으로 이동
    center = nose_tip
    
    # 4x4 변환 행렬 생성
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix.T  # 역회전
    transform[:3, 3] = -rotation_matrix.T @ center  # 이동
    
    # 메시 변환
    aligned_mesh = mesh.transform(transform)
    aligned_mesh.compute_vertex_normals()
    
    print(f"  Face aligned: looking +Y, up +Z, right +X")
    print(f"  Origin: nose tip")
    
    return aligned_mesh, transform


def filter_by_normal_direction(
    vertices: np.ndarray,
    normals: np.ndarray,
    vertex_indices: np.ndarray,
    front_direction: np.ndarray = None,
    threshold: float = 0.0
) -> np.ndarray:
    """
    법선 벡터가 특정 방향(기본: +Y)을 향하는 정점만 필터링
    
    정렬된 얼굴에서 +Y 방향이 정면이므로,
    법선이 +Y 방향을 향하면 카메라에서 보이는 표면.
    
    Args:
        vertices: 전체 메시 정점
        normals: 전체 메시 법선
        vertex_indices: 필터링할 정점 인덱스
        front_direction: 정면 방향 (기본: +Y)
        threshold: 내적 임계값 (0 = 수직, 양수 = 더 정면만)
        
    Returns:
        필터링된 정점 인덱스
    """
    # 해당 정점들의 법선 추출
    selected_normals = normals[vertex_indices]
    
    # 기본 정면 방향: +Y (정렬된 얼굴이 바라보는 방향)
    if front_direction is None:
        front_direction = np.array([0.0, 1.0, 0.0])
    
    print(f"  Front direction: {front_direction}")
    
    # 각 정점의 법선과 정면 방향의 내적 계산
    # 양수 = 법선이 정면을 향함 (보이는 표면)
    # 음수 = 법선이 뒤쪽을 향함 (안 보이는 표면)
    dot_products = np.dot(selected_normals, front_direction)
    
    # 정면을 향하는 정점만 선택
    front_facing_mask = dot_products > threshold
    filtered_indices = vertex_indices[front_facing_mask]
    
    print(f"  Normal filter: {len(vertex_indices)} -> {len(filtered_indices)} vertices")
    print(f"    (kept {len(filtered_indices)/len(vertex_indices)*100:.1f}% facing +Y)")
    
    return filtered_indices


def visualize_lip_extraction(folder_path: str, use_outer: bool = False, visualize: bool = True):
    """
    전체 3D 모델과 입술 영역을 함께 시각화
    
    Args:
        folder_path: 스캔 데이터 폴더 경로
        use_outer: 외부 입술 사용 여부
        visualize: 시각화 창을 띄울지 여부
    """
    print(f"\n{'='*60}")
    print(f"Processing: {folder_path}")
    print(f"{'='*60}")
    
    # 1. 전체 메시 로드 (시각화용)
    print("\n1. Loading full mesh...")
    full_mesh = load_mesh_from_folder(folder_path)
    print(f"  Open3D mesh: {len(full_mesh.vertices)} vertices, {len(full_mesh.triangles)} triangles")
    
    # 2. 얼굴 축정렬 및 입술 추출
    print("\n2. Aligning face axis...")
    with FaceLipExtractor() as extractor:
        aligned_mesh, transform = align_face_axis(full_mesh, extractor, folder_path)
        
        # 3. 입술 영역 추출
        print("\n3. Extracting lip region...")
        result = extractor.extract_lip_region(folder_path, use_outer=use_outer)
        
        # 원본 파서 정점 가져오기 (인덱스 일관성)
        folder = Path(folder_path)
        ply_files = list(folder.glob('*.ply'))
        if ply_files:
            mesh_data = extractor.parse_ply(str(ply_files[0]))
            parser_vertices = mesh_data['vertices']
            parser_normals = mesh_data.get('normals', np.zeros_like(parser_vertices))
        else:
            obj_files = list(folder.glob('*.obj'))
            mesh_data = extractor.parse_obj(str(obj_files[0]))
            parser_vertices = mesh_data['vertices']
            parser_normals = np.zeros_like(parser_vertices)  # OBJ 법선은 나중에 계산
    
    if result is None:
        print("Failed to extract lip region!")
        return
    
    all_lip_indices = result['all_vertex_indices']
    boundary_indices = result['vertex_indices']
    
    # 원본 정점에 변환 적용
    parser_vertices_homo = np.hstack([parser_vertices, np.ones((len(parser_vertices), 1))])
    transformed_vertices = (transform @ parser_vertices_homo.T).T[:, :3]
    
    # 법선 변환 (회전만 적용)
    rotation = transform[:3, :3]
    if np.allclose(parser_normals, 0):
        print("  Notice: Missing normals. Computing vertex normals using Open3D...")
        # Open3D 메시를 사용하여 법선 계산
        temp_mesh = o3d.geometry.TriangleMesh()
        temp_mesh.vertices = o3d.utility.Vector3dVector(parser_vertices)
        if 'faces' in mesh_data:
            faces = mesh_data['faces']
            # OBJ 파서의 경우 faces가 (vertex_indices, uv_indices) 튜플 리스트임
            if len(faces) > 0 and isinstance(faces[0], tuple):
                face_v_indices = [f[0] for f in faces]
                temp_mesh.triangles = o3d.utility.Vector3iVector(face_v_indices)
            else:
                temp_mesh.triangles = o3d.utility.Vector3iVector(faces)
                
            temp_mesh.compute_vertex_normals()
            parser_normals = np.asarray(temp_mesh.vertex_normals)
        else:
            print("  Warning: No face data available to compute normals.")
            parser_normals = np.zeros_like(parser_vertices)
            
    transformed_normals = (rotation @ parser_normals.T).T
    
    # 추출된 정점들
    boundary_vertices_aligned = transformed_vertices[boundary_indices]
    all_lip_vertices_aligned = transformed_vertices[all_lip_indices]
    
    print(f"  Boundary vertices: {len(boundary_indices)}")
    print(f"  All lip vertices: {len(all_lip_indices)}")
    print(f"  Parser vertices: {len(parser_vertices)}, Transformed: {len(transformed_vertices)}")
    
    # 4. 법선 필터링 적용 (조건 강화: 정면과 45도 이내인 것만)
    print("\n4. Applying strict normal filter (+Y direction, within 45deg)...")
    
    # 임계값: cos(45도) ≈ 0.707 (정면과 45도 이내인 면만 통과)
    strict_angle = 45
    threshold = np.cos(np.radians(strict_angle))
    
    filtered_indices = filter_by_normal_direction(
        vertices=transformed_vertices,
        normals=transformed_normals,
        vertex_indices=all_lip_indices,
        front_direction=np.array([0.0, 1.0, 0.0]),
        threshold=threshold
    )
    
    filtered_vertices = transformed_vertices[filtered_indices]
    
    # 5. 시각화 준비 (통과한 점은 노랑, 탈락한 점은 파랑)
    geometries = []
    
    # 필터링에서 탈락한 정점들 (파란색, 작게 표시)
    filtered_set = set(filtered_indices)
    rejected_indices = np.array([idx for idx in all_lip_indices if idx not in filtered_set])
    
    if len(rejected_indices) > 0:
        rejected_pcd = o3d.geometry.PointCloud()
        rejected_pcd.points = o3d.utility.Vector3dVector(transformed_vertices[rejected_indices])
        rejected_pcd.paint_uniform_color([0.2, 0.5, 1.0])  # 파란색
        geometries.append(rejected_pcd)
    
    # 필터링 통과한 점들에 대해 DBSCAN 클러스터링 수행
    if len(filtered_vertices) > 0:
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_vertices)
        
        # DBSCAN 클러스터링 (가중치 적용)
        print("\n5. Clustering filtered points (Weighted DBSCAN)...")
        
        # 축별 허용 거리 설정 (단위: mm)
        threshold_x = 1.5  # 좌우: 2mm까지는 같은 그룹으로 인정 (관대함)
        threshold_y = 1.0  # 깊이: 1mm (보통 편차)
        threshold_z = 0.5  # 상하: 1mm (엄격함 -> 윗니/아랫니 분리)
        
        # 좌표 스케일링: 허용 거리가 클수록 값을 작게 만들어 거리가 가깝게 측정되도록 함
        scaled_vertices = np.copy(filtered_vertices)
        scaled_vertices[:, 0] /= threshold_x
        scaled_vertices[:, 1] /= threshold_y
        scaled_vertices[:, 2] /= threshold_z
        
        temp_pcd = o3d.geometry.PointCloud()
        temp_pcd.points = o3d.utility.Vector3dVector(scaled_vertices)
        
        # eps=1.0 (정규화된 공간에서의 거리 1.0)
        labels = np.array(temp_pcd.cluster_dbscan(eps=1.0, min_points=10, print_progress=False))

        # DBSCAN 결과 분석 및 최종 그룹 선정
        unique_labels = np.unique(labels)
        candidates = []
        
        print(f"  Found {len(unique_labels)} clusters (Weighted X={threshold_x}, Z={threshold_z})")
        
        for label in unique_labels:
            if label == -1: continue  # 노이즈 건너뜀
            
            mask = (labels == label)
            points = filtered_vertices[mask]
            count = len(points)
            mean_z = np.mean(points[:, 2])
            
            # 조건 1: 점 개수 1000개 이상
            if count > 1000:
                candidates.append({
                    'label': label,
                    'count': count,
                    'mean_z': mean_z,
                    'mask': mask
                })
                print(f"    Candidate Cluster {label}: {count} points, Mean Z={mean_z:.2f}")
            else:
                # 1000개 미만은 로그만 (또는 생략)
                pass

        # 최종 선택 로직
        final_mask = np.zeros(len(filtered_vertices), dtype=bool)
        
        if candidates:
            # 조건 2: 평균 Z가 가장 높은(위쪽인) 그룹 선택
            best_cluster = max(candidates, key=lambda x: x['mean_z'])
            final_mask = best_cluster['mask']
            print(f"  => SELECTED Cluster {best_cluster['label']} (Highest Z): {best_cluster['count']} points")
        else:
            print("  => Warning: No cluster met the size criteria (>1000 points).")

        # 인덱스 재분류
        # kept_indices: 이번에 최종 선택된 녀석들
        # rejected_indices: 기존 탈락자 + 필터 통과했으나 클러스터링에서 탈락한 녀석들
        
        # filtered_indices는 전체 메쉬 기준 인덱스임
        # final_mask는 filtered_vertices 내에서의 마스크
        
        new_kept_indices = filtered_indices[final_mask]
        new_rejected_from_filter = filtered_indices[~final_mask]
        
        # 전체 탈락자 합치기
        final_rejected_indices = np.concatenate([rejected_indices, new_rejected_from_filter])
        
        # 업데이트
        filtered_vertices = filtered_vertices[final_mask]
        filtered_indices = new_kept_indices
        rejected_indices = final_rejected_indices

    else:
        print("  Warning: No vertices passed the normal filter.")

    # 5. 시각화 준비 (visualize=True일 때만 실행)
    if visualize:
        geometries = []

        # 최종 탈락자 (파란색, 작게)
        if len(rejected_indices) > 0:
            rejected_pcd = o3d.geometry.PointCloud()
            rejected_pcd.points = o3d.utility.Vector3dVector(transformed_vertices[rejected_indices])
            rejected_pcd.paint_uniform_color([0.2, 0.5, 1.0])  # 파란색
            geometries.append(rejected_pcd)

        # 최종 선택된 윗니 (노란색, 크게)
        if len(filtered_vertices) > 0:
            final_pcd = o3d.geometry.PointCloud()
            final_pcd.points = o3d.utility.Vector3dVector(filtered_vertices)
            final_pcd.paint_uniform_color([1.0, 1.0, 0.0])  # 노란색!!!!!!!!!!
            geometries.append(final_pcd)
        
        # 외곽선 정점 (빨간색)
        boundary_pcd = o3d.geometry.PointCloud()
        boundary_pcd.points = o3d.utility.Vector3dVector(boundary_vertices_aligned)
        boundary_pcd.paint_uniform_color([1.0, 0.0, 0.0])
        geometries.append(boundary_pcd)
        
        # 외곽선 연결 (빨간색 라인)
        lines = [[i, (i + 1) % len(boundary_vertices_aligned)] for i in range(len(boundary_vertices_aligned))]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(boundary_vertices_aligned)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color([1.0, 0.0, 0.0])
        geometries.append(line_set)
        
        # 정렬된 전체 메시 (회색) - 배경으로 추가
        aligned_mesh.paint_uniform_color([0.7, 0.7, 0.7])
        geometries.append(aligned_mesh)
        
        print("\n6. Visualization:")
        print(f"  Total lip points: {len(all_lip_indices)}")
        print(f"  Yellow (Final Selected): {len(filtered_indices)} points (Highest Cluster > 1000)")
        print(f"  Blue (Rejected): {len(rejected_indices)} points")
        print("\nPress Q to close the visualization window...")
        
        # Visualizer 생성 및 설정
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=f"Final Upper Teeth - {Path(folder_path).name}",
            width=1200,
            height=800
        )
        
        # 1. 메시를 먼저 추가 (배경)
        vis.add_geometry(aligned_mesh)
        
        # 2. 파란색 점들 추가 (탈락자)
        for geom in geometries:
            if isinstance(geom, o3d.geometry.LineSet) or geom != aligned_mesh:
                vis.add_geometry(geom)
        
        # 시각화 설정
        opt = vis.get_render_option()
        opt.point_size = 5.0  # 기본 크기
        opt.light_on = True
        
        # 카메라 설정: +Y 방향 멀리서 -Y 방향(원점)을 바라보도록 (얼굴 정면)
        ctr = vis.get_view_control()
        ctr.set_front([0, 1, 0])    # 카메라가 +Y 위치에서 원점(코 끝)을 바라봄
        ctr.set_up([0, 0, 1])       # 위쪽이 +Z
        ctr.set_lookat([0, 0, 0])   # 원점을 정면으로 바라봄
        ctr.set_zoom(0.3)           # 약간 더 멀리서 보기
        
        vis.run()
        vis.destroy_window()
    
    # 결과 반환
    result['filtered_vertices'] = filtered_vertices
    result['filtered_indices'] = filtered_indices
    result['aligned_mesh'] = aligned_mesh
    return result


def main():
    parser = argparse.ArgumentParser(description='Face Lip Preprocessing')
    parser.add_argument('--sample', '-s', type=int, default=1, 
                        help='샘플 번호 (1, 2, 3)')
    parser.add_argument('--outer', action='store_true',
                        help='외부 입술 사용')
    parser.add_argument('--folder', '-f', type=str, default=None,
                        help='직접 폴더 경로 지정')
    args = parser.parse_args()
    
    # 폴더 경로 결정
    if args.folder:
        folder_path = args.folder
    else:
        folder_path = f'./3dmodel/facescan/sample_{args.sample}'
    
    # 시각화 실행
    result = visualize_lip_extraction(folder_path, use_outer=args.outer)
    
    if result:
        print("\n" + "="*60)
        print("Extraction Results:")
        print(f"  Boundary vertices: {result['boundary_vertices'].shape}")
        print(f"  All lip vertices: {result['all_lip_vertices'].shape}")
        print(f"  Filtered vertices: {result['filtered_vertices'].shape}")


if __name__ == '__main__':
    main()

