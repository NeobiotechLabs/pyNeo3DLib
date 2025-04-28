def align_with_obb(mesh, debug=False):
    """
    OBB 축을 기준으로 메쉬를 정렬합니다. 
    가장 짧은 축은 z축, 가장 긴 축은 x축, 중간 길이 축은 y축으로 정렬됩니다.
    
    구체적인 변환 과정:
    1. 메쉬 중심점(OBB 중심)을 원점으로 이동
    2. PCA를 통해 주축을 찾아 회전 행렬 계산
    3. 회전 행렬을 적용하여 메쉬 정렬
    
    Args:
        mesh: PyVista 메쉬
        debug: 디버깅 모드 (중간 과정 시각화)
        
    Returns:
        aligned_mesh: 정렬된 메쉬
        obb_center: OBB 중심점
        rotation_matrix: 회전 행렬
    """
    try:
        print("[로그] align_with_obb 함수 시작")
        # 메쉬 복사
        aligned_mesh = mesh.copy()
        
        # 메쉬 정점
        vertices = mesh.points
        print(f"[로그] 정점 개수: {len(vertices)}")
        
        # OBB 계산에 필요한 값 추출
        # 점들의 평균 계산 (OBB 중심)
        mean_pt = np.mean(vertices, axis=0)
        print(f"[로그] OBB 중심 좌표: {mean_pt}")
        
        # 평균을 중심으로 점들을 이동
        centered_pts = vertices - mean_pt
        print(f"[로그] 센터링 완료")
        
        # 공분산 행렬 계산
        cov = np.cov(centered_pts, rowvar=False)
        print(f"[로그] 공분산 행렬 계산 완료")
        
        # 고유값과 고유벡터 계산
        try:
            print(f"[로그] 고유값 계산 시작")
            eigvals, eigvecs = np.linalg.eigh(cov)
            print(f"[로그] 고유값: {eigvals}")
            print(f"[로그] 고유벡터 행렬 형태: {eigvecs.shape}")
        except Exception as e:
            print(f"[오류] 고유값 계산 중 오류 발생: {e}")
            raise
        
        # 고유값이 큰 순서대로 정렬 (주축 순서대로)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        print(f"[로그] 정렬된 고유값: {eigvals}")
        
        # 고유값에 따라 축 정렬: 가장 큰 고유값 -> x축, 중간 -> y축, 가장 작은 -> z축
        rotation_matrix = eigvecs
        print(f"[로그] 회전 행렬 계산 완료")
        
        # 회전 변환 적용 (기존 좌표계에서 표준 좌표계로 변환)
        # 주의: centered_pts는 이미 중심이 원점으로 이동된 점들임
        transformed_vertices = np.dot(centered_pts, rotation_matrix)
        print(f"[로그] 정점 변환 완료")
        
        # 변환된 정점 적용
        aligned_mesh.points = transformed_vertices
        print(f"[로그] 변환된 메쉬 생성 완료")
        
        # 디버깅 모드: 변환 과정 시각화
        if debug:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(15, 5))
            
            # 원본 메쉬 점 시각화
            ax1 = fig.add_subplot(131, projection='3d')
            ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='b', marker='.', alpha=0.01)
            ax1.set_title('Original Mesh')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            
            # 중심이 원점으로 이동된 점 시각화
            ax2 = fig.add_subplot(132, projection='3d')
            ax2.scatter(centered_pts[:, 0], centered_pts[:, 1], centered_pts[:, 2], c='g', marker='.', alpha=0.01)
            ax2.set_title('Centered Mesh')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            
            # 회전 변환 후 점 시각화
            ax3 = fig.add_subplot(133, projection='3d')
            ax3.scatter(transformed_vertices[:, 0], transformed_vertices[:, 1], transformed_vertices[:, 2], c='r', marker='.', alpha=0.01)
            ax3.set_title('Rotated Mesh')
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.set_zlabel('Z')
            
            plt.tight_layout()
            plt.show()
        
        print(f"[로그] align_with_obb 함수 완료")
        return aligned_mesh, mean_pt, rotation_matrix
    except Exception as e:
        print(f"[오류] align_with_obb 함수에서 예외 발생: {e}")
        import traceback
        traceback.print_exc()
        # 에러가 발생해도 기본값 반환
        return mesh.copy(), np.zeros(3), np.eye(3)