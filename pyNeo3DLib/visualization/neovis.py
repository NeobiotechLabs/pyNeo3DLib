def visualize_meshes(meshes, names=None, title="메쉬 시각화", selected_vertices=None, mesh_index=0):
    """
    여러 메쉬를 한 번에 시각화합니다.
    
    Args:
        meshes: Mesh 객체 리스트
        names: 메쉬 이름 리스트 (기본값: None, None인 경우 "메쉬 1", "메쉬 2" 등으로 표시)
        title: 그래프 제목
        selected_vertices: 선택된 정점 인덱스 리스트 (기본값: None)
        mesh_index: 선택된 정점이 속한 메쉬의 인덱스 (기본값: 0)
    """
    import pyvista as pv
    import random
    
    # 이름이 지정되지 않은 경우 기본 이름 생성
    if names is None:
        names = [f"메쉬 {i+1}" for i in range(len(meshes))]
    
    # 플롯터 생성
    plotter = pv.Plotter()
    
    # 랜덤 색상 생성 함수
    def random_color():
        return [random.random(), random.random(), random.random()]
    
    # 각 메쉬 시각화
    for i, (mesh, name) in enumerate(zip(meshes, names)):
        # 메쉬를 PyVista 메쉬로 변환
        pv_mesh = _mesh_to_pyvista(mesh)
        
        # 랜덤 색상 생성
        color = random_color()
        
        # 메쉬 시각화 (반투명)
        plotter.add_mesh(pv_mesh, color=color, opacity=0.7, show_edges=False, 
                        edge_color='black', label=name)
        
        # 선택된 정점 강조 표시
        if selected_vertices is not None and i == mesh_index:
            # 선택된 정점 추출
            selected_points = mesh.vertices[selected_vertices]
            
            # 선택된 정점 시각화 (빨간색 점으로 표시)
            plotter.add_points(selected_points, color='red', point_size=10, 
                              render_points_as_spheres=True, label="선택된 정점")
    
    # 축 인디케이터 추가
    plotter.add_axes()
    
    # 범례 추가
    plotter.add_legend()
    
    # 제목 설정
    plotter.add_title(title, font_size=16)
    
    # 시각화
    plotter.show()

def _mesh_to_pyvista(mesh):
    """
    Mesh 객체를 PyVista 메쉬로 변환합니다.
    
    Args:
        mesh: Mesh 객체
        
    Returns:
        PyVista 메쉬 객체
    """
    import pyvista as pv
    
    # 정점과 면 추출
    vertices = mesh.vertices
    faces = mesh.faces
    
    # PyVista 메쉬 생성
    pv_mesh = pv.PolyData()
    
    # 정점 설정
    pv_mesh.points = vertices
    
    # 면 설정 (PyVista는 면의 정점 개수를 먼저 지정해야 함)
    face_list = []
    for face in faces:
        face_list.append(len(face))  # 면의 정점 개수
        face_list.extend(face)  # 면의 정점 인덱스
    
    pv_mesh.faces = face_list
    
    return pv_mesh