from pyNeo3DLib import fastserver

import requests
import time
import json
import threading
import asyncio

print("Server started, http://localhost:8000")


fastserver.start_server()


from pyNeo3DLib import Neo3DRegistration




print("\nPress Enter to stop server...")
input()

def show_result(result):
    print(result)
    
    # 시각화를 위해 필요한 라이브러리 가져오기
    import open3d as o3d
    import numpy as np
    import os

    # 시각화할 모델들의 리스트 생성
    models = []
    
    # 파일 존재 여부 확인 함수
    def check_file(file_path):
        if os.path.exists(file_path):
            return True
        else:
            print(f"경고: 파일을 찾을 수 없습니다: {file_path}")
            return False
    
    # 1. laminate 모델 (기준) 로드
    laminate_path = result['laminate']['path']
    if check_file(laminate_path):
        laminate_mesh = o3d.io.read_triangle_mesh(laminate_path)
        laminate_mesh.compute_vertex_normals()  # 빛 효과를 위한 법선 벡터 계산
        # 빨간색으로 설정 (더 밝은 빨간색)
        laminate_mesh.paint_uniform_color([1.0, 0.3, 0.3])
        models.append(laminate_mesh)
    
    # 2. ios[smileArch] 모델 로드 및 변환 매트릭스 적용
    for ios_item in result['ios']:
        if ios_item['subType'] == 'smileArch':
            ios_path = ios_item['path']
            if check_file(ios_path):
                ios_mesh = o3d.io.read_triangle_mesh(ios_path)
                ios_mesh.compute_vertex_normals()  # 빛 효과를 위한 법선 벡터 계산
                # 초록색으로 설정 (더 밝은 초록색)
                ios_mesh.paint_uniform_color([0.3, 1.0, 0.3])
                
                # 변환 매트릭스 적용
                transform_matrix = np.array(ios_item['transform_matrix'])
                ios_mesh.transform(transform_matrix)
                
                models.append(ios_mesh)
            break
    
    # 3. facescan[smile] 모델 로드 및 변환 매트릭스 적용
    for face_item in result['facescan']:
        if face_item['subType'] == 'faceSmile':
            face_path = face_item['path']
            if check_file(face_path):
                face_mesh = o3d.io.read_triangle_mesh(face_path)
                face_mesh.compute_vertex_normals()  # 빛 효과를 위한 법선 벡터 계산
                # 파란색으로 설정 (매우 옅은 파란색으로 변경)
                face_mesh.paint_uniform_color([0.6, 0.6, 1.0])
                
                # 변환 매트릭스 적용
                transform_matrix = np.array(face_item['transform_matrix'])
                face_mesh.transform(transform_matrix)
                
                models.append(face_mesh)
            break

    bow_path = result['smilearch_bow']['path']
    if check_file(bow_path):
        bow_mesh = o3d.io.read_triangle_mesh(bow_path)
        bow_mesh.compute_vertex_normals()  # 빛 효과를 위한 법선 벡터 계산
        bow_mesh.paint_uniform_color([0.8, 0.5, 0.5])
        transform_matrix = np.array(result['smilearch_bow']['transform_matrix'])
        bow_mesh.transform(transform_matrix)
        models.append(bow_mesh)
        
    # 콘딜 메시 생성
    condyle_vertices = np.array(result['condyle']['mesh']['vertices'])
    condyle_faces = np.array(result['condyle']['mesh']['faces'])
    condyle_mesh = o3d.geometry.TriangleMesh()
    condyle_mesh.vertices = o3d.utility.Vector3dVector(condyle_vertices)
    condyle_mesh.triangles = o3d.utility.Vector3iVector(condyle_faces)
    condyle_mesh.compute_vertex_normals()
    condyle_mesh.paint_uniform_color([1.0, 1.0, 0.0])  # 노란색으로 설정
    models.append(condyle_mesh)
    
    # 모델이 로드되지 않았다면 중단
    if not models:
        print("오류: 표시할 모델이 없습니다. 파일 경로를 확인하세요.")
        return
    
    # 시각화 시작
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # 각 모델 추가
    for model in models:
        vis.add_geometry(model)
    
    # 반투명 효과를 위한 렌더링 옵션 설정
    opt = vis.get_render_option()
    opt.background_color = np.array([0.9, 0.9, 1.0])  # 매우 밝은 연한 파란색 배경
    opt.light_on = True  # 조명 켜기
    
    # 조명 강화 시도
    try:
        opt.light_ambient_color = np.array([0.4, 0.4, 0.4])  # 환경광 강화
        opt.light_diffuse_color = np.array([1.0, 1.0, 1.0])  # 확산광 설정
        opt.light_specular_color = np.array([1.0, 1.0, 1.0])  # 반사광 설정
        opt.light_position = np.array([0.0, 0.0, 2.0])  # 조명 위치 조정
    except:
        # 일부 OpenSD 버전에서는 지원되지 않을 수 있음
        pass
    
    # 반투명 효과를 위한 설정
    opt.mesh_show_wireframe = False  # 와이어프레임 끄기 (더 깔끔한 표시)
    opt.mesh_show_back_face = True  # 뒷면 표시
    opt.point_size = 3.0  # 점 크기 설정
    
    # 카메라 위치 설정
    view_control = vis.get_view_control()
    view_control.set_zoom(0.8)
    view_control.set_front([0, 0, -1])
    view_control.set_lookat([0, 0, 0])
    view_control.set_up([0, 1, 0])
    
    # 시각화 실행
    vis.run()
    vis.destroy_window()

async def main():
    with open(f"{__file__}/../sampleInput.json", "r") as f:
        json_string = f.read()
        reg = Neo3DRegistration(json_string, fastserver.ws)
        print(reg.version)
        print(reg.parsed_json)
        result = await reg.run_registration(visualize=False)

    show_result(result)


asyncio.run(main())
input()

fastserver.stop_server()
print("Server stopped")




