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
    
    # 시각화할 모델들의 리스트 생성
    models = []
    
    # 1. laminate 모델 (기준) 로드
    laminate_path = result['laminate']['path']
    laminate_mesh = o3d.io.read_triangle_mesh(laminate_path)
    laminate_mesh.compute_vertex_normals()  # 빛 효과를 위한 법선 벡터 계산
    laminate_mesh.paint_uniform_color([1, 0, 0])  # 빨간색
    models.append(laminate_mesh)
    
    # 2. ios[smileArch] 모델 로드 및 변환 매트릭스 적용
    for ios_item in result['ios']:
        if ios_item['subType'] == 'smileArch':
            ios_path = ios_item['path']
            ios_mesh = o3d.io.read_triangle_mesh(ios_path)
            ios_mesh.compute_vertex_normals()  # 빛 효과를 위한 법선 벡터 계산
            ios_mesh.paint_uniform_color([0, 1, 0])  # 초록색
            
            # 변환 매트릭스 적용
            transform_matrix = np.array(ios_item['transform_matrix'])
            ios_mesh.transform(transform_matrix)
            
            models.append(ios_mesh)
            break
    
    # 3. facescan[smile] 모델 로드 및 변환 매트릭스 적용
    for face_item in result['facescan']:
        if face_item['subType'] == 'smile':
            face_path = face_item['path']
            face_mesh = o3d.io.read_triangle_mesh(face_path)
            face_mesh.compute_vertex_normals()  # 빛 효과를 위한 법선 벡터 계산
            face_mesh.paint_uniform_color([0, 0, 1])  # 파란색
            
            # 변환 매트릭스 적용
            transform_matrix = np.array(face_item['transform_matrix'])
            face_mesh.transform(transform_matrix)
            
            models.append(face_mesh)
            break
    
    # 모든 모델 시각화 (반투명 효과 적용)
    visualization_option = {
        'mesh_show_back_face': True,
        'mesh_show_wireframe': True,
        'light_on': True,
        'background_color': [1.0, 1.0, 1.0],  # 하얀색 배경
        'point_size': 5.0
    }
    
    # 반투명 효과를 위한 별도 시각화
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # 각 모델 추가 및 투명도 설정
    for i, model in enumerate(models):
        vis.add_geometry(model)
        # 각 모델의 재질 속성 설정
        opt = vis.get_render_option()
        opt.background_color = np.array([1, 1, 1])  # 하얀색 배경
        opt.light_on = True
        opt.mesh_show_wireframe = True
        opt.point_size = 5.0
        opt.transparency = 0.5  # 반투명 설정 (0.0은 완전 불투명, 1.0은 완전 투명)
    
    # 카메라 위치 설정
    vis.get_view_control().set_zoom(0.8)
    vis.get_view_control().set_front([0, 0, -1])
    vis.get_view_control().set_lookat([0, 0, 0])
    vis.get_view_control().set_up([0, 1, 0])
    
    # 시각화 실행
    vis.run()
    vis.destroy_window()

async def main():
    with open(f"{__file__}/../sampleInput.json", "r") as f:
        json_string = f.read()
        reg = Neo3DRegistration(json_string, fastserver.ws)
        print(reg.version)
        print(reg.parsed_json)
        result = await reg.run_registration(visualize=True)

    show_result(result)


asyncio.run(main())
input()

fastserver.stop_server()
print("Server stopped")




