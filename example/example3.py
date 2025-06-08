from pyNeo3DLib import fastserver

import requests
import time
import json
import threading
import asyncio
import base64
import io
from PIL import Image

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
            if check_file(face_path) and face_item['path'].endswith(".obj"):
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
    
    # condyle_vertices = np.array(result['condyle']['mesh']['vertices'])
    # condyle_faces = np.array(result['condyle']['mesh']['faces'])
    # condyle_mesh = o3d.geometry.TriangleMesh()
    # condyle_mesh.vertices = o3d.utility.Vector3dVector(condyle_vertices)
    # condyle_mesh.triangles = o3d.utility.Vector3iVector(condyle_faces)
    # condyle_mesh.compute_vertex_normals()
    # condyle_mesh.paint_uniform_color([1.0, 1.0, 0.0])  # 노란색으로 설정
    # models.append(condyle_mesh)
    
    # 데이터 타입을 원본과 동일하게 명시적으로 지정하여 안정성 확보
    photo_vertices = np.array(result['photo']['vertices'], dtype=np.float64)
    photo_triangles = np.array(result['photo']['triangles'], dtype=np.int32)
    
    photo_mesh = o3d.geometry.TriangleMesh()
    photo_mesh.vertices = o3d.utility.Vector3dVector(photo_vertices)
    photo_mesh.triangles = o3d.utility.Vector3iVector(photo_triangles)
    photo_mesh.compute_vertex_normals()
    
    # 텍스처 좌표(UV) 할당
    if 'triangle_uvs' in result['photo']:
        triangle_uvs = np.array(result['photo']['triangle_uvs'], dtype=np.float64)
        if len(triangle_uvs) > 0:
            photo_mesh.triangle_uvs = o3d.utility.Vector2dVector(triangle_uvs)
    
    # Base64 텍스처 데이터 처리
    base64_data = result['photo']['textures'][0]['data']
    image_data = base64.b64decode(base64_data)
    image_stream = io.BytesIO(image_data)
    pil_image = Image.open(image_stream)
    
    if pil_image.mode == 'RGBA':
        pil_image = pil_image.convert('RGB')
        
    o3d_image = o3d.geometry.Image(np.array(pil_image))

    # 메시에 텍스처와 재질 ID 할당
    photo_mesh.textures = [o3d_image]
    
    # === 문제의 핵심 코드 ===
    # 각 삼각형이 어떤 텍스처를 사용할지 명시적으로 지정합니다.
    # 이 속성이 누락되어 렌더링이 실패한 것으로 보입니다.
    num_triangles = len(photo_mesh.triangles)
    photo_mesh.triangle_material_ids = o3d.utility.IntVector([0] * num_triangles)
    
    models.append(photo_mesh)
    
    # 모델이 로드되지 않았다면 중단
    if not models:
        print("오류: 표시할 모델이 없습니다. 파일 경로를 확인하세요.")
        return
    
    # 모든 모델의 경계 상자 계산
    print("\n=== 모델 경계 상자 정보 ===")
    all_points = []
    for i, model in enumerate(models):
        bbox = model.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        extent = bbox.get_extent()
        print(f"모델 {i+1}: 중심={center}, 크기={extent}")
        
        # 모든 점들을 수집
        vertices = np.asarray(model.vertices)
        if len(vertices) > 0:
            all_points.extend(vertices.tolist())
    
    # 전체 경계 상자 계산
    if all_points:
        all_points = np.array(all_points)
        overall_min = np.min(all_points, axis=0)
        overall_max = np.max(all_points, axis=0)
        overall_center = (overall_min + overall_max) / 2
        overall_size = np.max(overall_max - overall_min)
        print(f"전체: 중심={overall_center}, 최대크기={overall_size:.1f}")
    else:
        overall_center = np.array([0, 0, 0])
        overall_size = 100.0
    
    # 시각화 시작
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1200, height=800)  # 창 크기도 키우기
    
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
    
    # 개선된 카메라 위치 설정
    view_control = vis.get_view_control()
    
    # 전체 모델이 잘 보이도록 카메라 거리 계산
    camera_distance = 1000 #overall_size * 1.5  # 모델 크기의 1.5배 거리
    
    # 카메라를 모델 중심을 향하게 설정
    view_control.set_lookat(overall_center)  # 전체 모델의 중심을 바라보기
    
    # 카메라를 앞쪽 대각선에서 바라보도록 설정
    camera_position = overall_center + np.array([camera_distance*0.7, camera_distance*0.5, camera_distance*0.7])
    view_control.set_front((overall_center - camera_position) / np.linalg.norm(overall_center - camera_position))
    view_control.set_up([0, 1, 0])  # Y축을 위쪽으로
    
    # 줌 설정 (더 가깝게)
    view_control.set_zoom(1.2)  # 0.8에서 1.2로 증가
    
    print(f"카메라 설정: 중심={overall_center}, 거리={camera_distance:.1f}, 줌=1.2")
    print("시각화 창이 열렸습니다. 마우스로 회전/줌 가능합니다.")
    
    # 시각화 실행
    vis.run()
    vis.destroy_window()

async def main():
    with open(f"{__file__}/../sampleInput_photo.json", "r") as f:
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




