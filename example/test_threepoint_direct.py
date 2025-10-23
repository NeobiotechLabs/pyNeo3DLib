"""
3점 정합 직접 테스트 (API 서버 없이)

이 스크립트는 API 서버를 거치지 않고 직접 ThreePointRegistration 클래스를 테스트합니다.
서버 설정이나 네트워크 문제 없이 기능을 확인할 수 있습니다.
"""

import sys
import os
import asyncio
import numpy as np

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("PyVista가 설치되지 않았습니다. 시각화 기능이 제한됩니다.")


def get_test_config():
    """
    테스트용 공통 설정을 반환하는 함수
    모든 테스트에서 동일한 파일과 점 좌표를 사용하도록 통일
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data/threepoint")
    
    # 통일된 파일 경로 설정
    config = {
        "target_file": os.path.join(data_dir, "smilearch.stl"),
        "source_file": os.path.join(data_dir, "maxillary.stl"),
        
        # 통일된 테스트 점 좌표
        # "target_points": [
        #     {"x": 25.22, "y": 19.31, "z": -8.87},
        #     {"x": 2.15, "y": 25.95, "z": 23.12},
        #     {"x": -18.17, "y": 20.14, "z": -10.78}
        # ],
        
        # "source_points": [
        #     {"x": 22.82, "y": 13.63, "z": -15.43},
        #     {"x": 2.73, "y": 11.35, "z": 16.73},
        #     {"x": -20.67, "y": 17.08, "z": -14.43}
        # ]
        "target_points": [
            {"x": 24.98, "y": 19.40, "z": -8.24},
            {"x": 2.13, "y": 25.61, "z": 22.71},
            {"x": -17.69, "y": 20.04, "z": -10.98}
        ],

        "source_points": [
            {"x": 22.73, "y": 13.12, "z": -13.16},
            {"x": 2.58, "y": 10.61, "z": 19.80},
            {"x": -20.25, "y": 15.79, "z": -11.00}
        ]
    }
    
    return config


async def test_mesh_loading():
    """메시 로딩만 테스트"""
    
    try:
        from pyNeo3DLib.fileLoader.mesh import Mesh
        print("✅ Mesh 클래스 import 성공")
    except Exception as e:
        print(f"❌ Mesh import 오류: {e}")
        return
    
    # 공통 설정 가져오기
    config = get_test_config()
    target_file = config["target_file"]
    
    if not os.path.exists(target_file):
        print(f"❌ 테스트 파일을 찾을 수 없습니다: {target_file}")
        return
    
    try:
        print(f"\n메시 로딩 테스트: {target_file}")
        mesh = Mesh.from_file(target_file)
        print(f"✅ 메시 로딩 성공")
        print(f"   정점 수: {len(mesh.vertices)}")
        print(f"   면 수: {len(mesh.faces)}")
        print(f"   정점 범위: {np.min(mesh.vertices, axis=0)} ~ {np.max(mesh.vertices, axis=0)}")
        
        return mesh
        
    except Exception as e:
        print(f"❌ 메시 로딩 오류: {e}")
        import traceback
        traceback.print_exc()
        return None


async def visualize_meshes_with_points(target_file, source_file, target_points, source_points):
    """
    메시와 점들을 시각화하는 독립적인 함수
    
    Args:
        target_file: 타겟 메시 파일 경로
        source_file: 소스 메시 파일 경로
        target_points: 타겟 점들
        source_points: 소스 점들
    """
    if not PYVISTA_AVAILABLE:
        print("PyVista가 설치되지 않아 시각화를 건너뜁니다.")
        return
    
    try:
        from pyNeo3DLib.fileLoader.mesh import Mesh
        
        # 메시 로드
        target_mesh = Mesh.from_file(target_file)
        source_mesh = Mesh.from_file(source_file)
        
        # PyVista 플로터 생성
        plotter = pv.Plotter(window_size=(1200, 800))
        plotter.set_background('white')
        
        # 메시를 PyVista 형식으로 변환
        def mesh_to_pyvista(mesh):
            vertices = mesh.vertices
            faces = mesh.faces
            pv_faces = []
            for face in faces:
                pv_faces.append(len(face))
                pv_faces.extend(face)
            return pv.PolyData(vertices, pv_faces)
        
        # 타겟 메시 (파란색)
        target_pv = mesh_to_pyvista(target_mesh)
        plotter.add_mesh(target_pv, color='lightblue', opacity=0.6, label='Target Mesh')
        
        # 소스 메시 (빨간색)
        source_pv = mesh_to_pyvista(source_mesh)
        plotter.add_mesh(source_pv, color='lightcoral', opacity=0.6, label='Source Mesh')
        
        # 점 좌표를 numpy 배열로 변환
        target_coords = np.array([[p["x"], p["y"], p["z"]] for p in target_points])
        source_coords = np.array([[p["x"], p["y"], p["z"]] for p in source_points])
        
        # 타겟 점들 (파란색 구)
        plotter.add_points(target_coords, color='blue', point_size=20,
                         render_points_as_spheres=True, label='Target Points')
        
        # 소스 점들 (빨간색 구)
        plotter.add_points(source_coords, color='red', point_size=20,
                         render_points_as_spheres=True, label='Source Points')
        
        # 점들에 번호 표시
        for i, coord in enumerate(target_coords):
            plotter.add_point_labels([coord], [f'T{i+1}'], point_size=15, font_size=12)
        
        for i, coord in enumerate(source_coords):
            plotter.add_point_labels([coord], [f'S{i+1}'], point_size=15, font_size=12)
        
        # 범례 및 제목 추가
        plotter.add_legend(size=(0.3, 0.3), loc='upper right')
        plotter.add_title('Three-Point Registration Input Data', font_size=16)
        plotter.add_axes()
        
        # 카메라 설정
        plotter.camera_position = 'iso'
        
        print("시각화 창이 열렸습니다. 창을 닫으면 계속 진행됩니다.")
        plotter.show()
        
    except Exception as e:
        print(f"시각화 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


async def test_with_visualization():
    """시각화를 포함한 테스트"""
    
    # 공통 설정 가져오기
    config = get_test_config()
    target_file = config["target_file"]
    source_file = config["source_file"]
    target_points = config["target_points"]
    source_points = config["source_points"]
    
    # 파일 존재 확인
    if not os.path.exists(target_file) or not os.path.exists(source_file):
        print("테스트 파일을 찾을 수 없습니다.")
        return
    
    print("=== 시각화 포함 테스트 ===")
    
    # 1. 입력 데이터 시각화
    print("\n1. 입력 데이터 시각화")
    await visualize_meshes_with_points(target_file, source_file, target_points, source_points)
    
    # 2. 3점 정합 실행 (내장 시각화 포함)
    print("\n2. 3점 정합 실행 (결과 시각화 포함)")
    try:
        from pyNeo3DLib.threePointRegistration.threePointRegistration import ThreePointRegistration
        
        three_point_reg = ThreePointRegistration(
            target_mesh_path=target_file,
            source_mesh_path=source_file,
            target_points=target_points,
            source_points=source_points,
            visualization=True  # 시각화 활성화
        )
        
        result = await three_point_reg.run_registration()
        
        if result is not None:
            print("🎉 시각화 포함 테스트가 성공적으로 완료되었습니다!")
            return result
        else:
            print("❌ 3점 정합에 실패했습니다.")
            return None
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None


async def main():
    """메인 테스트 함수"""
    
    print("=== 3점 정합 직접 테스트 시작 ===")
    print("API 서버 없이 직접 클래스를 테스트합니다.\n")    
    
    print("\n=== 시각화 포함 테스트 모드 ===")
    result = await test_with_visualization()
    if result is not None:
        print("\n🎉 시각화 포함 테스트가 성공적으로 완료되었습니다!")
    else:
        print("\n❌ 시각화 포함 테스트에 실패했습니다.")



if __name__ == "__main__":
    asyncio.run(main())
