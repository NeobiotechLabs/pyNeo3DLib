import sys
import os
import time
# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

# smileArchOuterline의 core 모듈만 직접 import
sys.path.insert(0, os.path.join(project_root, 'pyNeo3DLib'))
from pyNeo3DLib.smileArchOuterline.core.arch_curve_finder import analyze_upper_IOS_scandata


if __name__ == "__main__":
    # smileArchOuterline 디렉토리의 데이터 파일 사용
    time_start = time.time()
    print("run_analysis.py")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    smile_arch_dir = os.path.dirname(script_dir)
    mesh_path = os.path.join(smile_arch_dir, "data","smile_arch_origin.stl")
    
    arch_depth, molar_width, landmarks = analyze_upper_IOS_scandata(
        mesh_path=mesh_path,
        visualize_result=False
    )
    
    print(f"arch_depth: {arch_depth}")
    print(f"molar_width: {molar_width}")
    print(f"landmarks: {landmarks}")
    time_end = time.time()
    print(f"time: {time_end - time_start}")