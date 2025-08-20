import sys
import os
# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from analyzing_IOS.core.arch_curve_finder import analyze_upper_IOS_scandata


if __name__ == "__main__":
    mesh_path = "./analyzing_IOS/data/Upper 안지숙님 편집.stl"
    
    arch_depth, molar_width, landmarks = analyze_upper_IOS_scandata(
        mesh_path=mesh_path,
        visualize_result=True
    )
    
    print(f"arch_depth: {arch_depth}")
    print(f"molar_width: {molar_width}")
    print(f"landmarks: {landmarks}")
