"""
IOS Registration Test Script
=============================

상악/하악 정합 전체 프로세스를 테스트합니다.

1. SmileArch laminate 정렬 (IOSLaminateRegistration)
2. 상악/하악 정합 (IOSUpperLowerRegistration)

Usage:
    python -m pyNeo3DLib.iosRegistration.test_ios_registration --config config.json --visualize

JSON 설정 파일 예시:
{
    "smileArch": "path/to/smileArch.stl",
    "upper": "path/to/upper.stl",
    "lower": "path/to/lower.stl"
}

Author: Antigravity Assistant
Date: 2026-01-09
"""

import json
import asyncio
import argparse
import numpy as np
from pathlib import Path

from pyNeo3DLib.fileLoader.mesh import Mesh


def load_config(config_path: str) -> dict:
    """JSON 설정 파일을 로드합니다."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # ios 배열 형식 지원 (기존 registration.py 포맷)
    if 'ios' in config:
        paths = {}
        for item in config['ios']:
            sub_type = item.get('subType', '')
            path = item.get('path', '')
            if sub_type and path:
                paths[sub_type] = path
        return paths
    
    # 단순 형식 지원
    required_keys = ['smileArch', 'upper', 'lower']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key '{key}' in config file")
    
    return config


def validate_paths(paths: dict) -> bool:
    """파일 경로 유효성을 검증합니다."""
    all_valid = True
    for key, path in paths.items():
        if not path:
            print(f"[WARNING] '{key}' path is empty")
            all_valid = False
        elif not Path(path).exists():
            print(f"[ERROR] '{key}' file not found: {path}")
            all_valid = False
        else:
            print(f"[OK] {key}: {path}")
    return all_valid


async def run_registration(paths: dict, visualize: bool = False) -> dict:
    """전체 정합 프로세스를 실행합니다."""
    import os
    
    # Lazy imports
    from pyNeo3DLib.iosRegistration.iosLaminateRegistration import IOSLaminateRegistration
    from pyNeo3DLib.iosRegistration.iosUpperLowerRegistration import IOSUpperLowerRegistration
    
    results = {}
    
    # Laminate 경로
    laminate_path = os.path.join(os.path.dirname(__file__), "..", "smile_arch_half.stl")
    if not os.path.exists(laminate_path):
        # 대체 경로 시도
        laminate_path = os.path.join(os.path.dirname(__file__), "../smile_arch_half.stl")
    
    print("\n" + "=" * 60)
    print("[Step 1] SmileArch Laminate Registration")
    print("=" * 60)
    
    try:
        smile_arch_path = paths['smileArch']
        laminate_reg = IOSLaminateRegistration(smile_arch_path, laminate_path, visualization=False)
        ios_laminate_result = laminate_reg.run_registration()
        results['smileArch_transform'] = ios_laminate_result.tolist()
        print(f"[SUCCESS] SmileArch 정렬 완료")
        print(f"  Transform:\n{ios_laminate_result}")
    except Exception as e:
        print(f"[ERROR] SmileArch 정렬 실패: {e}")
        import traceback
        traceback.print_exc()
        ios_laminate_result = np.eye(4)
        results['smileArch_transform'] = ios_laminate_result.tolist()
    
    print("\n" + "=" * 60)
    print("[Step 2] Upper/Lower Registration")
    print("=" * 60)
    
    try:
        # 메시 로드
        smile_arch_mesh = Mesh.from_file(paths['smileArch'])
        upper_mesh = Mesh.from_file(paths['upper'])
        lower_mesh = Mesh.from_file(paths['lower'])
        
        print(f"  SmileArch: {len(smile_arch_mesh.vertices)} vertices")
        print(f"  Upper: {len(upper_mesh.vertices)} vertices")
        print(f"  Lower: {len(lower_mesh.vertices)} vertices")
        
        # 정합 실행
        registration = IOSUpperLowerRegistration(
            upper_mesh, lower_mesh, smile_arch_mesh, ios_laminate_result
        )
        
        upper_transform, lower_transform = await registration.compute_transformations(
            visualize=visualize
        )
        
        results['upper_transform'] = upper_transform.tolist()
        results['lower_transform'] = lower_transform.tolist()
        
        print(f"\n[SUCCESS] Upper/Lower 정합 완료")
        
        # 시각화
        if visualize:
            visualize_final_result(
                smile_arch_mesh, upper_mesh, lower_mesh,
                ios_laminate_result, upper_transform, lower_transform
            )
        
    except Exception as e:
        print(f"[ERROR] Upper/Lower 정합 실패: {e}")
        import traceback
        traceback.print_exc()
        results['upper_transform'] = np.eye(4).tolist()
        results['lower_transform'] = np.eye(4).tolist()
    
    return results


def visualize_final_result(
    smile_arch_mesh: Mesh,
    upper_mesh: Mesh,
    lower_mesh: Mesh,
    smilearch_transform: np.ndarray,
    upper_transform: np.ndarray,
    lower_transform: np.ndarray
):
    """정합 결과를 3D로 시각화합니다."""
    try:
        import trimesh
    except ImportError:
        print("[WARNING] trimesh가 설치되지 않아 시각화를 건너뜁니다.")
        return
    
    print("\n" + "=" * 60)
    print("[Visualization] 3D 뷰어 열기...")
    print("=" * 60)
    print("  - 초록: SmileArch")
    print("  - 빨강: Upper (상악)")
    print("  - 파랑: Lower (하악)")
    print("  - 마우스 드래그: 회전, 휠: 줌")
    
    def apply_transform(vertices: np.ndarray, transform: np.ndarray) -> np.ndarray:
        ones = np.ones((len(vertices), 1))
        vertices_homo = np.hstack([vertices, ones])
        transformed = (transform @ vertices_homo.T).T
        return transformed[:, :3]
    
    scene = trimesh.Scene()
    
    # SmileArch (초록)
    arch_vertices = apply_transform(smile_arch_mesh.vertices, smilearch_transform)
    arch_tri = trimesh.Trimesh(vertices=arch_vertices, faces=smile_arch_mesh.faces)
    arch_tri.visual.face_colors = [100, 255, 100, 180]
    scene.add_geometry(arch_tri, node_name='smileArch')
    
    # Upper (빨강)
    upper_vertices = apply_transform(upper_mesh.vertices, upper_transform)
    upper_tri = trimesh.Trimesh(vertices=upper_vertices, faces=upper_mesh.faces)
    upper_tri.visual.face_colors = [255, 100, 100, 180]
    scene.add_geometry(upper_tri, node_name='upper')
    
    # Lower (파랑)
    lower_vertices = apply_transform(lower_mesh.vertices, lower_transform)
    lower_tri = trimesh.Trimesh(vertices=lower_vertices, faces=lower_mesh.faces)
    lower_tri.visual.face_colors = [100, 100, 255, 180]
    scene.add_geometry(lower_tri, node_name='lower')
    
    scene.show(flags={'cull': False})


def print_results(results: dict):
    """결과를 출력합니다."""
    print("\n" + "=" * 60)
    print("[Results]")
    print("=" * 60)
    
    for key, value in results.items():
        print(f"\n{key}:")
        if isinstance(value, list) and len(value) == 4:
            for row in value:
                print(f"  {row}")
        else:
            print(f"  {value}")


def main():
    parser = argparse.ArgumentParser(
        description='IOS Registration Test Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
JSON 설정 파일 예시:
{
    "smileArch": "path/to/smileArch.stl",
    "upper": "path/to/upper.stl",
    "lower": "path/to/lower.stl"
}

또는 기존 registration.py 포맷:
{
    "ios": [
        {"subType": "smileArch", "path": "..."},
        {"subType": "upper", "path": "..."},
        {"subType": "lower", "path": "..."}
    ]
}
"""
    )
    
    parser.add_argument(
        '--config', '-c',
        required=True,
        help='JSON 설정 파일 경로'
    )
    
    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='결과 시각화'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='결과 JSON 출력 파일 경로'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("IOS Registration Test")
    print("=" * 60)
    
    # 설정 로드
    print(f"\n[Config] Loading: {args.config}")
    try:
        paths = load_config(args.config)
    except Exception as e:
        print(f"[ERROR] Config loading failed: {e}")
        return 1
    
    # 경로 검증
    print("\n[Validation] Checking file paths...")
    if not validate_paths(paths):
        print("\n[ERROR] Some files are missing. Please check the paths.")
        return 1
    
    # 정합 실행
    results = asyncio.run(run_registration(paths, visualize=args.visualize))
    
    # 결과 출력
    print_results(results)
    
    # 결과 저장
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\n[Saved] Results saved to: {args.output}")
    
    print("\n[Done]")
    return 0


if __name__ == "__main__":
    exit(main())
