"""
메쉬 I/O 유틸리티
파일 입출력 관련 공통 기능을 제공합니다.
"""

import os
import shutil
import tempfile
from pathlib import Path
import open3d as o3d


def load_mesh_safe(file_path: str) -> o3d.geometry.TriangleMesh:
    """
    파일 경로에서 메쉬를 안전하게 로드합니다.
    Windows 환경에서 한글이나 특수문자가 포함된 경로를 처리하기 위해
    필요한 경우 임시 파일을 사용합니다.
    
    Args:
        file_path: 메쉬 파일 경로
        
    Returns:
        Open3D TriangleMesh 객체
        
    Raises:
        FileNotFoundError: 파일이 존재하지 않는 경우
        Exception: 메쉬 로드 실패
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    # Windows에서 한글/특수문자 경로 문제 해결을 위한 임시 파일 사용
    if os.name == 'nt' and any(ord(c) > 127 for c in str(file_path)):
        suffix = Path(file_path).suffix
        # delete=False는 Windows에서 파일을 닫은 후 다시 열기 위해 필요
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            temp_path = tmp.name
        
        try:
            shutil.copy2(file_path, temp_path)
            mesh = o3d.io.read_triangle_mesh(temp_path)
        except Exception:
            # 실패 시 정리 후 예외 전파
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
            raise
        finally:
            # 성공여부와 상관없이 임시 파일 삭제 시도
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
    else:
        mesh = o3d.io.read_triangle_mesh(file_path)
        
    return mesh

