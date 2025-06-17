"""
스마일 아치 랜드마크 감지 예제 스크립트
"""
# lib 모듈을 찾을 수 있도록 파이썬 경로에 상위 디렉토리 추가
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyNeo3DLib.smileArchOuterline.landmark.landmark_detector import SmileArchOuterlineDetector
import numpy as np

def align_mesh(mesh):
    """
    스마일 아치 메쉬를 정렬하는 함수
    
    Args:
        mesh: 원본 메쉬 객체
        
    Returns:
        aligned_mesh: 정렬된 메쉬 객체
        
    """

    transform_matrix_ios_with_smilearch = np.array([
      [ 0.9945144137441229,-0.006768583008681566,0.1043803963363083,-0.14194603681323126],
      [-0.10107535830535964,0.19466621425217234,0.9756479062511428,-20.966101355623444],
      [-0.02692309043763661,-0.9808461914657591,0.19291421898995373,-13.550275397218972],
      [0,0,0,1]
    ])


    return mesh.transform(transform_matrix_ios_with_smilearch, inplace=False)


def main():
    """
    스마일 아치 랜드마크 감지 테스트를 위한 메인 함수
    """


    detector = SmileArchOuterlineDetector()
    # 입력: 원본 IOS 데이터 파일 경로
    mesh_file_path = "data/ios_with_smilearch.stl"
    smile_arch_mesh = detector.load_mesh(mesh_file_path)

    # 정합이후 정렬된 메쉬( 스마일아치 프로그램내 동차변환행렬 적용)
    aligned_smile_arch_mesh = align_mesh(smile_arch_mesh)


    # 정렬된 메쉬로 랜드마크 감지
    arch_depth, molar_width, landmark_points = detector.analyze_smile_arch(aligned_smile_arch_mesh)
    print(arch_depth, molar_width, landmark_points)    
    
    # output example
    # 38.61655675576856 61.29471969604489 [[0.0, 0.15], [21.11, 9.72], [27.28, 19.27], [30.16, 28.86], [30.65, 38.47]] 


if __name__ == "__main__":
    main()