"""
메인 템플릿 생성 클래스
"""

import os
from typing import List
import time

from single_template_maker_lib import TeethTemplateMaker

if __name__ == "__main__":

    start_time = time.time()

    # 기본 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    teeth_input_path = os.path.join(script_dir, "data/input")
    gingiva_output_path = os.path.join(script_dir, "f:\work\smile\output")
    # 생성할 치아탬플릿릿 타입 리스트
    maxilla_madibular_type_list = ["mandibular"]
    
    # 템플릿 생성기 인스턴스 생성 및 실행
    template_maker = TeethTemplateMaker(teeth_input_path, gingiva_output_path, maxilla_madibular_type_list)
    template_maker.run()

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"치아탬플릿 생성 소요 시간: {minutes}분 {seconds}초")


    