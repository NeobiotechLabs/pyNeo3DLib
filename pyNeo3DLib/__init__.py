"""
pyNeo3DLib - 3D Dental Library

Lazy import를 사용하여 필요한 모듈만 로드됩니다.

사용 예시:
    # 치은 생성만 사용
    from pyNeo3DLib.gingivaGenerator import GingivaGenerator
    
    # FastAPI 서버 실행
    from pyNeo3DLib.fastserver import run_server
    
    # 전체 registration 기능 사용
    from pyNeo3DLib.registration import Neo3DRegistration
"""

import os
import logging
import warnings

# TensorFlow 경고 메시지 숨기기 (import 전에 설정해야 함)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # GPU 사용 비활성화

# TensorFlow 관련 경고 메시지 필터링
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=FutureWarning, module='tensorflow')

# TensorFlow 로깅 레벨 설정
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Lazy import: 실제 사용 시점에만 import
# from .fastserver import *  # 제거 - 필요시 명시적으로 import
# from .registration import Neo3DRegistration  # 제거 - 필요시 명시적으로 import

# 서브패키지는 명시적으로 import하도록 변경
# from . import alignment  # 제거
# from . import faceRegisration  # 제거
# from . import iosRegistration  # 제거
# from . import fileLoader  # 제거
# from . import visualization  # 제거
# from . import teethTemplateFinder  # 제거
# from . import goldenProportion  # 제거

__version__ = "1.0.0"

__all__ = [
    "fastserver", 
    "registration", 
    "alignment",
    "faceRegisration",
    "iosRegistration",
    "fileLoader",
    "visualization",
    "teethTemplateFinder",
    "goldenProportion",
    "gingivaGenerator",
    "threePointRegistration"
]
