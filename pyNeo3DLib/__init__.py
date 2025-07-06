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

from .fastserver import *
from .registration import Neo3DRegistration

# 서브패키지 가져오기
from . import alignment
from . import faceRegisration
from . import iosRegistration
from . import fileLoader
from . import visualization
from . import teethTemplateFinder

__all__ = [
    "fastserver", 
    "registration", 
    "alignment",
    "faceRegisration",
    "iosRegistration",
    "fileLoader",
    "visualization",
    "teethTemplateFinder"
]
