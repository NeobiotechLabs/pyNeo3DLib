"""
치아 잇몸(Gingiva) 생성 라이브러리

이 라이브러리는 STL 파일을 처리하여 잇몸을 생성하는 기능을 제공합니다.
"""

__version__ = '0.1.0'

# 필요한 클래스만 라이브러리에서 직접 가져오기
from pyNeo3DLib.smileArchOuterline.core.arch_curve_finder import analyze_upper_IOS_scandata

# 외부에 노출할 이름 목록을 제한
__all__ = ['analyze_upper_IOS_scandata']
