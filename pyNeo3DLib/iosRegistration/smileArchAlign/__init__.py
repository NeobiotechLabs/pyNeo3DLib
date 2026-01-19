"""
SmileArch Alignment Library
===========================

SmileArch 메쉬를 SmileGuide에 정합하는 라이브러리입니다.

사용 예시:
    from smileArchAlign import align_to_smileguide
    
    transform = align_to_smileguide("smileArch.stl", "smileguide.stl")
    
    source = trimesh.load("smileArch.stl")
    source.apply_transform(transform)  # 이제 SmileGuide에 정합됨
"""

from .align_to_smileguide import align_to_smileguide

__all__ = ['align_to_smileguide']
