"""
CBCT Depth Map 모듈

레이캐스팅을 통한 CBCT 깊이 맵 추출 관련 모듈들을 제공합니다.

주요 클래스:
- CBCTDepthMapExtractor: 메인 조정 클래스
- RayGridGenerator: 격자 생성
- RayCaster: 레이캐스팅 실행
- DepthMapVisualizer: 시각화

사용 예제:
```python
from pyNeo3DLib.cbctRegistration.processing.depth_map import CBCTDepthMapExtractor

extractor = CBCTDepthMapExtractor(
    pts_face=pts_face,
    grid_center=[77.7, 85.0, 94.23],
    grid_width_mm=80.0,
    grid_height_mm=100.0,
    grid_resolution=(50, 50),
)

result = extractor.extract()
extractor.visualize_3d()
```
"""

from .extractor import CBCTDepthMapExtractor
from .ray_grid_generator import RayGridGenerator
from .ray_caster import RayCaster
from .depth_map_visualizer import DepthMapVisualizer
from .dicom_loader import CBCTDicomLoader
from .surface_extractor import CBCTSurfaceExtractor

__all__ = [
    "CBCTDepthMapExtractor",
    "RayGridGenerator",
    "RayCaster",
    "DepthMapVisualizer",
    "CBCTDicomLoader",
    "CBCTSurfaceExtractor",
]

__version__ = "1.0.0"

