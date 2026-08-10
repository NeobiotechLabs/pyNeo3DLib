"""
cbctLandmark: CBCT 볼륨에서 ALI 에이전트 기반 치과 랜드마크 좌표 추론 라이브러리.

원본: dental-cbct-landmark 프로젝트의 dental_landmarks_lib 패키지를 배치(vendoring)함.
출력 좌표계는 LPS mm (pyNeo3DLib 표준인 RAS와 다르므로 통합 시 변환 필요).

사용 예시
---------
from pyNeo3DLib.cbctLandmark import predict

results = predict(
    volume="path/to/volume.nii.gz",
    models_dir="path/to/models",
    landmarks=["Gn", "Pog", "B", "RCo", "LCo"],
    output_dir="path/to/output",
)
"""

from .config import PredictConfig
from .environment import Environment
from .pipeline import predict
from .types import LandmarkCoord, PredictResult

__all__ = [
    "Environement",
    "Environment",
    "LandmarkCoord",
    "PredictConfig",
    "PredictResult",
    "predict",
]


def __getattr__(name: str):
    if name == "Environement":
        from .environment import Environement

        return Environement
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
