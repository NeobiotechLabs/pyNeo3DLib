"""신경관 끝점(LMeF/RMeF) CLI 진입점.

사용 예::

    python -m canal_endpoint.canal_core --input case01_nerve_canal.stl --output case01_nerve_canal_mef.mrk.json
    (또는 폴더 지정: -o ./results/ → case01_nerve_canal_mef.mrk.json 으로 저장)
"""

import sys
from pathlib import Path

_PIPELINE_ROOT = Path(__file__).resolve().parents[2]
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from canal_endpoint.find_canal_endpoints import main

if __name__ == "__main__":
    main()
