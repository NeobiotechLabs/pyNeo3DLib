# IOS Initial Alignment Module

## 개요
IOS(Intraoral Scan) 스캔 데이터의 초기 정합을 위한 통합 모듈입니다.

## 모듈 구조

```
ios_initial_alignment/
├── __init__.py                 # 모듈 진입점
├── initial_alignment/          # 초기 정합 알고리즘
│   ├── mesh_aligner.py        # 메시 정렬 메인 클래스
│   ├── preprocessing.py       # 메시 전처리
│   ├── mesh_loader.py         # 메시 로딩
│   ├── initial_alignment_finder.py  # OBB 기반 초기 정렬
│   ├── icp_registration.py    # ICP 정합
│   ├── obb_analyzer.py        # OBB 분석
│   ├── transform_calculator.py # 변환 행렬 계산
│   ├── distance_calculator.py  # 거리 계산
│   └── constants.py           # 상수 정의
├── global_fit/                # 메시 변환 유틸리티
│   ├── mesh_converter.py      # 메시/포인트클라우드 변환
│   └── constants.py           # 상수 정의
└── utils/                     # 메시 I/O 유틸리티
    └── mesh_io.py             # 메시 파일 로딩
```

## 사용 방법

### 기본 사용
```python
from pyNeo3DLib.ios_initial_alignment import MeshAligner

# MeshAligner 사용
aligner = MeshAligner()
result = aligner.align_from_files(source_path, target_path)
```

### 함수 기반 사용
```python
from pyNeo3DLib.ios_initial_alignment import align_3d_meshes

# STL 파일로부터 직접 정렬
result = align_3d_meshes(source_path, target_path)
```

## 변경 이력

### 2025-12-04 (최신)
- `__align_3d_meshes` 호출 시 사용되지 않는 코드 및 파일 제거
- `mesh_alignment.py` (레거시 함수) 제거
- `global_fit/deviationMap/` 폴더 및 시각화 관련 파일 제거
- `global_fit/common_utils/` 폴더 구조 단순화 (mesh_converter.py를 global_fit/로 이동)
- 모든 임포트 경로 최적화

### 2025-12-04 (이전)
- 기존 `global_fit/`, `initial_alignment/`, `utils/` 폴더를 `ios_initial_alignment/` 폴더로 통합
- 모든 임포트 경로를 `pyNeo3DLib.ios_initial_alignment.*`로 변경
- 모듈 구조 개선 및 네이밍 명확화

## 임포트 경로

### 메인 사용
```python
from pyNeo3DLib.ios_initial_alignment import MeshAligner, align_3d_meshes
```

### 세부 모듈 사용
```python
from pyNeo3DLib.ios_initial_alignment.initial_alignment import (
    MeshAligner,
    MeshPreprocessor,
    MeshLoader,
    InitialAlignmentFinder,
    ICPRegistration
)
from pyNeo3DLib.ios_initial_alignment.global_fit import MeshConverter
from pyNeo3DLib.ios_initial_alignment.utils import load_mesh_safe
```

## 주요 클래스

### MeshAligner
메시 정렬의 전체 프로세스를 조율하는 메인 클래스

### InitialAlignmentFinder
OBB(Oriented Bounding Box) 기반 초기 정렬을 탐색

### ICPRegistration
ICP(Iterative Closest Point) 정합 수행

### MeshConverter
메시와 포인트클라우드 간 변환을 담당

