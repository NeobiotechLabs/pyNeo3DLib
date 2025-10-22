# 3점 정합 API 사용 가이드

## 개요

3점 정합 API는 두 개의 3D 메시와 각 메시에서 선택된 3개 이상의 점 좌표를 사용하여 ICP 알고리즘으로 정합을 수행하고 변환 행렬을 반환합니다.

## API 엔드포인트

```
POST /threepoint_registration
```

## 입력 형식

```json
{
    "target_mesh": {
        "path": "타겟 메시 파일 경로 (.stl, .ply 등)"
    },
    "source_mesh": {
        "path": "소스 메시 파일 경로 (.stl, .ply 등)"
    },
    "target_points": [
        {"x": 10.5, "y": 20.3, "z": 15.7},
        {"x": 25.1, "y": 18.9, "z": 12.4},
        {"x": 35.8, "y": 22.1, "z": 18.2}
    ],
    "source_points": [
        {"x": 12.3, "y": 19.8, "z": 16.1},
        {"x": 27.4, "y": 17.5, "z": 13.8},
        {"x": 38.2, "y": 21.7, "z": 19.5}
    ],
    "region_growing_radius": 5.0,
    "icp_max_iterations": 1000,
    "normal_similarity_threshold": 0.8,
    "visualization": false
}
```

### 필수 매개변수

- `target_mesh.path`: 타겟 메시 파일 경로
- `source_mesh.path`: 소스 메시 파일 경로  
- `target_points`: 타겟 메시의 점 좌표 배열 (최소 3개)
- `source_points`: 소스 메시의 점 좌표 배열 (최소 3개, target_points와 개수 일치)

### 선택적 매개변수

- `region_growing_radius`: Region growing 반경 (기본값: 5.0)
- `icp_max_iterations`: ICP 최대 반복 횟수 (기본값: 1000)
- `normal_similarity_threshold`: 법선 벡터 유사성 임계값 (기본값: 0.8)
- `visualization`: 시각화 여부 (기본값: false)

## 출력 형식

### 성공 시

```json
{
    "status": "success",
    "transformation_matrix": [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ],
    "request_id": "abc123def4",
    "message": "3점 정합이 성공적으로 완료되었습니다.",
    "parameters": {
        "target_points_count": 3,
        "source_points_count": 3,
        "region_growing_radius": 5.0,
        "icp_max_iterations": 1000,
        "normal_similarity_threshold": 0.8
    },
    "timestamp": "2024-10-21 15:30:45"
}
```

### 실패 시

```json
{
    "status": "error",
    "message": "3점 정합 중 오류가 발생했습니다: 파일을 찾을 수 없습니다",
    "request_id": "abc123def4",
    "timestamp": "2024-10-21 15:30:45"
}
```

## 동작 과정

1. **메시 로딩**: 타겟과 소스 메시 파일을 로드
2. **최근접 정점 찾기**: 각 점 좌표에서 KDTree를 사용하여 가장 가까운 메시 정점 찾기
3. **Region Growing**: 각 정점 주변에서 법선 벡터 유사성을 기반으로 영역 확장
4. **ICP 정합**: 선택된 영역들을 사용하여 ICP 알고리즘으로 정합 수행
5. **변환 행렬 반환**: 소스 메시를 타겟 메시로 정합하는 4x4 변환 행렬 반환

## 사용 예제

### Python 코드 예제

```python
import requests
import json

# API 호출
url = "http://127.0.0.1:8000/threepoint_registration"
data = {
    "target_mesh": {"path": "/path/to/target.stl"},
    "source_mesh": {"path": "/path/to/source.stl"},
    "target_points": [
        {"x": 10.0, "y": 20.0, "z": 15.0},
        {"x": 25.0, "y": 18.0, "z": 12.0},
        {"x": 35.0, "y": 22.0, "z": 18.0}
    ],
    "source_points": [
        {"x": 12.0, "y": 19.0, "z": 16.0},
        {"x": 27.0, "y": 17.0, "z": 13.0},
        {"x": 38.0, "y": 21.0, "z": 19.0}
    ]
}

response = requests.post(url, json=data)
result = response.json()

if result["status"] == "success":
    transformation_matrix = result["transformation_matrix"]
    print("변환 행렬:", transformation_matrix)
else:
    print("오류:", result["message"])
```

### cURL 예제

```bash
curl -X POST "http://127.0.0.1:8000/threepoint_registration" \
     -H "Content-Type: application/json" \
     -d @sampleInput_threepoint.json
```

## 테스트 실행

```bash
# 서버 시작
python -m pyNeo3DLib.fastserver

# 다른 터미널에서 테스트 실행
cd example
python example_threepoint_registration.py
```

## 주의사항

1. **점 좌표 정확성**: 입력된 점 좌표가 실제 메시 표면 근처에 있어야 합니다.
2. **파일 형식**: STL, PLY 등의 3D 메시 파일 형식을 지원합니다.
3. **점 개수**: 타겟과 소스의 점 개수가 일치해야 하며, 최소 3개 이상이어야 합니다.
4. **메모리 사용량**: 큰 메시 파일의 경우 메모리 사용량이 클 수 있습니다.
5. **처리 시간**: 메시 크기와 매개변수에 따라 처리 시간이 달라질 수 있습니다.

## 매개변수 조정 가이드

- **region_growing_radius**: 값이 클수록 더 넓은 영역을 선택하지만 정확도가 떨어질 수 있음
- **normal_similarity_threshold**: 값이 클수록 더 유사한 법선을 가진 영역만 선택
- **icp_max_iterations**: 값이 클수록 더 정확한 정합이 가능하지만 처리 시간 증가
