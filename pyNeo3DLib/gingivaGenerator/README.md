# 치은(Gingiva) 생성 API 가이드

## 📖 개요

`GingivaGenerator`는 치아 입력 데이터로부터 치은(잇몸) 메쉬를 생성하는 모듈입니다.

## 🚀 API 사용 방법

### 엔드포인트

```
POST /generate_gingiva
```

### 요청 본문

```json
{
  "input_path": "/path/to/teeth/input/files",
  "output_path": "/path/to/output/directory",
  "arch_types": ["maxilla", "mandibular"]
}
```

#### 파라미터 설명

| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| `input_path` | string | ✅ | 치아 입력 파일들이 있는 디렉토리 경로 |
| `output_path` | string | ✅ | 생성된 치은 파일을 저장할 디렉토리 경로 |
| `arch_types` | array | ❌ | 생성할 치은 타입 리스트 (기본값: `["mandibular"]`) |

#### arch_types 옵션

- `"maxilla"`: 상악 치은 생성
- `"mandibular"`: 하악 치은 생성
- 둘 다 포함 가능: `["maxilla", "mandibular"]`

### 응답

#### 성공 응답 (200 OK)

```json
{
  "status": "processing",
  "message": "치은 생성이 시작되었습니다. 결과는 WebSocket을 통해 전송됩니다.",
  "request_id": "abc123XYZ",
  "input_path": "/path/to/input",
  "output_path": "/path/to/output",
  "arch_types": ["mandibular"],
  "timestamp": "2025-10-30 16:00:00"
}
```

#### 오류 응답

```json
{
  "status": "error",
  "message": "입력 경로가 존재하지 않습니다: /invalid/path",
  "request_id": "xyz789ABC",
  "timestamp": "2025-10-30 16:00:00"
}
```

## 📡 WebSocket 알림

치은 생성 과정은 비동기로 진행되며, 진행 상황은 WebSocket을 통해 전송됩니다.

### WebSocket 연결

```javascript
const ws = new WebSocket('ws://127.0.0.1:8000/ws');
```

### 메시지 타입

#### 1. 시작 알림

```json
{
  "type": "gingiva_generation_started",
  "request_id": "abc123XYZ",
  "message": "치은 생성이 시작되었습니다.",
  "timestamp": "2025-10-30 16:00:00"
}
```

#### 2. 완료 알림

```json
{
  "type": "gingiva_generation_completed",
  "request_id": "abc123XYZ",
  "generated_files": [
    {
      "arch_type": "mandibular",
      "file_path": "/path/to/output/mandibular.stl"
    }
  ],
  "message": "치은 생성이 완료되었습니다.",
  "timestamp": "2025-10-30 16:01:30"
}
```

#### 3. 실패 알림

```json
{
  "type": "gingiva_generation_failed",
  "request_id": "abc123XYZ",
  "error": "오류 메시지",
  "timestamp": "2025-10-30 16:00:15"
}
```

## 🧪 테스트 방법

### 방법 1: Python 스크립트

```bash
# 서버 실행
python -m pyNeo3DLib.fastserver

# 다른 터미널에서 테스트 스크립트 실행
cd example
python test_gingiva_api.py
```

### 방법 2: curl 명령어

```bash
curl -X POST http://127.0.0.1:8000/generate_gingiva \
  -H "Content-Type: application/json" \
  -d '{
    "input_path": "F:\\work\\neo_git\\working_now\\pyNeo3DLib\\example\\data\\input",
    "output_path": "F:\\work\\neo_git\\working_now\\pyNeo3DLib\\example\\output",
    "arch_types": ["mandibular"]
  }'
```

### 방법 3: Python requests

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/generate_gingiva",
    json={
        "input_path": "F:\\work\\neo_git\\working_now\\pyNeo3DLib\\example\\data\\input",
        "output_path": "F:\\work\\neo_git\\working_now\\pyNeo3DLib\\example\\output",
        "arch_types": ["mandibular"]
    }
)

print(response.json())
```

### 방법 4: WebSocket 통합 테스트 (권장)

브라우저에서 HTML 파일 열기:

```bash
# example 디렉토리에서
# test_gingiva_websocket.html 파일을 브라우저로 열기
```

웹 인터페이스에서:
1. "WebSocket 연결" 버튼 클릭
2. 입력/출력 경로 설정
3. "치은 생성 시작" 버튼 클릭
4. 실시간으로 진행 상황 확인

## 💡 코드 예제

### 직접 클래스 사용

```python
from pyNeo3DLib.gingivaGenerator import GingivaGenerator
import asyncio

async def main():
    generator = GingivaGenerator()
    
    result = await generator.generate_gingiva(
        input_path="./data/input",
        output_path="./data/output",
        arch_types=["mandibular"],
        request_id="test_001"
    )
    
    if result["status"] == "success":
        print("생성 완료!")
        for file_info in result["generated_files"]:
            print(f"- {file_info['arch_type']}: {file_info['file_path']}")
    else:
        print(f"오류 발생: {result['error']}")

asyncio.run(main())
```

## 📂 입력 파일 구조

입력 디렉토리는 다음과 같은 구조를 가져야 합니다:

```
input/
├── Crown 11.stl
├── Crown 12.stl
├── Crown 13.stl
├── ...
├── Crown 47.stl
└── metadata.json
```

## 📦 출력 파일

생성된 치은 파일은 다음과 같이 저장됩니다:

```
output/
├── mandibular.stl  (하악 치은)
└── maxilla.stl   (상악 치은, 요청 시)
```

## ⚠️ 주의사항

1. **입력 경로**: 반드시 존재하는 디렉토리여야 합니다.
2. **출력 경로**: 없는 경우 자동으로 생성됩니다.
3. **처리 시간**: 치은 생성은 1~2분 정도 소요될 수 있습니다.
4. **WebSocket**: 실시간 진행 상황을 확인하려면 WebSocket 연결이 필요합니다.

## 🔧 문제 해결

### 서버가 실행되지 않는 경우

```bash
# 포트 사용 확인
netstat -ano | findstr :8000

# 다른 프로세스가 포트를 사용 중이면 종료
```

### 입력 파일이 없는 경우

```
❌ 오류: 입력 경로가 존재하지 않습니다
```

→ `input_path`가 올바른지 확인하세요.

### WebSocket 연결 실패

```
❌ WebSocket 연결 실패
```

→ 서버가 실행 중인지 확인하고, 방화벽 설정을 확인하세요.

## 📚 관련 모듈

- `registration.py`: 3D 정합 관련
- `teethTemplateFinder`: 치아 템플릿 검색
- `threePointRegistration`: 3점 정합

## 🤝 기여

버그 리포트나 기능 제안은 이슈로 등록해주세요.

