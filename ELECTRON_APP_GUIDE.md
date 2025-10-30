# Electron 앱에서 pyNeo3DLib 사용 가이드 (NeoSmileArch)

## 🎯 개요

NeoSmileArch와 같은 Electron 앱에서 pyNeo3DLib를 사용할 때는 **mediapipe 의존성 없이** 설치하여 DLL 로드 문제를 방지합니다.

## ⚠️ 중요: mediapipe 의존성 문제

Electron 앱의 내장 Python 환경에서 mediapipe를 사용하면 다음 오류가 발생할 수 있습니다:

```
ImportError: DLL load failed while importing _framework_bindings: 
DLL 초기화 루틴이 실패했습니다.
```

**해결책:** mediapipe를 **선택적 의존성**으로 분리했습니다.

## 📦 설치 방법

### NeoSmileArch 프로젝트에서 설치

```bash
# Electron 앱의 Python 환경에서
cd F:\work\neo_git\working_now\pyNeo3DLib

# mediapipe 없이 설치 (기본)
pip install -e .

# 또는 requirements.txt 사용
pip install -r requirements.txt
```

## ✅ 설치 확인

### 1. msvc-runtime 확인

```python
import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-m", "pip", "show", "msvc-runtime"],
    capture_output=True, text=True
)

if result.returncode == 0:
    print("[OK] msvc-runtime 설치됨")
else:
    print("[FAIL] msvc-runtime 설치 필요")
    print("설치: pip install msvc-runtime")
```

### 2. pyNeo3DLib import 테스트

```python
# mediapipe가 로드되지 않는지 확인
import sys

# pyNeo3DLib import
import pyNeo3DLib

if 'mediapipe' in sys.modules:
    print("[FAIL] mediapipe가 자동으로 로드됨")
else:
    print("[OK] mediapipe가 로드되지 않음")
```

### 3. GingivaGenerator 사용 테스트

```python
from pyNeo3DLib.gingivaGenerator import GingivaGenerator

# 인스턴스 생성
generator = GingivaGenerator()
print("[OK] GingivaGenerator 사용 가능")
```

## 🔧 사용 예시

### 치은 생성 API 사용

```python
from pyNeo3DLib.gingivaGenerator import GingivaGenerator

async def generate_gingiva_example():
    generator = GingivaGenerator()
    
    result = await generator.generate_gingiva(
        input_path="path/to/teeth/files",
        output_path="path/to/output",
        arch_types=["mandibular", "maxillary"],
        request_id="unique_request_id"
    )
    
    if result["status"] == "success":
        print("치은 생성 완료!")
        for file_info in result["generated_files"]:
            print(f"생성된 파일: {file_info['file_path']}")
    else:
        print(f"오류: {result['error']}")
```

### FastAPI 서버로 사용

```python
from pyNeo3DLib.fastserver import run_server

# 서버 실행 (http://127.0.0.1:8000)
run_server()
```

## 🐛 문제 해결

### DLL 로드 오류가 계속 발생하는 경우

#### 1. Visual C++ Redistributable 설치 확인

Windows 시스템에 Visual C++ Redistributable이 설치되어 있어야 합니다:

- [Microsoft Visual C++ 2015-2022 Redistributable 다운로드](https://aka.ms/vs/17/release/vc_redist.x64.exe)

#### 2. msvc-runtime 재설치

```bash
pip uninstall msvc-runtime -y
pip install msvc-runtime
```

#### 3. pyNeo3DLib 재설치

```bash
cd F:\work\neo_git\working_now\pyNeo3DLib
pip uninstall pyNeo3DLib -y
pip install -e .
```

#### 4. Python 환경 확인

```python
import sys
print(f"Python 경로: {sys.executable}")
print(f"Python 버전: {sys.version}")
```

Electron 앱이 올바른 Python 인터프리터를 사용하는지 확인하세요.

### mediapipe가 필요한 경우

Face Registration 기능이 필요하면:

```bash
pip install -e .[full]
```

하지만 Electron 앱 환경에서는 권장하지 않습니다.

## 📝 주요 변경 사항

### v1.0.0 (2025-10-30)

- ✅ **mediapipe를 선택적 의존성으로 변경**
  - 기본 설치: mediapipe 제외
  - 전체 설치: `pip install -e .[full]`

- ✅ **Lazy Import 적용**
  - `import pyNeo3DLib`만으로 mediapipe가 로드되지 않음
  - 필요한 모듈만 명시적으로 import

- ✅ **의존성 순서 최적화**
  - msvc-runtime을 가장 먼저 설치
  - numpy 버전 고정 (1.23.5)

## 🔗 관련 문서

- [설치 가이드](INSTALL.md)
- [치은 생성 API 문서](pyNeo3DLib/gingivaGenerator/README.md)
- [FastAPI 서버 문서](pyNeo3DLib/fastserver.py)

## 💬 지원

문제가 발생하면 다음 정보와 함께 이슈를 등록하세요:

```python
import sys
import subprocess

print(f"Python 버전: {sys.version}")
print(f"Python 경로: {sys.executable}")

# 설치된 패키지 목록
result = subprocess.run(
    [sys.executable, "-m", "pip", "list"],
    capture_output=True, text=True
)
print(result.stdout)
```

