# pyNeo3DLib 설치 가이드

## 📋 요구사항

- Python 3.10.x
- Windows 10/11 (권장)
- Git

## 🚀 설치 방법

### 방법 1: 개발 모드로 설치 (권장)

개발 중이거나 코드를 수정할 경우 이 방법을 사용하세요.

```bash
# 1. 저장소 클론
git clone https://github.com/NeobiotechLabs/pyNeo3DLib.git
cd pyNeo3DLib

# 2. 가상환경 생성 및 활성화
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. 개발 모드로 설치
pip install -e .
```

### 방법 2: requirements.txt로 설치

```bash
# 1. 저장소 클론
git clone https://github.com/NeobiotechLabs/pyNeo3DLib.git
cd pyNeo3DLib

# 2. 가상환경 생성 및 활성화
python -m venv venv
venv\Scripts\activate  # Windows

# 3. requirements.txt로 설치
pip install -r requirements.txt
```

## ⚠️ 의존성 충돌 해결

### numpy 버전 충돌

`single_template_maker_lib`와 `tensorflow` 간의 numpy 버전 충돌이 발생할 수 있습니다.

#### 해결 방법 1: numpy 버전 강제 지정

```bash
# 먼저 numpy를 설치
pip install numpy==1.23.5

# 그 다음 tensorflow 설치
pip install tensorflow==2.12.0

# single_template_maker_lib를 의존성 무시하고 설치
pip install --no-deps git+https://github.com/NeobiotechLabs/Single_template_maker.git

# 나머지 패키지 설치
pip install -r requirements.txt --no-deps
pip install fastapi uvicorn pydantic scipy mediapipe open3d opencv-python pyvista imageio trimesh retina-face keras==2.12.0 scikit-image==0.22.0 qdrant-client
```

#### 해결 방법 2: 단계별 설치

```bash
# 1. 핵심 패키지 먼저 설치
pip install numpy==1.23.5 tensorflow==2.12.0 keras==2.12.0

# 2. pyNeo3DLib 개발 모드로 설치 (의존성 무시)
pip install -e . --no-deps

# 3. single_template_maker_lib 설치 (의존성 무시)
pip install --no-deps git+https://github.com/NeobiotechLabs/Single_template_maker.git

# 4. 나머지 의존성 설치
pip install fastapi uvicorn pydantic scipy mediapipe open3d opencv-python pyvista imageio trimesh retina-face scikit-image==0.22.0 qdrant-client
```

#### 해결 방법 3: pip --use-deprecated 옵션 사용

```bash
pip install --use-deprecated=legacy-resolver -e .
```

## 🔍 설치 확인

설치가 완료되면 다음 명령으로 확인할 수 있습니다:

```python
python -c "import pyNeo3DLib; print('설치 성공!')"
```

또는 서버를 실행해보세요:

```bash
python -m pyNeo3DLib.fastserver
```

서버가 성공적으로 실행되면 `http://127.0.0.1:8000/health`에서 확인할 수 있습니다.

## 🐛 문제 해결

### ImportError: numpy 관련 오류

```bash
pip uninstall numpy -y
pip install numpy==1.23.5
```

### TensorFlow 설치 실패

Windows에서 TensorFlow 2.12.0 설치 시 문제가 발생하면:

```bash
pip install tensorflow-cpu==2.12.0
```

또는:

```bash
pip install tensorflow-intel==2.12.0
```

### single_template_maker_lib 설치 실패

GitHub 접근 권한이 필요할 수 있습니다:

```bash
# SSH 키가 설정되어 있다면
pip install git+ssh://git@github.com/NeobiotechLabs/Single_template_maker.git

# HTTPS + 토큰 사용
pip install git+https://<YOUR_TOKEN>@github.com/NeobiotechLabs/Single_template_maker.git
```

### 패키지 의존성 확인

설치된 패키지와 버전을 확인하려면:

```bash
pip list
```

특정 패키지의 의존성 확인:

```bash
pip show numpy
pip show tensorflow
```

## 📦 주요 의존성 버전

| 패키지 | 버전 | 비고 |
|--------|------|------|
| numpy | 1.23.5 | TensorFlow 2.12.0과 호환 |
| tensorflow | 2.12.0 | numpy<1.24 요구 |
| keras | 2.12.0 | TensorFlow와 버전 맞춤 |
| scikit-image | 0.22.0 | |
| open3d | latest | |
| pyvista | latest | |
| fastapi | latest | |

## 🔄 업그레이드

```bash
# 저장소 업데이트
git pull

# 의존성 업데이트
pip install -e . --upgrade
```

## 🗑️ 제거

```bash
pip uninstall pyNeo3DLib
```

## 💡 팁

### 1. 가상환경 사용 (강력 권장)

다른 프로젝트와의 충돌을 방지하기 위해 항상 가상환경을 사용하세요.

```bash
# conda 사용
conda create -n pyneolib python=3.10
conda activate pyneolib

# venv 사용
python -m venv venv
venv\Scripts\activate
```

### 2. 캐시 제거 후 재설치

설치에 문제가 있을 경우:

```bash
pip cache purge
pip install -e . --no-cache-dir
```

### 3. 의존성 트리 확인

```bash
pip install pipdeptree
pipdeptree -p pyNeo3DLib
```

## 📞 지원

문제가 계속되면 다음 정보와 함께 이슈를 등록하세요:

1. Python 버전: `python --version`
2. pip 버전: `pip --version`
3. OS 정보
4. 오류 메시지 전문

```bash
# 환경 정보 수집
python --version
pip --version
pip list > installed_packages.txt
```

## 🚀 빠른 시작

설치가 완료되면:

```bash
# 서버 실행
python -m pyNeo3DLib.fastserver

# 새 터미널에서 테스트
cd example
python test_gingiva_api.py
```

자세한 사용법은 각 모듈의 README.md를 참조하세요.

