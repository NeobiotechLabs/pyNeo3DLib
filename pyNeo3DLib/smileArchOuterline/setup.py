import os
from setuptools import setup, find_packages

# 스크립트 파일의 디렉토리 경로 구하기
script_dir = os.path.dirname(os.path.abspath(__file__))
# requirements.txt의 정확한 경로 구하기
req_path = os.path.join(script_dir, "requirements.txt")

# requirements.txt 파일 읽기
with open(req_path) as f:
    requirements = f.read().splitlines()

# 특정 버전 제한이 필요한 패키지들
# Python 3.8 호환성을 위해 pyparsing 버전 제한
requirements.append("pyparsing<3.1.0")

setup(
    name="analyzing_IOS",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    description="Analyzing IOS library",
    keywords="IOS, dental, stl, arch_curve_finder",
    python_requires=">=3.10",  # 최소 Python 버전 명시
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
)