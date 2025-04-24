from setuptools import setup, find_packages

setup(
    name="pyNeo3DLib",
    version="0.1.0",
    packages=find_packages(
        include=["pyNeo3DLib", "pyNeo3DLib.*"]
    ),
    package_data={
        "pyNeo3DLib": ["*.stl"]
    },
    install_requires=[
        "numpy",
        "fastapi",
        "uvicorn",
        "pydantic",
        "scipy",
        "mediapipe",
        "open3d",
        "opencv-python"
    ],
    author="NeoBiotech",
    description="3D 데이터 처리를 위한 파이썬 라이브러리",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/NeobiotechLabs/pyNeo3DLib",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
