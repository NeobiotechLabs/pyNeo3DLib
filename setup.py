from setuptools import setup, find_packages

setup(
    name="pyneo3dlib",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
    ],
    author="NeoBiotech",
    description="3D 데이터 처리를 위한 파이썬 라이브러리",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/사용자명/pyNeo3DLib",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
