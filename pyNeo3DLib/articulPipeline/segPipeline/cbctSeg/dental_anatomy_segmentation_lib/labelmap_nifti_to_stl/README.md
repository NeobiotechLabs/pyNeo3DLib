# labelmap_nifti_to_stl

**라벨맵 NIfTI**를 **참조 볼륨 NIfTI(예: CBCT)와 같은 ITK 격자**로 맞춘 뒤, 라벨별 **STL / OBJ / PLY**로 저장합니다.  
PyTorch·nnU-Net 없이 동작합니다.

## 다른 프로젝트에 그대로 쓰기

1. 이 **`labelmap_nifti_to_stl` 폴더 전체**를 원하는 저장소로 복사합니다.
2. 그 폴더에서 의존성을 설치합니다.

```bash
cd labelmap_nifti_to_stl
pip install -e .
```

3. 코드에서 한 번에 호출합니다.

```python
from pathlib import Path
from labelmap_nifti_to_stl import run_align_prediction_to_meshes

result = run_align_prediction_to_meshes(
    prediction_nifti=Path("segmentation.nii.gz"),   # 예측 라벨맵 (아무 격자)
    reference_nifti=Path("cbct_volume.nii.gz"),     # 기준이 될 입력 볼륨
    mesh_output_dir=Path("out_meshes"),
    dataset_json=Path("dataset.json"),              # 선택: 라벨 id → 이름(파일명)
)
print(result.aligned_labelmap_nifti)
print(result.mesh_files)
```

- 격자만 맞춘 NIfTI만 필요하면: `run_align_labelmap_to_reference(...)`  
- 이미 참조와 같은 격자인 라벨맵만 메시화: `export_meshes_from_label_nifti(...)`

## CLI (설치 후)

```bash
python -m labelmap_nifti_to_stl reference.nii.gz labelmap.nii.gz ./out_meshes --dataset-json dataset.json
```

## 의존성

`numpy`, `itk`, `trimesh`, `vtk`, `pyvista` — `pyproject.toml`의 `[project.dependencies]`와 동일합니다.

## 설치 없이 (임시)

단독 복사 시: **PYTHONPATH에 `labelmap_nifti_to_stl` 폴더의 부모 경로**를 넣으면 `import labelmap_nifti_to_stl` 이 됩니다.

이 저장소 전체를 쓸 때는 루트에서 **`pip install -e .`** 로 설치하는 것이 가장 단순합니다 (`pyproject.toml`의 `package-dir` 매핑).
