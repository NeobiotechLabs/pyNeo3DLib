# 연구노트 — pyNeo3DLib CBCT 랜드마크 파이프라인 통합

- 작성일: 2026-08-10
- 작성자: jw.go
- 대상: pyNeo3DLib (네오바이오텍 스마일 디자인 3D 기하·정합 라이브러리)

## ① 목적 / 배경

왜 이 활동을 수행했는가?

- pyNeo3DLib의 기존 CBCT 정합(cbctRegistration)은 ICP 기반 표면 정합만 수행하여, 두부계측(cephalometric)
  랜드마크 기반의 정밀 정렬 기능이 없었음.
- 사내 프로젝트 `dental-cbct-landmark`(ALI 에이전트 기반 CBCT 랜드마크 탐지)와 `dcm2nii`(DICOM→NIfTI 변환)를
  pyNeo3DLib에 모듈로 배치하여 랜드마크 탐지 기능을 통합하는 것이 이번 주 과제.
- 랜드마크 모델은 입력으로 `.nii.gz`를 요구하지만 병원 측 CBCT 데이터는 DICOM이므로,
  dcm2nii 변환 모듈의 통합이 필수였음.

## ② 수행 내용

무엇을 검토·설계·개발·시험했는가?

1. **모듈 배치(vendoring)**
   - `dental-cbct-landmark/dental_landmarks_lib` → `pyNeo3DLib/cbctLandmark/` (19개 파일 + landmark_model_registry.json).
     절대경로 import 3줄(cli.py)을 상대경로로 수정, 레거시 동적 import 경로(utils.py)·비권장 경고 문구(global_var.py)를 새 패키지명에 맞게 갱신.
   - `dcm2nii/dicom_nifti` → `pyNeo3DLib/dicomNifti/` (전부 상대경로 import라 무수정 복사).
   - 학습 가중치(13GB, 8팩·238개 .pth)는 복사하지 않고 `dental-cbct-landmark/models` 경로 참조 방식으로 결정.
2. **환경 구성**
   - 전용 venv 생성. 최소 의존성 설치: numpy 1.23.5, torch 1.12.1(CPU), torchvision 0.13.1, monai 0.7.0, itk 5.4.5, SimpleITK 2.5.3.
   - 버전 충돌 분석: monai 0.7.0(2021년)은 torch 2.x와 호환 불확실, RTX 4060(sm_89)은 CUDA 11.8+ 필요 →
     이번 검증은 호환성이 확실한 CPU 조합으로 수행(GPU 전환은 후속 과제).
3. **통합 파이프라인 개발**
   - `pyNeo3DLib/cbctLandmark/dicom_pipeline.py` 신설: DICOM 폴더 → dcm2nii 변환(.nii.gz) → ALI 에이전트 추론 → 랜드마크 좌표 반환.
   - `patient_origin=True` 기본: NIfTI 원점을 DICOM ImagePositionPatient로 유지해 환자 LPS 좌표로 출력
     (기존 cbctRegistration과 좌표계 일치 목적).
4. **콘솔 검증**
   - `example/test_cbct_landmark.py` 작성. 실 CBCT(1,466 slices, 800×800, 0.2mm spacing)로 end-to-end 성공(약 215초, CPU).
5. **버그 발견·수정**
   - vendored 코드 `agents.py`의 `SetPosAtCenter()`가 `/ 2`(실수 나눗셈)로 에이전트 위치를 float로 만들어
     monai `SpatialCrop`의 int16 캐스팅에서 `TypeError`로 중단되는 문제 발견 → 정수 나눗셈(`// 2` + astype int16)으로 수정.
     (원본 dental-cbct-landmark repo에도 동일 잠재 버그 존재 가능 → 모델 관리자에게 전달 예정)
6. **시각 검증 (3D Slicer 5.12.3)**
   - 산출물(`cbctdata.nii.gz` + `cbctdata_merged.mrk.json`)을 Slicer에서 중첩 확인.
   - 좌표별 CBCT 신호값 샘플링으로 뼈 위 여부 수치 교차 검증.
7. **웹 확인 (FastAPI)**
   - `fastserver.py`에 `POST /cbct_landmark` 엔드포인트 추가(동기 처리, 스레드풀 실행).
   - 로컬 서버(127.0.0.1:8000) 기동 후 NeoSmileArch dev tools 콘솔에서 fetch 호출로 JSON 결과 확인(약 211초).

## ③ 결과 / 판단

무엇을 확인했고 어떤 결론 또는 결정을 내렸는가?

- **파이프라인 동작 확인**: DICOM → NIfTI → 추론 → 좌표 출력이 콘솔·Slicer·웹(dev tools) 세 경로 모두에서 정상 동작.
- **랜드마크 정확도(샘플 1건)**: Gn·Pog·B·RCo·LCo 5개 중 4개(Gn, B, RCo, LCo)는 해부학적으로 타당한 위치.
  - bones intensity 확인(B 2232, RCo 1170, LCo 2005), 좌우 방향 정확(LCo가 환자 좌측),
    양과두 중점(x≈156.6)과 정중선 랜드마크 Gn·B의 x 일치.
  - **Pog는 이 샘플에서 실패**: Gn보다 17mm 아래(해부학적으로 Pog은 Gn 위), 정중선에서 13mm 이탈,
    좌표 지점 신호값이 공기(-1396) → 모델 탐지 품질 이슈로 판단. 좌표계 문제는 아님(나머지 4개 정확).
- **좌표계 판단**: 출력은 LPS mm(환자 원점). pyNeo3DLib 표준 축 방향은 RAS이므로,
  IOS/FaceScan과 통합하는 시점에 X·Y 부호 반전(LPS→RAS) 변환 적용 필요 — 이번 단계에서는 미적용으로 결정.
- **후속 과제**: ① Pog 탐지 실패 및 float/int 잠재 버그를 dental-cbct-landmark 관리자에게 전달,
  ② GPU(CUDA) 전환으로 추론 시간 단축(215초 → 수십 초 기대), ③ 엔드포인트를 백그라운드+WebSocket 패턴으로 전환,
  ④ NeoSmileArch UI 정식 연동, ⑤ LPS→RAS 변환 적용.

## ④ 관련 자료

결과를 확인할 수 있는 자료는 무엇인가?

- **Git**:
  - 아카이브: `GOJINWOO/SmileDesign` — `docs/research_note_2026-08-10_cbct-landmark/` (연구노트 + 검증 캡처 2종)
  - 코드: pyNeo3DLib 로컬 브랜치 `feature/cbctLandmark` (팀 repo 미푸시 — 업로드 정책상 개인 repo만 사용)
- **Confluence**: 본 페이지 (개인 워크스페이스 Smile Design 폴더) —
  <https://neobiotech.atlassian.net/wiki/spaces/~712020dafd2f8568bc4cf0ae7427c7fc148e72/pages/494239764>
- **Screenshot**:
  - `pyNeo3DLib-landmarks_model 추가 콘솔 로그 확인(로컬 서버).png` — dev tools 콘솔에서 엔드포인트 호출 결과 (repo 루트, 로컬 보관)
  - `output/cbct_landmark/testdata/1차.png` — 3D Slicer 시각 검증 캡처 (로컬 보관)
- **Test Result / Log**:
  - 콘솔 검증: `example/test_cbct_landmark.py` 실 (Gn 158.15/-38.05/-40.95, Pog 143.60/-48.05/-57.80,
    B 159.85/-38.25/-29.90, RCo 101.95/21.75/49.10, LCo 211.15/19.90/43.50, LPS mm)
  - 산출물: `output/cbct_landmark/cbctdata.nii/cbctdata_merged.mrk.json` (로컬 보관)
- **참고 코드**: `pyNeo3DLib/cbctLandmark/dicom_pipeline.py`, `pyNeo3DLib/dicomNifti/`, `pyNeo3DLib/fastserver.py`(`/cbct_landmark`)
