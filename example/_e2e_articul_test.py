"""E2E smoke: 실제 real_Input.json 경로로 정합 후 통합 파이프라인 흐름 검증.

registration.py 의 __run_articul_pipeline 이 하는 일과 동일하게
bridge.run_articul_pipeline(cbct.path, pipeline_results.path) 를 실행하고
케이스별 최종 산출물(생성 메쉬 STL · 통합 랜드마크 · 평면 JSON 경로)과
교합평면·시상정중면 중심/법선을 프린팅한다.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 리다이렉트(cp949) 환경에서 장식 문자 출력 시 UnicodeEncodeError 방지
from pyNeo3DLib.articulPipeline.run_integrated_pipeline import _configure_stdio_utf8
_configure_stdio_utf8()

from pyNeo3DLib.articulPipeline import registration_bridge as bridge

with open(Path(__file__).parent / "real_Input.json", encoding="utf-8") as f:
    cfg = json.load(f)

cbct_dir = cfg["cbct"]["path"]
results_dir = cfg["pipeline_results"]["path"]

print(f"--input  : {cbct_dir}")
print(f"--output : {results_dir}")

rc = bridge.run_articul_pipeline(cbct_dir, results_dir)
print(f"\nexit_code = {rc}")

artifacts = bridge.collect_case_artifacts(results_dir)
bridge.print_case_artifacts(artifacts)
assert artifacts, "생성된 세그멘테이션 산출물(STL)이 없습니다"

planes = bridge.collect_plane_results(results_dir)
bridge.print_plane_results(planes)

message = bridge.plane_success_message(planes, results_dir, artifacts)
json.dumps(message, allow_nan=False)  # 외부 전송 직렬화 가능 확인
print(json.dumps(message["artifacts"], indent=2, ensure_ascii=False))

computed = [p for p in planes if p.get("computed")]
assert computed, "computed 평면 결과가 없습니다"
print("\nE2E_OK")
