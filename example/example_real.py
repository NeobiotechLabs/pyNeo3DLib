import asyncio
import sys
import os

# 로컬 pyNeo3DLib 모듈을 우선 import하도록 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyNeo3DLib.registration import Neo3DRegistration

async def main():
    with open(f"{__file__}/../real_Input.json", "r", encoding="utf-8") as f:
        json_string = f.read()
    
    reg = Neo3DRegistration(json_string)
    print(reg.version)
    print(reg.parsed_json)
    result = await reg.run_registration(visualize=True)
    print(result)

    # 정합 시작 시점에 백그라운드로 시작되어 정합과 병렬로 돌아가는
    # 통합 파이프라인(articulPipeline) 대기.
    # --input: cbct.path / --output: pipeline_results.path 로 실행되며,
    # 완료되면 교합평면·시상정중면(MSP) 중심/법선 벡터가 프린팅된다.
    if getattr(reg, "articul_pipeline_task", None) is not None:
        await reg.articul_pipeline_task

if __name__ == "__main__":
    asyncio.run(main())
