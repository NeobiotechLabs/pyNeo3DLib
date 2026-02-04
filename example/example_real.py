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

if __name__ == "__main__":
    asyncio.run(main())
