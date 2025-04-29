from pyNeo3DLib import fastserver

import requests
import time
import json
import threading
import asyncio

print("Server started, http://localhost:8000")


fastserver.start_server()


from pyNeo3DLib import Neo3DRegistration




print("\nPress Enter to stop server...")
input()

async def main():
    with open(f"{__file__}/../sampleInput.json", "r") as f:
        json_string = f.read()
        reg = Neo3DRegistration(json_string, fastserver.ws)
        print(reg.version)
        print(reg.parsed_json)
        result = await reg.run_registration(visualize=True)
    print(result)


asyncio.run(main())
input()

fastserver.stop_server()
print("Server stopped")




