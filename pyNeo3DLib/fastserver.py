from fastapi import FastAPI, WebSocket, BackgroundTasks, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import threading
import os, signal
import asyncio
import random
import string
import datetime
from typing import Dict, Any
import json

from .registration import Neo3DRegistration

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

s_thread = None
ws = None

async def process_registration_async(registration_data, request_id):
    try:
        reg = Neo3DRegistration(json.dumps(registration_data))
        print(f"[{request_id}] Registration started")
        result = await reg.run_registration(visualize=False)
        print(f"[{request_id}] Registration completed")
        
        global ws
        if ws:
            await ws.send_json({
                "type": "registration_completed",
                "request_id": request_id,
                "result": result,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

        return result
    except Exception as e:
        print(f"[{request_id}] 정합 중 오류 발생: {str(e)}")
        if ws:
            await ws.send_json({
                "type": "registration_failed",
                "request_id": request_id,
                "error": str(e),
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global ws
    ws = websocket
    await websocket.accept()
    while True:
        # random_text = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        # # 현재 시간
        # current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # # 데이터 전송
        # await websocket.send_json({
        #     "random_text": random_text,
        #     "timestamp": current_time
        # })
        
        # 1초 대기
        await asyncio.sleep(1)
        data = await websocket.receive_text()
        print(f"Received: {data}")
        

@app.post("/registration")
async def get_registration(background_tasks: BackgroundTasks, registration: Dict[str, Any] = Body(...)):
    
    request_id = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    print(f"[{request_id}] 정합 API 호출됨")
    
    reg = Neo3DRegistration(json.dumps(registration))
    
    print(reg.version)
    print(reg.parsed_json)
    
    background_tasks.add_task(process_registration_async, registration, request_id)
    return {
        "status": "processing",
        "message": "등록 처리가 시작되었습니다. 결과는 웹소켓으로 전송됩니다.",
        "request_id": request_id
    }

# @app.get("/")
# async def root():
#     return {"message": "Hello World"}

def stop_server():
    print("stop_server")
    if s_thread:
        s_thread.stop()
    os.kill(os.getpid(), 2)

def run_server():
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info",
        loop="asyncio"
    )
    
    # 서버 실행
    server = uvicorn.Server(config)
    server.run()

def start_server():        
    server_thread = threading.Thread(target=run_server)
    s_thread = server_thread
    server_thread.daemon = True
    server_thread.start()      
    
def signal_handler(sig, frame):
    print(f"Received signal {sig}, stopping server")
    stop_server()
    exit(0)
    
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    start_server()
    
