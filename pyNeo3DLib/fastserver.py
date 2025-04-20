from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import threading
import os, signal
import asyncio
import random
import string
import datetime
from .registrationModel import RegistrationModels, RegistrationItem

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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global ws
    ws = websocket
    await websocket.accept()
    while True:
        random_text = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        # 현재 시간
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 데이터 전송
        await websocket.send_json({
            "random_text": random_text,
            "timestamp": current_time
        })
        
        # 1초 대기
        await asyncio.sleep(1)
        data = await websocket.receive_text()
        print(f"Received: {data}")
        

@app.post("/registration/")
async def get_registration(registration: RegistrationItem):
    global ws
    item_dict = registration
    registration_models = RegistrationModels()
    result = registration_models.request_registration(item_dict.origin_type, item_dict.origin_model, item_dict.target_type, item_dict.target_model)
    # 웹소켓을 통해 등록 결과를 클라이언트에게 전송
    try:
        await ws.send_json({
            "random_text": f"Registration: {item_dict.origin_type} to {item_dict.target_type}",
            "timestamp": "abvsd"
        })
    except Exception as e:
        print(f"웹소켓 메시지 전송 중 오류 발생: {str(e)}")
    return result

@app.get("/")
async def root():
    return {"message": "Hello World"}

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
    
