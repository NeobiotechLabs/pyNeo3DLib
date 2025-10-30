import os
import logging
import warnings

# TensorFlow 경고 메시지 숨기기 (다른 import 전에 설정)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # GPU 사용 비활성화

# TensorFlow 관련 경고 메시지 필터링
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=FutureWarning, module='tensorflow')

# TensorFlow 로깅 레벨 설정
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import threading
import signal
import asyncio
import random
import string
import datetime
from typing import Dict, Any
import json

# Lazy import: 실제 사용 시점에만 import (mediapipe 의존성 회피)
# from .registration import Neo3DRegistration  # registration API 사용 시에만 import
# from .teethTemplateFinder.teethTemplateFinder import TeethTemplateFinder  # 사용 시에만 import
# from .threePointRegistration.threePointRegistration import ThreePointRegistration  # 사용 시에만 import
# from .gingivaGenerator.gingivaGenerator import GingivaGenerator  # 사용 시에만 import

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
teeth_template_finder = None

async def process_registration_async(registration_data, request_id):
    global ws
    try:
        # Lazy import: registration 기능 사용 시에만 import
        from .registration import Neo3DRegistration
        
        reg = Neo3DRegistration(json.dumps(registration_data), ws)
        print(f"[{request_id}] Registration started")
        result = await reg.run_registration(visualize=False)
        print(f"[{request_id}] Registration completed")
        
        if ws:
            await ws.send_json({
                "type": "registration_completed",
                "request_id": request_id,
                "result": result,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

        return result
    except Exception as e:
        print(f"[{request_id}] Error during registration: {str(e)}")
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
    try:
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
    except WebSocketDisconnect:
        print("WebSocket disconnected")
        ws = None
        

@app.post("/registration")
async def get_registration(background_tasks: BackgroundTasks, registration: Dict[str, Any] = Body(...)):
    global ws
    request_id = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    print(f"[{request_id}] Registration API called")
    
    # Lazy import: registration 기능 사용 시에만 import
    from .registration import Neo3DRegistration
    
    reg = Neo3DRegistration(json.dumps(registration), ws)
    
    print(reg.version)
    print(reg.parsed_json)
    
    background_tasks.add_task(process_registration_async, registration, request_id)
    return {
        "status": "processing",
        "message": "Registration process has started. Results will be sent via WebSocket.",
        "request_id": request_id
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint for server status monitoring
    Returns 200 status code when server is running properly
    """
    return {
        "status": "healthy",
        "message": "Server is running",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

@app.post("/template_finder/start")
async def start_template_finder(db_path: str = Body(..., embed=True)):
    """
    템플릿 파인더 시작 API
    """
    global teeth_template_finder
    
    try:
        # Lazy import: template finder 기능 사용 시에만 import
        from .teethTemplateFinder.teethTemplateFinder import TeethTemplateFinder
        
        teeth_template_finder = TeethTemplateFinder()
        teeth_template_finder.start_template_finder(db_path)
        
        return {
            "status": "success",
            "message": "Template finder started successfully",
            "db_path": db_path,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to start template finder: {str(e)}",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

@app.post("/template_finder/search")
async def find_template(search_request: Dict[str, Any] = Body(...)):
    """
    템플릿 검색 API
    """
    global teeth_template_finder
    
    if teeth_template_finder is None:
        return {
            "status": "error",
            "message": "Template finder not initialized. Please call /template_finder/start first.",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    try:
        # 필수 파라미터 추출
        arch_depth = search_request.get("arch_depth")
        molar_width = search_request.get("molar_width")
        landmarks = search_request.get("landmarks")
        
        if arch_depth is None or molar_width is None or landmarks is None:
            return {
                "status": "error",
                "message": "Missing required parameters: arch_depth, molar_width, landmarks",
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        # 선택적 파라미터 추출
        arch_type = search_request.get("arch_type")
        teeth_shape_type = search_request.get("teeth_shape_type")
        teeth_height_type = search_request.get("teeth_height_type")
        teeth_size_type = search_request.get("teeth_size_type")
        removed_teeth_index = search_request.get("removed_teeth_index")
        top_k = search_request.get("top_k", 5)
        
        # 템플릿 검색 실행
        results = teeth_template_finder.find_template(
            arch_depth=arch_depth,
            molar_width=molar_width,
            landmarks=landmarks,
            arch_type=arch_type,
            teeth_shape_type=teeth_shape_type,
            teeth_height_type=teeth_height_type,
            teeth_size_type=teeth_size_type,
            removed_teeth_index=removed_teeth_index,
            top_k=top_k
        )
        
        return {
            "status": "success",
            "message": f"Found {len(results)} templates",
            "results": results,
            "search_params": {
                "arch_depth": arch_depth,
                "molar_width": molar_width,
                "landmarks_count": len(landmarks),
                "arch_type": arch_type,
                "teeth_shape_type": teeth_shape_type,
                "teeth_height_type": teeth_height_type,
                "teeth_size_type": teeth_size_type,
                "removed_teeth_index": removed_teeth_index,
                "top_k": top_k
            },
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Search failed: {str(e)}",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

# @app.get("/")
# async def root():
#     return {"message": "Hello World"}

@app.post("/threepoint_registration")
async def threepoint_registration(request: Dict[str, Any] = Body(...)):
    global ws
    request_id = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    print(f"[{request_id}] Three-point registration API called")
    
    try:
        # 입력 데이터 검증
        required_fields = ["target_mesh", "source_mesh", "target_points", "source_points"]
        for field in required_fields:
            if field not in request:
                raise ValueError(f"필수 필드가 누락되었습니다: {field}")
        
        target_mesh_path = request["target_mesh"]["path"]
        source_mesh_path = request["source_mesh"]["path"]
        target_points = request["target_points"]
        source_points = request["source_points"]
        
        # 모든 정확도 관련 매개변수는 threePointRegistration.py의 상수 사용
        
        print(f"[{request_id}] 타겟 메시: {target_mesh_path}")
        print(f"[{request_id}] 소스 메시: {source_mesh_path}")
        print(f"[{request_id}] 타겟 점 개수: {len(target_points)}")
        print(f"[{request_id}] 소스 점 개수: {len(source_points)}")
        
        # Lazy import: threepoint registration 기능 사용 시에만 import
        from .threePointRegistration.threePointRegistration import ThreePointRegistration
        
        # 3점 정합 실행 (모든 매개변수는 기본값/상수 사용)
        three_point_reg = ThreePointRegistration(
            target_mesh_path=target_mesh_path,
            source_mesh_path=source_mesh_path,
            target_points=target_points,
            source_points=source_points
            # visualization=False (기본값)
            # 나머지 모든 매개변수는 threePointRegistration.py의 상수 사용
        )
        
        transformation_matrix = await three_point_reg.run_registration()
        
        print(f"[{request_id}] 3점 정합 완료")
        
        return {
            "status": "success",
            "transformation_matrix": transformation_matrix.tolist(),
            "request_id": request_id,
            "message": "3점 정합이 성공적으로 완료되었습니다.",
            "parameters": {
                "target_points_count": len(target_points),
                "source_points_count": len(source_points)
            },
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        print(f"[{request_id}] 오류 발생: {str(e)}")
        return {
            "status": "error",
            "message": f"3점 정합 중 오류가 발생했습니다: {str(e)}",
            "request_id": request_id,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

@app.post("/generate_gingiva")
async def generate_gingiva(background_tasks: BackgroundTasks, request: Dict[str, Any] = Body(...)):
    """
    치은(gingiva) 생성 API
    
    요청 본문 예시:
    {
        "input_path": "/path/to/input/teeth/files",
        "output_path": "/path/to/output",
        "arch_types": ["maxillary", "mandibular"]  # "maxillary" 또는 "mandibular" 또는 둘 다
    }
    """
    global ws
    request_id = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    print(f"[{request_id}] 치은 생성 API 호출됨")
    
    try:
        # 필수 파라미터 검증
        if "input_path" not in request:
            return {
                "status": "error",
                "message": "필수 파라미터가 누락되었습니다: input_path",
                "request_id": request_id,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        if "output_path" not in request:
            return {
                "status": "error",
                "message": "필수 파라미터가 누락되었습니다: output_path",
                "request_id": request_id,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        input_path = request["input_path"]
        output_path = request["output_path"]
        arch_types = request.get("arch_types", ["mandibular"])  # 기본값: mandibular
        
        # Lazy import: gingiva generation 기능 사용 시에만 import
        from .gingivaGenerator.gingivaGenerator import GingivaGenerator
        
        # GingivaGenerator 인스턴스 생성
        gingiva_generator = GingivaGenerator(websocket=ws)
        
        # arch_types 유효성 검증
        is_valid, error_message = gingiva_generator.validate_arch_types(arch_types)
        if not is_valid:
            return {
                "status": "error",
                "message": error_message,
                "request_id": request_id,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        # 입력 경로 존재 확인
        is_valid, error_message = GingivaGenerator.validate_input_path(input_path)
        if not is_valid:
            return {
                "status": "error",
                "message": error_message,
                "request_id": request_id,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        print(f"[{request_id}] 입력 경로: {input_path}")
        print(f"[{request_id}] 출력 경로: {output_path}")
        print(f"[{request_id}] 생성할 타입: {arch_types}")
        
        # 백그라운드에서 치은 생성 실행
        background_tasks.add_task(
            gingiva_generator.generate_gingiva,
            input_path,
            output_path,
            arch_types,
            request_id
        )
        
        return {
            "status": "processing",
            "message": "치은 생성이 시작되었습니다. 결과는 WebSocket을 통해 전송됩니다.",
            "request_id": request_id,
            "input_path": input_path,
            "output_path": output_path,
            "arch_types": arch_types,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        print(f"[{request_id}] 오류 발생: {str(e)}")
        return {
            "status": "error",
            "message": f"치은 생성 요청 처리 중 오류가 발생했습니다: {str(e)}",
            "request_id": request_id,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

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
    print("start_server")
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
    
