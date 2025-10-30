"""
pyNeo3DLib를 모듈로 실행할 때의 진입점
python -m pyNeo3DLib 로 fastserver 실행
"""

if __name__ == "__main__":
    # fastserver만 직접 import (다른 모듈의 의존성 문제 우회)
    from pyNeo3DLib.fastserver import run_server, signal_handler
    import signal
    
    # 시그널 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 서버 직접 실행 (스레드가 아닌 메인 프로세스에서)
    print("Starting FastAPI server...")
    run_server()

