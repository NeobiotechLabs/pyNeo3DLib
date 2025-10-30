"""
FastAPI 서버 실행 스크립트
mediapipe 의존성 문제를 우회하여 서버만 실행합니다.
"""
import os
import sys

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if __name__ == "__main__":
    # fastserver의 run_server 함수만 import
    from pyNeo3DLib.fastserver import run_server, signal_handler
    import signal
    
    print("=" * 60)
    print("🚀 FastAPI 서버 시작")
    print("=" * 60)
    print("서버 주소: http://127.0.0.1:8000")
    print("API 문서: http://127.0.0.1:8000/docs")
    print("Health Check: http://127.0.0.1:8000/health")
    print("=" * 60)
    print("\n종료하려면 Ctrl+C를 누르세요.\n")
    
    # 시그널 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 서버 실행
    try:
        run_server()
    except KeyboardInterrupt:
        print("\n서버를 종료합니다...")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()

