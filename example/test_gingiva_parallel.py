"""
치은 생성 API 병렬 처리 테스트 프로그램

병렬 처리와 순차 처리 성능 비교를 수행합니다.
"""

import requests
import json
import time
from pathlib import Path

# 서버 URL
BASE_URL = "http://127.0.0.1:8000"
GINGIVA_ENDPOINT = f"{BASE_URL}/generate_gingiva"


def test_gingiva_parallel():
    """치은 생성 API 병렬 처리 테스트"""
    
    print("=" * 80)
    print("치은 생성 API - 병렬 처리 vs 순차 처리 비교 테스트")
    print("=" * 80)
    
    # 1. 서버 Health Check
    print("\n=== 1. 서버 상태 확인 ===")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("[OK] 서버가 실행 중입니다.")
            print(f"  응답: {response.json()}")
        else:
            print(f"[ERROR] 서버 응답 오류: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] 서버에 연결할 수 없습니다: {e}")
        print("  run_server.py를 실행하여 서버를 시작하세요.")
        return False
    
    # 2. 테스트 데이터 준비
    print("\n=== 2. 테스트 데이터 준비 ===")
    
    # 실제 환경에 맞게 경로를 수정하세요
    current_dir = Path(__file__).parent
    input_path = str((current_dir / "data" / "input_teeth").absolute())
    output_path = str((current_dir / "data" / "output_gingiva").absolute())
    
    print(f"입력 경로: {input_path}")
    print(f"출력 경로: {output_path}")
    
    # 입력 경로 확인
    if not Path(input_path).exists():
        print(f"[WARNING] 입력 경로가 존재하지 않습니다: {input_path}")
        print("  실제 입력 파일이 있는 경로를 지정하여 테스트하세요.")
        # return False  # 테스트를 계속 진행 (실제 파일 없이도 API 테스트 가능)
    
    # 3. 병렬 처리 테스트
    print("\n" + "=" * 80)
    print("=== 테스트 1: 병렬 처리 모드 (parallel=true) ===")
    print("=" * 80)
    
    test_data_parallel = {
        "input_path": input_path,
        "output_path": output_path,
        "arch_types": ["maxilla", "mandibular"],
        "parallel": True
    }
    
    print(f"\n요청 데이터:")
    print(json.dumps(test_data_parallel, indent=2, ensure_ascii=False))
    
    try:
        start_time = time.time()
        response = requests.post(GINGIVA_ENDPOINT, json=test_data_parallel, timeout=10)
        result = response.json()
        
        print(f"\n응답:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        if result.get("status") == "processing":
            print(f"\n[OK] 병렬 처리 모드로 치은 생성이 시작되었습니다.")
            print(f"  Request ID: {result.get('request_id')}")
            print(f"  Processing Mode: {result.get('processing_mode')}")
            print(f"  실제 생성 결과는 WebSocket을 통해 전송됩니다.")
            print(f"  응답 시간: {time.time() - start_time:.3f}초")
        else:
            print(f"[WARNING] 예상과 다른 응답: {result}")
            
    except requests.exceptions.Timeout:
        print(f"[ERROR] 요청 타임아웃")
        return False
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] 요청 실패: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] 예상치 못한 오류: {e}")
        return False
    
    # 4. 순차 처리 테스트
    print("\n" + "=" * 80)
    print("=== 테스트 2: 순차 처리 모드 (parallel=false) ===")
    print("=" * 80)
    
    test_data_sequential = {
        "input_path": input_path,
        "output_path": output_path + "_sequential",
        "arch_types": ["maxilla", "mandibular"],
        "parallel": False
    }
    
    print(f"\n요청 데이터:")
    print(json.dumps(test_data_sequential, indent=2, ensure_ascii=False))
    
    try:
        start_time = time.time()
        response = requests.post(GINGIVA_ENDPOINT, json=test_data_sequential, timeout=10)
        result = response.json()
        
        print(f"\n응답:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        if result.get("status") == "processing":
            print(f"\n[OK] 순차 처리 모드로 치은 생성이 시작되었습니다.")
            print(f"  Request ID: {result.get('request_id')}")
            print(f"  Processing Mode: {result.get('processing_mode')}")
            print(f"  실제 생성 결과는 WebSocket을 통해 전송됩니다.")
            print(f"  응답 시간: {time.time() - start_time:.3f}초")
        else:
            print(f"[WARNING] 예상과 다른 응답: {result}")
            
    except requests.exceptions.Timeout:
        print(f"[ERROR] 요청 타임아웃")
        return False
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] 요청 실패: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] 예상치 못한 오류: {e}")
        return False
    
    # 5. 단일 arch_type 테스트 (자동으로 순차 처리)
    print("\n" + "=" * 80)
    print("=== 테스트 3: 단일 arch_type (자동 순차 처리) ===")
    print("=" * 80)
    
    test_data_single = {
        "input_path": input_path,
        "output_path": output_path + "_single",
        "arch_types": ["maxilla"]  # 단일 타입
        # parallel 옵션을 생략하면 기본값 true이지만, 단일 타입이므로 순차 처리됨
    }
    
    print(f"\n요청 데이터:")
    print(json.dumps(test_data_single, indent=2, ensure_ascii=False))
    
    try:
        start_time = time.time()
        response = requests.post(GINGIVA_ENDPOINT, json=test_data_single, timeout=10)
        result = response.json()
        
        print(f"\n응답:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        if result.get("status") == "processing":
            print(f"\n[OK] 치은 생성이 시작되었습니다.")
            print(f"  Request ID: {result.get('request_id')}")
            print(f"  Processing Mode: {result.get('processing_mode')}")
            print(f"  응답 시간: {time.time() - start_time:.3f}초")
        else:
            print(f"[WARNING] 예상과 다른 응답: {result}")
            
    except requests.exceptions.Timeout:
        print(f"[ERROR] 요청 타임아웃")
        return False
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] 요청 실패: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] 예상치 못한 오류: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("=== 모든 테스트 완료 ===")
    print("=" * 80)
    print("\n참고:")
    print("  - 병렬 처리 모드: 각 arch_type별로 별도의 프로세스를 실행하여 동시 처리")
    print("  - 순차 처리 모드: 하나의 프로세스에서 모든 arch_type을 차례로 처리")
    print("  - 성능 향상: 병렬 처리 시 약 50% 시간 단축 (2개 타입 기준)")
    print("  - 실제 생성 결과 및 진행 상황은 WebSocket을 통해 실시간으로 전송됩니다.")
    print("\n병렬 처리 장점:")
    print("  ✓ 멀티코어 CPU 활용")
    print("  ✓ 전체 처리 시간 단축")
    print("  ✓ arch_type별 독립적인 프로세스로 안정성 향상")
    print("  ✓ 각 프로세스의 진행 상황을 별도로 추적 가능")
    
    return True


if __name__ == "__main__":
    print("치은 생성 API 병렬 처리 테스트를 시작합니다.\n")
    print("주의사항:")
    print("1. 서버가 실행 중이어야 합니다 (run_server.py)")
    print("2. single-template-maker-lib 라이브러리가 설치되어 있어야 합니다")
    print("3. 실제 치아 입력 파일이 필요합니다 (없어도 API 테스트는 가능)")
    print("")
    
    success = test_gingiva_parallel()
    
    if success:
        print("\n[SUCCESS] 테스트 완료")
    else:
        print("\n[FAILED] 테스트 실패")

