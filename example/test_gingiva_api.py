"""
치은 생성 API 테스트 스크립트
"""
import requests
import json
import os
import time

# API 설정
BASE_URL = "http://127.0.0.1:8000"
API_ENDPOINT = f"{BASE_URL}/generate_gingiva"

# 테스트 데이터 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, "data/input")
output_path = os.path.join(script_dir, "f:\work\smile\output")

def test_health_check():
    """서버 상태 확인"""
    print("=" * 60)
    print("[1] 서버 Health Check 테스트")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"상태 코드: {response.status_code}")
        print(f"응답: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        return response.status_code == 200
    except Exception as e:
        print(f"[ERROR] 오류: {str(e)}")
        return False

def test_generate_gingiva():
    """치은 생성 API 테스트"""
    print("\n" + "=" * 60)
    print("[2] 치은 생성 API 테스트")
    print("=" * 60)
    
    # 요청 데이터
    request_data = {
        "input_path": input_path,
        "output_path": output_path,
        "arch_types": ["mandibular"]  # 또는 ["maxilla", "mandibular"]
    }
    
    print(f"\n[요청 데이터]")
    print(json.dumps(request_data, indent=2, ensure_ascii=False))
    
    try:
        # API 호출
        print(f"\n[API 호출 중...]")
        response = requests.post(
            API_ENDPOINT,
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"\n[응답 상태 코드] {response.status_code}")
        print(f"[응답 본문]")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        
        if response.status_code == 200:
            result = response.json()
            if result.get("status") == "processing":
                print("\n[SUCCESS] 치은 생성이 시작되었습니다!")
                print(f"[Request ID] {result.get('request_id')}")
                print(f"[INFO] WebSocket을 통해 진행 상황을 확인할 수 있습니다.")
                return True
        else:
            print(f"\n[FAIL] API 호출 실패")
            return False
            
    except Exception as e:
        print(f"\n[ERROR] 오류 발생: {str(e)}")
        return False

def test_invalid_input():
    """잘못된 입력으로 테스트"""
    print("\n" + "=" * 60)
    print("[3] 잘못된 입력 테스트")
    print("=" * 60)
    
    # 존재하지 않는 경로
    request_data = {
        "input_path": "/invalid/path",
        "output_path": output_path,
        "arch_types": ["mandibular"]
    }
    
    print(f"\n[요청 데이터] (잘못된 경로)")
    print(json.dumps(request_data, indent=2, ensure_ascii=False))
    
    try:
        response = requests.post(
            API_ENDPOINT,
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"\n[응답 상태 코드] {response.status_code}")
        result = response.json()
        print(f"[응답 본문]")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        if result.get("status") == "error":
            print("\n[SUCCESS] 예상대로 에러가 반환되었습니다!")
            return True
        else:
            print("\n[FAIL] 에러가 반환되지 않았습니다.")
            return False
            
    except Exception as e:
        print(f"\n[ERROR] 오류 발생: {str(e)}")
        return False

def test_invalid_arch_type():
    """잘못된 arch_type으로 테스트"""
    print("\n" + "=" * 60)
    print("[4] 잘못된 arch_type 테스트")
    print("=" * 60)
    
    request_data = {
        "input_path": input_path,
        "output_path": output_path,
        "arch_types": ["invalid_type"]
    }
    
    print(f"\n[요청 데이터] (잘못된 arch_type)")
    print(json.dumps(request_data, indent=2, ensure_ascii=False))
    
    try:
        response = requests.post(
            API_ENDPOINT,
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"\n[응답 상태 코드] {response.status_code}")
        result = response.json()
        print(f"[응답 본문]")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        if result.get("status") == "error":
            print("\n[SUCCESS] 예상대로 에러가 반환되었습니다!")
            return True
        else:
            print("\n[FAIL] 에러가 반환되지 않았습니다.")
            return False
            
    except Exception as e:
        print(f"\n[ERROR] 오류 발생: {str(e)}")
        return False

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("치은 생성 API 테스트 시작")
    print("=" * 60)
    
    # 서버가 실행 중인지 확인
    if not test_health_check():
        print("\n[ERROR] 서버가 실행되지 않았습니다. 먼저 서버를 실행해주세요:")
        print("   python -m pyNeo3DLib.fastserver")
        exit(1)
    
    # 테스트 실행
    results = []
    results.append(("정상 입력 테스트", test_generate_gingiva()))
    results.append(("잘못된 경로 테스트", test_invalid_input()))
    results.append(("잘못된 arch_type 테스트", test_invalid_arch_type()))
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    
    for test_name, result in results:
        status = "[성공]" if result else "[실패]"
        print(f"{test_name}: {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\n총 {total}개 테스트 중 {passed}개 성공")
    
    if passed == total:
        print(">>> 모든 테스트가 성공했습니다!")
    else:
        print(f">>> {total - passed}개의 테스트가 실패했습니다.")

