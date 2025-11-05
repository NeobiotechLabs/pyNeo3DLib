"""
템플릿 편집 API 테스트 프로그램

FastAPI의 템플릿 편집 엔드포인트들을 테스트합니다:
- /template_editor/start: 편집 세션 시작
- /template_editor/transform: 변환 적용 (여러 번 가능)
- /template_editor/stop: 편집 세션 종료
"""

import requests
import json
import sys
from pathlib import Path

# 서버 URL
BASE_URL = "http://127.0.0.1:8000"
START_ENDPOINT = f"{BASE_URL}/template_editor/start"
TRANSFORM_ENDPOINT = f"{BASE_URL}/template_editor/transform"
STOP_ENDPOINT = f"{BASE_URL}/template_editor/stop"

def test_template_editor_api():
    """템플릿 편집 API 테스트 - 세 개의 엔드포인트를 순차적으로 호출"""
    
    print("=" * 60)
    print("템플릿 편집 API 테스트")
    print("=" * 60)
    
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
    blend_template_path = str((current_dir / "data" / "templates").absolute())
    stl_export_path = str((current_dir / "data" / "exports").absolute())
    
    # 사용 가능한 템플릿 파일 찾기
    templates_dir = current_dir / "data" / "templates"
    available_templates = list(templates_dir.glob("*.blend"))
    
    if not available_templates:
        print(f"[ERROR] .blend 템플릿 파일이 없습니다: {templates_dir}")
        print("  실제 .blend 파일을 준비하여 다시 테스트하세요.")
        return False
    
    # 첫 번째 사용 가능한 템플릿 사용
    template_file = available_templates[0].name
    print(f"사용할 템플릿 파일: {template_file}")
    print(f"사용 가능한 템플릿 목록: {[t.name for t in available_templates]}")
    print(f"템플릿 경로: {blend_template_path}")
    print(f"내보내기 경로: {stl_export_path}")
    
    # 3. 테스트 케이스 1: transform 포함 (여러 번 호출)
    print("\n" + "=" * 60)
    print("=== 테스트 1: transform 여러 번 호출 ===")
    print("=" * 60)
    
    try:
        # 3-1. 편집 시작
        print("\n[STEP 1] 편집 시작 (start_editing)")
        start_data = {
            "blend_template_path": blend_template_path,
            "stl_export_path": stl_export_path,
            "blend_template": template_file,
            "arch_degree": 15.5,
            "y_scale": 1.2
        }
        print(f"요청 데이터: {json.dumps(start_data, indent=2, ensure_ascii=False)}")
        
        response = requests.post(START_ENDPOINT, json=start_data, timeout=120)
        result = response.json()
        print(f"응답: {json.dumps(result, indent=2, ensure_ascii=False)}")
        
        if result.get("status") != "success":
            print(f"[ERROR] 편집 시작 실패: {result.get('message')}")
            return False
        print("[OK] 편집 시작 성공")
        
        # 3-2. 변환 적용 #1
        print("\n[STEP 2] 변환 적용 #1")
        transform_data_1 = {
            "arch_degree": 20.0,
            "y_scale": 0.8
        }
        print(f"요청 데이터: {json.dumps(transform_data_1, indent=2, ensure_ascii=False)}")
        
        response = requests.post(TRANSFORM_ENDPOINT, json=transform_data_1, timeout=120)
        result = response.json()
        print(f"응답: {json.dumps(result, indent=2, ensure_ascii=False)}")
        
        if result.get("status") != "success":
            print(f"[ERROR] 변환 적용 #1 실패: {result.get('message')}")
            return False
        print("[OK] 변환 적용 #1 성공")
        
        # 3-3. 변환 적용 #2
        print("\n[STEP 3] 변환 적용 #2")
        transform_data_2 = {
            "arch_degree": 25.0,
            "y_scale": 0.9
        }
        print(f"요청 데이터: {json.dumps(transform_data_2, indent=2, ensure_ascii=False)}")
        
        response = requests.post(TRANSFORM_ENDPOINT, json=transform_data_2, timeout=120)
        result = response.json()
        print(f"응답: {json.dumps(result, indent=2, ensure_ascii=False)}")
        
        if result.get("status") != "success":
            print(f"[ERROR] 변환 적용 #2 실패: {result.get('message')}")
            return False
        print("[OK] 변환 적용 #2 성공")
        
        # 3-4. 편집 종료
        print("\n[STEP 4] 편집 종료 (stop_editing)")
        stop_data = {
            "arch_degree": 30.0,
            "y_scale": 1.0
        }
        print(f"요청 데이터: {json.dumps(stop_data, indent=2, ensure_ascii=False)}")
        
        response = requests.post(STOP_ENDPOINT, json=stop_data, timeout=120)
        result = response.json()
        print(f"응답: {json.dumps(result, indent=2, ensure_ascii=False)}")
        
        if result.get("status") != "success":
            print(f"[ERROR] 편집 종료 실패: {result.get('message')}")
            return False
        print("[OK] 편집 종료 성공")
        
        # STL 폴더 경로 확인
        stl_folder = result.get("stl_folder_path")
        if stl_folder:
            print(f"\nSTL 파일 경로: {stl_folder}")
            
            # 실제 파일 존재 확인
            stl_path = Path(stl_folder)
            if stl_path.exists():
                stl_files = list(stl_path.glob("*.stl"))
                print(f"생성된 STL 파일 수: {len(stl_files)}")
                for stl_file in stl_files[:5]:
                    print(f"  - {stl_file.name}")
            else:
                print(f"[WARNING] STL 폴더가 존재하지 않습니다.")
        
        print("\n[SUCCESS] 테스트 1 완료")
        
    except requests.exceptions.Timeout:
        print(f"[ERROR] 요청 타임아웃")
        return False
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] 요청 실패: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. 테스트 케이스 2: transform 생략
    print("\n" + "=" * 60)
    print("=== 테스트 2: transform 생략 (start -> stop) ===")
    print("=" * 60)
    
    try:
        # 4-1. 편집 시작
        print("\n[STEP 1] 편집 시작 (start_editing)")
        start_data = {
            "blend_template_path": blend_template_path,
            "stl_export_path": stl_export_path,
            "blend_template": template_file,
            "arch_degree": 18.0,
            "y_scale": 1.1
        }
        print(f"요청 데이터: {json.dumps(start_data, indent=2, ensure_ascii=False)}")
        
        response = requests.post(START_ENDPOINT, json=start_data, timeout=120)
        result = response.json()
        print(f"응답: {json.dumps(result, indent=2, ensure_ascii=False)}")
        
        if result.get("status") != "success":
            print(f"[ERROR] 편집 시작 실패: {result.get('message')}")
            return False
        print("[OK] 편집 시작 성공")
        
        # 4-2. 편집 종료 (transform 생략)
        print("\n[STEP 2] 편집 종료 (stop_editing, transform 생략)")
        stop_data = {
            "arch_degree": 25.0,
            "y_scale": 1.0
        }
        print(f"요청 데이터: {json.dumps(stop_data, indent=2, ensure_ascii=False)}")
        
        response = requests.post(STOP_ENDPOINT, json=stop_data, timeout=120)
        result = response.json()
        print(f"응답: {json.dumps(result, indent=2, ensure_ascii=False)}")
        
        if result.get("status") != "success":
            print(f"[ERROR] 편집 종료 실패: {result.get('message')}")
            return False
        print("[OK] 편집 종료 성공")
        
        print("\n[SUCCESS] 테스트 2 완료")
        
    except Exception as e:
        print(f"[ERROR] 테스트 2 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n=== 모든 테스트 완료 ===")
    return True


def test_api_validation():
    """API 파라미터 검증 및 세션 상태 테스트"""
    
    print("\n" + "=" * 60)
    print("API 파라미터 검증 및 세션 상태 테스트")
    print("=" * 60)
    
    # 테스트 1: start 전에 transform 호출
    print("\n=== 테스트 1: start 전에 transform 호출 (실패 예상) ===")
    try:
        response = requests.post(TRANSFORM_ENDPOINT, json={"arch_degree": 20.0, "y_scale": 0.8}, timeout=5)
        result = response.json()
        
        if result.get("status") == "error" and "시작되지 않았습니다" in result.get("message", ""):
            print(f"[OK] 예상대로 오류 반환됨")
            print(f"  메시지: {result.get('message')}")
        else:
            print(f"[WARNING] 예상과 다른 응답")
            print(f"  응답: {result}")
    except Exception as e:
        print(f"[ERROR] 테스트 실패: {e}")
    
    # 테스트 2: start 전에 stop 호출
    print("\n=== 테스트 2: start 전에 stop 호출 (실패 예상) ===")
    try:
        response = requests.post(STOP_ENDPOINT, json={"arch_degree": 30.0, "y_scale": 1.0}, timeout=5)
        result = response.json()
        
        if result.get("status") == "error" and "시작되지 않았습니다" in result.get("message", ""):
            print(f"[OK] 예상대로 오류 반환됨")
            print(f"  메시지: {result.get('message')}")
        else:
            print(f"[WARNING] 예상과 다른 응답")
            print(f"  응답: {result}")
    except Exception as e:
        print(f"[ERROR] 테스트 실패: {e}")
    
    # 테스트 3: start 필수 파라미터 누락
    print("\n=== 테스트 3: start 필수 파라미터 누락 (실패 예상) ===")
    test_cases = [
        ("blend_template_path 누락", {
            "stl_export_path": "/path/to/exports",
            "blend_template": "template.blend",
            "arch_degree": 15.5,
            "y_scale": 1.2
        }),
        ("arch_degree 누락", {
            "blend_template_path": "/path/to/templates",
            "stl_export_path": "/path/to/exports",
            "blend_template": "template.blend",
            "y_scale": 1.2
        }),
    ]
    
    for test_name, test_data in test_cases:
        print(f"\n  {test_name}:")
        
        try:
            response = requests.post(START_ENDPOINT, json=test_data, timeout=5)
            result = response.json()
            
            if result.get("status") == "error" and "누락" in result.get("message", ""):
                print(f"  [OK] 예상대로 오류 반환됨")
                print(f"    메시지: {result.get('message')}")
            else:
                print(f"  [WARNING] 예상과 다른 응답")
                print(f"    응답: {result}")
        except Exception as e:
            print(f"  [ERROR] 테스트 실패: {e}")
    
    # 테스트 4: transform 필수 파라미터 누락
    print("\n=== 테스트 4: transform 필수 파라미터 누락 (실패 예상) ===")
    
    # 먼저 start를 호출해서 세션 시작
    current_dir = Path(__file__).parent
    blend_template_path = str((current_dir / "data" / "templates").absolute())
    stl_export_path = str((current_dir / "data" / "exports").absolute())
    templates_dir = current_dir / "data" / "templates"
    available_templates = list(templates_dir.glob("*.blend"))
    
    if available_templates:
        template_file = available_templates[0].name
        start_data = {
            "blend_template_path": blend_template_path,
            "stl_export_path": stl_export_path,
            "blend_template": template_file,
            "arch_degree": 15.5,
            "y_scale": 1.2
        }
        
        try:
            # 세션 시작
            response = requests.post(START_ENDPOINT, json=start_data, timeout=120)
            result = response.json()
            
            if result.get("status") == "success":
                print("  세션 시작됨. transform 파라미터 검증 테스트 진행...")
                
                # arch_degree 누락
                response = requests.post(TRANSFORM_ENDPOINT, json={"y_scale": 0.8}, timeout=5)
                result = response.json()
                
                if result.get("status") == "error" and "누락" in result.get("message", ""):
                    print(f"  [OK] arch_degree 누락 시 오류 반환됨")
                    print(f"    메시지: {result.get('message')}")
                else:
                    print(f"  [WARNING] 예상과 다른 응답: {result}")
                
                # 세션 종료 (정리)
                stop_data = {"arch_degree": 30.0, "y_scale": 1.0}
                requests.post(STOP_ENDPOINT, json=stop_data, timeout=120)
                print("  세션 종료됨.")
            else:
                print(f"  [SKIP] 세션 시작 실패: {result.get('message')}")
        except Exception as e:
            print(f"  [ERROR] 테스트 실패: {e}")
    else:
        print("  [SKIP] .blend 파일이 없어 테스트를 건너뜁니다.")
    
    print("\n=== 검증 테스트 완료 ===")



if __name__ == "__main__":
    print("템플릿 편집 API 테스트를 시작합니다.\n")
    print("주의사항:")
    print("1. 서버가 실행 중이어야 합니다 (run_server.py)")
    print("2. Blender가 설치되어 있어야 합니다")
    print("3. 올바른 .blend 템플릿 파일이 필요합니다")
    print("")
    
    # 기본 API 테스트
    success = test_template_editor_api()
    
    # 검증 테스트
    test_api_validation()
    
    if success:
        print("\n[SUCCESS] 테스트 완료")
        sys.exit(0)
    else:
        print("\n[FAILED] 테스트 실패")
        sys.exit(1)

