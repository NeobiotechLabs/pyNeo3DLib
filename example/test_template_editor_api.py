"""
템플릿 편집 API 테스트 프로그램

FastAPI의 /template_editor/process endpoint를 테스트합니다.
"""

import requests
import json
import sys
from pathlib import Path

# 서버 URL
BASE_URL = "http://127.0.0.1:8000"
API_ENDPOINT = f"{BASE_URL}/template_editor/process"

def test_template_editor_api():
    """템플릿 편집 API 테스트"""
    
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
    
    # 테스트 케이스 1: transform_params 포함
    test_case_1 = {
        "blend_template_path": blend_template_path,
        "stl_export_path": stl_export_path,
        "blend_template": template_file,
        "start_params": {
            "arch_degree": 15.5,
            "y_scale": 1.2
        },
        "transform_params": {
            "arch_degree": 20.0,
            "y_scale": 0.8
        },
        "stop_params": {
            "arch_degree": 30.0,
            "y_scale": 1.0
        }
    }
    
    # 테스트 케이스 2: transform_params 생략
    test_case_2 = {
        "blend_template_path": blend_template_path,
        "stl_export_path": stl_export_path,
        "blend_template": template_file,
        "start_params": {
            "arch_degree": 15.5,
            "y_scale": 1.2
        },
        "stop_params": {
            "arch_degree": 25.0,
            "y_scale": 1.0
        }
    }
    
    print(f"템플릿 경로: {blend_template_path}")
    print(f"내보내기 경로: {stl_export_path}")
    
    # 3. API 테스트 실행
    test_cases = [
        ("테스트 1: transform_params 포함", test_case_1),
        ("테스트 2: transform_params 생략", test_case_2)
    ]
    
    for test_name, test_data in test_cases:
        print(f"\n=== 3. {test_name} ===")
        print(f"요청 데이터:")
        print(json.dumps(test_data, indent=2, ensure_ascii=False))
        
        try:
            response = requests.post(
                API_ENDPOINT,
                json=test_data,
                timeout=300  # 5분 타임아웃 (Blender 처리 시간 고려)
            )
            
            print(f"\n응답 상태 코드: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"\n응답 데이터:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
                
                if result.get("status") == "success":
                    print(f"\n[OK] {test_name} 성공")
                    
                    # STL 폴더 경로 확인
                    stl_folder = result.get("stl_folder_path")
                    if stl_folder:
                        print(f"  STL 파일 경로: {stl_folder}")
                        
                        # 실제 파일 존재 확인
                        stl_path = Path(stl_folder)
                        if stl_path.exists():
                            stl_files = list(stl_path.glob("*.stl"))
                            print(f"  생성된 STL 파일 수: {len(stl_files)}")
                            for stl_file in stl_files[:5]:
                                print(f"    - {stl_file.name}")
                        else:
                            print(f"  [WARNING] STL 폴더가 존재하지 않습니다.")
                    
                elif result.get("status") == "error":
                    print(f"\n[ERROR] {test_name} 실패")
                    print(f"  메시지: {result.get('message')}")
                    
                    # Blender 관련 오류인 경우
                    if "Blender" in result.get('message', ''):
                        print("\n  참고: Blender가 설치되어 있고 경로가 올바르게 설정되어 있는지 확인하세요.")
                        print("  또는 'teeth-editor --check-blender' 명령을 실행하세요.")
                    
                    # 템플릿 파일 관련 오류인 경우
                    if "template" in result.get('message', '').lower():
                        print("\n  참고: .blend 템플릿 파일이 올바른 경로에 있는지 확인하세요.")
                        print(f"  예상 경로: {blend_template_path}/template.blend")
                
            else:
                print(f"[ERROR] HTTP 오류 발생: {response.status_code}")
                print(f"응답: {response.text}")
                
        except requests.exceptions.Timeout:
            print(f"[ERROR] 요청 타임아웃 (5분 초과)")
            print("  Blender 처리에 시간이 오래 걸릴 수 있습니다.")
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] 요청 실패: {e}")
        except Exception as e:
            print(f"[ERROR] 예상치 못한 오류: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n=== 모든 테스트 완료 ===")
    return True


def test_api_validation():
    """API 파라미터 검증 테스트"""
    
    print("\n" + "=" * 60)
    print("API 파라미터 검증 테스트")
    print("=" * 60)
    
    # 필수 파라미터 누락 테스트
    test_cases = [
        ("blend_template_path 누락", {
            "stl_export_path": "/path/to/exports",
            "blend_template": "template.blend",
            "start_params": {"arch_degree": 15.5, "y_scale": 1.2},
            "stop_params": {"arch_degree": 30.0, "y_scale": 1.0}
        }),
        ("start_params.arch_degree 누락", {
            "blend_template_path": "/path/to/templates",
            "stl_export_path": "/path/to/exports",
            "blend_template": "template.blend",
            "start_params": {"y_scale": 1.2},
            "stop_params": {"arch_degree": 30.0, "y_scale": 1.0}
        }),
    ]
    
    for test_name, test_data in test_cases:
        print(f"\n=== {test_name} ===")
        
        try:
            response = requests.post(API_ENDPOINT, json=test_data, timeout=5)
            result = response.json()
            
            if result.get("status") == "error":
                print(f"[OK] 예상대로 오류 반환됨")
                print(f"  메시지: {result.get('message')}")
            else:
                print(f"[WARNING] 오류가 반환되지 않음")
                
        except Exception as e:
            print(f"[ERROR] 테스트 실패: {e}")


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

