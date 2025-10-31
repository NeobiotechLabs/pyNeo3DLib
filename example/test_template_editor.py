"""
teeth-template-editor 모듈 직접 테스트 프로그램

이 프로그램은 teeth-template-editor 모듈의 기본 동작을 테스트합니다.
create_editing_session을 사용하여 세션을 생성하고,
start_editing -> transform -> stop_editing 워크플로우를 실행합니다.
"""

import os
import sys
from pathlib import Path

def test_template_editor():
    """템플릿 편집 모듈 테스트"""
    
    try:
        from teeth_template_editor import create_editing_session
        print("[OK] teeth-template-editor 모듈 import 성공")
    except ImportError as e:
        print(f"[ERROR] teeth-template-editor 모듈 import 실패: {e}")
        print("  pip install teeth-template-editor를 실행하세요.")
        return False
    
    # 테스트용 경로 설정
    # 실제 환경에 맞게 수정이 필요합니다
    current_dir = Path(__file__).parent
    blend_template_path = current_dir / "data" / "templates"
    stl_export_path = current_dir / "data" / "exports"
    
    # 디렉토리 생성 (없는 경우)
    blend_template_path.mkdir(parents=True, exist_ok=True)
    stl_export_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n=== 테스트 경로 설정 ===")
    print(f"템플릿 경로: {blend_template_path}")
    print(f"내보내기 경로: {stl_export_path}")
    
    try:
        # 1. 세션 생성
        print(f"\n=== 1. 편집 세션 생성 ===")
        handler = create_editing_session(
            blend_template_path=str(blend_template_path),
            stl_export_path=str(stl_export_path)
        )
        print("[OK] 편집 세션 생성 성공")
        print(f"  Handler 타입: {type(handler).__name__}")
        
    except FileNotFoundError as e:
        print(f"[ERROR] Blender를 찾을 수 없습니다: {e}")
        print("  Blender를 설치하거나 'teeth-editor --check-blender' 명령을 실행하세요.")
        return False
    except Exception as e:
        print(f"[ERROR] 세션 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 2. 편집 시작 테스트
    print(f"\n=== 2. 편집 시작 테스트 ===")
    print("참고: 실제 .blend 템플릿 파일이 필요합니다.")
    print("테스트용 파일명: 'template.blend'")
    
    # 실제 템플릿 파일이 있는지 확인
    # 사용 가능한 템플릿 파일 찾기
    available_templates = list(blend_template_path.glob("*.blend"))
    
    if not available_templates:
        print(f"[WARNING] .blend 템플릿 파일이 없습니다: {blend_template_path}")
        print("  실제 .blend 파일을 준비하여 다시 테스트하세요.")
        print("\n=== Handler 메서드 확인 ===")
        print(f"사용 가능한 메서드: {[m for m in dir(handler) if not m.startswith('_')]}")
        return True  # 세션 생성까지는 성공
    
    # 첫 번째 사용 가능한 템플릿 사용
    template_file = available_templates[0].name
    print(f"사용할 템플릿 파일: {template_file}")
    print(f"사용 가능한 템플릿 목록: {[t.name for t in available_templates]}")
    
    try:
        result1 = handler.start_editing(
            blend_template=template_file,
            arch_degree=15.5,
            y_scale=1.2
        )
        print("[OK] 편집 시작 성공")
        print(f"  결과: {result1}")
        
    except Exception as e:
        print(f"[ERROR] 편집 시작 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. 변환 적용 테스트 (선택적)
    print(f"\n=== 3. 변환 적용 테스트 ===")
    try:
        result2 = handler.transform(
            arch_degree=20.0,
            y_scale=0.8
        )
        print("[OK] 변환 적용 성공")
        print(f"  결과: {result2}")
        
    except Exception as e:
        print(f"[ERROR] 변환 적용 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. 편집 종료 및 STL 내보내기 테스트
    print(f"\n=== 4. 편집 종료 및 STL 내보내기 테스트 ===")
    try:
        result3 = handler.stop_editing(
            arch_degree=30.0,
            y_scale=1.0
        )
        print("[OK] 편집 종료 성공")
        print(f"  결과: {result3}")
        
        if result3 and 'stl_folder_path' in result3:
            stl_folder = result3['stl_folder_path']
            print(f"\n  STL 파일 경로: {stl_folder}")
            
            # STL 파일 확인
            if os.path.exists(stl_folder):
                stl_files = list(Path(stl_folder).glob("*.stl"))
                print(f"  생성된 STL 파일 수: {len(stl_files)}")
                for stl_file in stl_files[:5]:  # 처음 5개만 출력
                    print(f"    - {stl_file.name}")
            else:
                print(f"  [WARNING] STL 폴더가 존재하지 않습니다: {stl_folder}")
        
    except Exception as e:
        print(f"[ERROR] 편집 종료 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n=== 모든 테스트 완료 ===")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("teeth-template-editor 모듈 테스트")
    print("=" * 60)
    
    success = test_template_editor()
    
    if success:
        print("\n[SUCCESS] 테스트 성공")
        sys.exit(0)
    else:
        print("\n[FAILED] 테스트 실패")
        sys.exit(1)

