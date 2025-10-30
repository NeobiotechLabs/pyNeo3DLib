"""
치은 생성을 별도 프로세스에서 실행하기 위한 스크립트
"""
import sys
import json
import os

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python run_gingiva_generation.py <input_path> <output_path> <arch_types_json>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    arch_types = json.loads(sys.argv[3])
    
    print("=" * 60)
    print("치은 생성 프로세스 시작")
    print("=" * 60)
    print(f"입력 경로: {input_path}")
    print(f"출력 경로: {output_path}")
    print(f"생성 타입: {arch_types}")
    print("=" * 60)
    
    # 입력 경로 검증
    if not os.path.exists(input_path):
        print(f"오류: 입력 경로가 존재하지 않습니다: {input_path}")
        sys.exit(1)
    
    try:
        print("\n[1/3] TeethTemplateMaker 모듈 import 중...")
        from single_template_maker_lib import TeethTemplateMaker
        print("[OK] TeethTemplateMaker import 완료")
        
        print("\n[2/3] TeethTemplateMaker 인스턴스 생성 중...")
        template_maker = TeethTemplateMaker(input_path, output_path, arch_types)
        print("[OK] 인스턴스 생성 완료")
        
        print("\n[3/3] 치은 생성 실행 중...")
        print("(이 작업은 수 분이 걸릴 수 있습니다...)")
        template_maker.run()
        
        print("\n" + "=" * 60)
        print("[SUCCESS] 치은 생성 완료!")
        print("=" * 60)
        
        # 생성된 파일 확인
        for arch_type in arch_types:
            output_file = os.path.join(output_path, f"{arch_type}.stl")
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                print(f"생성된 파일: {output_file} ({file_size:,} bytes)")
            else:
                print(f"경고: 예상 파일이 생성되지 않음: {output_file}")
        
        sys.exit(0)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("[ERROR] 오류 발생!")
        print("=" * 60)
        print(f"오류 메시지: {str(e)}")
        print("\n상세 정보:")
        import traceback
        traceback.print_exc()
        sys.exit(1)

