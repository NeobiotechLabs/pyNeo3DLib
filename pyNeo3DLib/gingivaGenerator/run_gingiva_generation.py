"""
치은 생성을 별도 프로세스에서 실행하기 위한 스크립트
"""
import sys
import json
import os
import time
import psutil

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python run_gingiva_generation.py <input_path> <output_path> <arch_types_json> [cpu_affinity_json]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    arch_types = json.loads(sys.argv[3])
    
    # CPU Affinity 설정 (옵션)
    cpu_affinity = None
    if len(sys.argv) >= 5:
        try:
            cpu_affinity = json.loads(sys.argv[4])
            if arch_types[0] in cpu_affinity:
                # CPU Affinity 적용
                p = psutil.Process()
                cores = cpu_affinity[arch_types[0]]
                p.cpu_affinity(cores)
                print(f"[CPU AFFINITY] 코어 {cores}로 제한됨 ✅")
        except Exception as e:
            print(f"[CPU AFFINITY] 설정 실패: {e}")
    
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
    
    total_start = time.perf_counter()
    
    try:
        print("\n[1/3] TeethTemplateMaker 모듈 import 중...")
        import_start = time.perf_counter()
        from single_template_maker_lib import TeethTemplateMaker
        import_elapsed = time.perf_counter() - import_start
        print(f"[OK] TeethTemplateMaker import 완료 ({import_elapsed:.4f}초)")
        
        print("\n[2/3] TeethTemplateMaker 인스턴스 생성 중...")
        init_start = time.perf_counter()
        template_maker = TeethTemplateMaker(input_path, output_path, arch_types)
        init_elapsed = time.perf_counter() - init_start
        print(f"[OK] 인스턴스 생성 완료 ({init_elapsed:.4f}초)")
        
        print("\n[3/3] 치은 생성 실행 중...")
        print("(이 작업은 수 분이 걸릴 수 있습니다...)")
        
        run_start = time.perf_counter()
        template_maker.run()
        run_elapsed = time.perf_counter() - run_start
        
        print("\n" + "=" * 60)
        print("[SUCCESS] 치은 생성 완료!")
        print("=" * 60)
        print(f"[PERF] 실제 생성 시간: {run_elapsed:.4f}초 ({run_elapsed/60:.2f}분)")
        
        # 생성된 파일 확인
        for arch_type in arch_types:
            output_file = os.path.join(output_path, f"{arch_type}.stl")
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                print(f"생성된 파일: {output_file} ({file_size:,} bytes)")
            else:
                print(f"경고: 예상 파일이 생성되지 않음: {output_file}")
        
        # 전체 실행 시간
        total_elapsed = time.perf_counter() - total_start
        print(f"\n[PERF] 전체 소요 시간: {total_elapsed:.4f}초 ({total_elapsed/60:.2f}분)")
        print(f"[PERF]   - Import: {import_elapsed:.4f}초")
        print(f"[PERF]   - 초기화: {init_elapsed:.4f}초")
        print(f"[PERF]   - 실제 생성: {run_elapsed:.4f}초")
        
        sys.exit(0)
        
    except Exception as e:
        total_elapsed = time.perf_counter() - total_start
        print(f"\n[PERF] 오류 발생 전까지 소요 시간: {total_elapsed:.4f}초")
        
        print("\n" + "=" * 60)
        print("[ERROR] 오류 발생!")
        print("=" * 60)
        print(f"오류 메시지: {str(e)}")
        print("\n상세 정보:")
        import traceback
        traceback.print_exc()
        sys.exit(1)

