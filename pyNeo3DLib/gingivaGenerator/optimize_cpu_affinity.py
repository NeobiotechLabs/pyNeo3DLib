"""
CPU Affinity 최적화를 통한 병렬 처리 성능 개선

이 스크립트는 각 프로세스를 특정 CPU 코어에 할당하여
리소스 경쟁을 최소화합니다.
"""
import psutil
import sys


def get_optimal_cpu_allocation(num_processes=2):
    """
    프로세스별 최적 CPU 코어 할당 계산
    
    Args:
        num_processes: 프로세스 개수 (기본값: 2)
        
    Returns:
        dict: 프로세스별 CPU 코어 리스트
        예: {"mandibular": [0, 1, 2, 3], "maxilla": [4, 5, 6, 7]}
    """
    cpu_count = psutil.cpu_count(logical=False)  # 물리 코어
    cpu_count_logical = psutil.cpu_count(logical=True)  # 논리 코어
    
    print(f"[CPU INFO] 물리 코어: {cpu_count}개, 논리 코어: {cpu_count_logical}개")
    
    if cpu_count is None or cpu_count < 4:
        print(f"[WARNING] CPU 코어가 부족합니다 ({cpu_count}개). 최적화 비활성화.")
        return None
    
    # 전략: CPU 코어를 반으로 나누어 각 프로세스에 할당
    # 하이퍼스레딩을 고려하여 논리 코어 사용
    
    cores_per_process = cpu_count_logical // num_processes
    
    allocation = {}
    
    if num_processes == 2:
        # mandibular와 maxilla
        mid = cpu_count_logical // 2
        allocation["mandibular"] = list(range(0, mid))
        allocation["maxilla"] = list(range(mid, cpu_count_logical))
        
        print(f"[CPU ALLOCATION] mandibular: 코어 {allocation['mandibular']}")
        print(f"[CPU ALLOCATION] maxilla: 코어 {allocation['maxilla']}")
    else:
        # 일반적인 경우: 균등 분배
        for i in range(num_processes):
            start = i * cores_per_process
            end = (i + 1) * cores_per_process if i < num_processes - 1 else cpu_count_logical
            allocation[f"process_{i}"] = list(range(start, end))
    
    return allocation


def set_process_affinity(arch_type, allocation):
    """
    현재 프로세스에 CPU Affinity 설정
    
    Args:
        arch_type: "mandibular" 또는 "maxilla"
        allocation: CPU 코어 할당 정보
    """
    if allocation is None or arch_type not in allocation:
        print(f"[CPU AFFINITY] {arch_type}: 최적화 미적용 (기본 동작)")
        return False
    
    try:
        cores = allocation[arch_type]
        p = psutil.Process()
        p.cpu_affinity(cores)
        print(f"[CPU AFFINITY] {arch_type}: 코어 {cores}로 제한됨 ✅")
        return True
    except Exception as e:
        print(f"[CPU AFFINITY] {arch_type}: 설정 실패 - {e}")
        return False


def recommend_parallel_mode():
    """
    시스템 CPU 기반으로 병렬 처리 권장 여부 판단
    
    Returns:
        tuple: (권장 여부, 이유)
    """
    cpu_count = psutil.cpu_count(logical=False)
    
    if cpu_count is None:
        return True, "CPU 정보를 가져올 수 없음 (기본값: 병렬)"
    
    if cpu_count >= 12:
        return True, f"CPU 코어 충분 ({cpu_count}개) - 병렬 처리 강력 권장"
    elif cpu_count >= 8:
        return True, f"CPU 코어 적당 ({cpu_count}개) - 병렬 처리 권장"
    elif cpu_count >= 6:
        return True, f"CPU 코어 보통 ({cpu_count}개) - 병렬 처리 유효 (오버헤드 있음)"
    elif cpu_count >= 4:
        return False, f"CPU 코어 부족 ({cpu_count}개) - 순차 처리 권장"
    else:
        return False, f"CPU 코어 매우 부족 ({cpu_count}개) - 순차 처리 강력 권장"


if __name__ == "__main__":
    print("=" * 60)
    print("CPU Affinity 최적화 분석")
    print("=" * 60)
    
    # CPU 정보 출력
    cpu_count = psutil.cpu_count(logical=False)
    cpu_count_logical = psutil.cpu_count(logical=True)
    print(f"\n시스템 정보:")
    print(f"  - 물리 코어: {cpu_count}개")
    print(f"  - 논리 코어: {cpu_count_logical}개")
    
    # 병렬 처리 권장 여부
    recommended, reason = recommend_parallel_mode()
    print(f"\n권장 사항: {'병렬 처리' if recommended else '순차 처리'}")
    print(f"  이유: {reason}")
    
    # CPU 할당 계산
    print(f"\n최적 CPU 코어 할당 (2개 프로세스):")
    allocation = get_optimal_cpu_allocation(2)
    
    if allocation:
        for proc_name, cores in allocation.items():
            print(f"  - {proc_name}: {len(cores)}개 코어 (코어 번호: {cores})")
        
        print(f"\n예상 효과:")
        print(f"  - 리소스 경쟁 최소화")
        print(f"  - 각 프로세스가 독립된 코어 사용")
        print(f"  - 오버헤드 감소 예상")
    else:
        print(f"  - CPU 코어가 부족하여 최적화 불가능")
        print(f"  - 순차 처리 권장")
    
    print("\n" + "=" * 60)

