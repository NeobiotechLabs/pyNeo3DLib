import os
import logging

# TensorFlow 경고 메시지 숨기기
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # GPU 사용 비활성화

# TensorFlow 로깅 레벨 설정
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from pyNeo3DLib.teethTemplateFinder.teethTemplateFinder import TeethTemplateFinder


def print_separator(title):
    """구분선과 제목 출력"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def print_search_results(results, title="Search Results"):
    """검색 결과 출력"""
    print(f"\n{title}:")
    print(f"Found {len(results)} templates")
    
    if not results:
        print("  No templates found.")
        return
    
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"  Template ID: {result['template_id']}")
        print(f"  Score: {result['score']:.4f}")
        
        payload = result['payload']
        print(f"  Maxilla/Mandibular: {payload.get('maxilla_madibular_type', 'N/A')}")
        print(f"  Arch Type: {payload.get('arch_type', 'N/A')}")
        print(f"  Shape Type: {payload.get('teeth_shape_type', 'N/A')}")
        print(f"  Height Type: {payload.get('teeth_height_type', 'N/A')}")
        print(f"  Size Type: {payload.get('teeth_size_type', 'N/A')}")
        print(f"  Arch Depth: {payload.get('arch_depth', 'N/A')}")
        print(f"  Molar Width: {payload.get('molar_width', 'N/A')}")
        print(f"  Landmarks Count: {len(payload.get('landmarks', []))}")
        print(f"  Filename: {payload.get('filename', 'N/A')}")


def test_basic_search(finder):
    """기본 검색 테스트"""
    print_separator("Basic Search Test")
    
    # 샘플 랜드마크 데이터 (5개 랜드마크 - 12차원 벡터에 맞춤)
    search_landmarks = [
        [10.0, 15.0], [12.0, 16.5], [14.0, 18.0], [16.5, 20.0], [18.5, 21.0]
    ]
    
    try:
        results = finder.find_template(
            arch_depth=45.0,
            molar_width=35.0,
            landmarks=search_landmarks,
            top_k=5
        )
        print_search_results(results, "Basic Search Results")
        
    except Exception as e:
        print(f"Basic search failed: {e}")


def test_filtered_search(finder):
    """필터링된 검색 테스트"""
    print_separator("Filtered Search Test")
    
    search_landmarks = [
        [8.0, 12.0], [10.5, 14.2], [12.8, 16.1], [15.2, 18.3], [17.6, 20.0]
    ]
    
    # 1. Shape type 필터 테스트
    try:
        results = finder.find_template(
            arch_depth=42.5,
            molar_width=33.8,
            landmarks=search_landmarks,
            teeth_shape_type="square",
            top_k=3
        )
        print_search_results(results, "Shape Type Filter: 'square'")
        
    except Exception as e:
        print(f"Shape type filter search failed: {e}")
    
    # 2. 복합 필터 테스트
    try:
        results = finder.find_template(
            arch_depth=42.5,
            molar_width=33.8,
            landmarks=search_landmarks,
            teeth_shape_type="oval",
            teeth_height_type="medium",
            teeth_size_type="large",
            top_k=3
        )
        print_search_results(results, "Multiple Filters: oval + medium + large")
        
    except Exception as e:
        print(f"Multiple filter search failed: {e}")


def test_different_measurements(finder):
    """다양한 측정값으로 검색 테스트"""
    print_separator("Different Measurements Test")
    
    test_cases = [
        {
            "name": "Small Arch",
            "arch_depth": 38.0,
            "molar_width": 30.0,
            "landmarks": [[5.0, 8.0], [7.0, 9.5], [9.0, 11.0], [11.0, 12.5], [13.0, 14.0]]
        },
        {
            "name": "Large Arch", 
            "arch_depth": 52.0,
            "molar_width": 42.0,
            "landmarks": [[15.0, 20.0], [18.0, 22.5], [21.0, 25.0], [24.0, 27.5], [27.0, 30.0]]
        },
        {
            "name": "Narrow Arch",
            "arch_depth": 45.0,
            "molar_width": 28.0,
            "landmarks": [[12.0, 15.0], [14.0, 17.0], [16.0, 19.0], [18.0, 21.0], [20.0, 23.0]]
        }
    ]
    
    for case in test_cases:
        try:
            results = finder.find_template(
                arch_depth=case["arch_depth"],
                molar_width=case["molar_width"],
                landmarks=case["landmarks"],
                top_k=2
            )
            print_search_results(results, f"{case['name']} Results")
            
        except Exception as e:
            print(f"{case['name']} search failed: {e}")


def test_edge_cases(finder):
    """엣지 케이스 테스트"""
    print_separator("Edge Cases Test")
    
    # 1. 정확한 랜드마크 수 (5개)
    try:
        results = finder.find_template(
            arch_depth=40.0,
            molar_width=32.0,
            landmarks=[[10.0, 12.0], [15.0, 18.0], [20.0, 22.0], [25.0, 26.0], [30.0, 28.0]],
            top_k=1
        )
        print_search_results(results, "Standard Landmarks (5 points)")
        
    except Exception as e:
        print(f"Standard landmarks test failed: {e}")
    
    # 2. 존재하지 않는 필터
    try:
        results = finder.find_template(
            arch_depth=45.0,
            molar_width=35.0,
            landmarks=[[10.0, 12.0], [15.0, 18.0], [20.0, 22.0], [25.0, 26.0], [30.0, 28.0]],
            teeth_shape_type="nonexistent_type",
            top_k=1
        )
        print_search_results(results, "Non-existent Filter Type")
        
    except Exception as e:
        print(f"Non-existent filter test failed: {e}")


def main():
    """메인 테스트 함수"""
    print("Starting Teeth Template Finder Tests")
    
    # 템플릿 파인더 초기화
    teeth_template_finder = TeethTemplateFinder()
    db_path = os.path.join(os.path.dirname(__file__), "../pyNeo3DLib/teethTemplateFinder/templateDB")
    
    try:
        # 데이터베이스 연결
        print(f"Connecting to database: {db_path}")
        teeth_template_finder.start_template_finder(db_path)
        print("Database connection successful!")
        
        # 각종 테스트 실행
        test_basic_search(teeth_template_finder)
        test_filtered_search(teeth_template_finder)
        test_different_measurements(teeth_template_finder)
        test_edge_cases(teeth_template_finder)
        
        print_separator("All Tests Completed")
        print("All search tests have been completed successfully!")
        
    except Exception as e:
        print(f"Test setup failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()