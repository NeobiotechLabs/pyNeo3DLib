import os
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint, Filter, FieldCondition, MatchValue
from typing import List, Dict, Optional


class TeethTemplateFinder:
    DEFAULT_TOP_K = 5
    COLLECTION_NAME = "tooth_templates"
    
    def __init__(self):
        self.db_path = os.path.join(os.path.dirname(__file__), "templateDB")
        self.collection_name = self.COLLECTION_NAME
        self.client = None
        print(f"Database path: {self.db_path}")
    
    def start_template_finder(self, db_path: str):
        """템플릿 파인더 시작"""
        self.db_path = db_path

        try:
            # 로컬 파일 시스템 데이터베이스 연결
            self.client = QdrantClient(path=self.db_path)
            print("QdrantClient initialized successfully")
        except Exception as e:
            print(f"Error initializing QdrantClient: {e}")
            raise e

    def find_template(
        self,
        arch_depth: float,
        molar_width: float,
        landmarks: List[List[float]],
        teeth_shape_type: Optional[str] = None,
        teeth_height_type: Optional[str] = None,
        teeth_size_type: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K
    ) -> List[Dict]:
        """템플릿 검색"""
        try:
            print(f"Searching templates with arch_depth: {arch_depth}, molar_width: {molar_width}, landmarks count: {len(landmarks)}")
            
            query_vector = self._create_query_vector(arch_depth, molar_width, landmarks)
            
            # 필터 조건 생성
            query_filter = self._create_filter(teeth_shape_type, teeth_height_type, teeth_size_type)
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k
            )
            
            formatted_results = [self._format_search_result(result) for result in results]
            print(f"Found {len(formatted_results)} templates")
            return formatted_results
            
        except Exception as e:
            print(f"Error finding template: {e}")
            raise e

    def _create_filter(self, teeth_shape_type: Optional[str], teeth_height_type: Optional[str], teeth_size_type: Optional[str]):
        """검색 필터 생성"""
        conditions = []
        
        if teeth_shape_type:
            conditions.append(FieldCondition(key="teeth_shape_type", match=MatchValue(value=teeth_shape_type)))
        
        if teeth_height_type:
            conditions.append(FieldCondition(key="teeth_height_type", match=MatchValue(value=teeth_height_type)))
            
        if teeth_size_type:
            conditions.append(FieldCondition(key="teeth_size_type", match=MatchValue(value=teeth_size_type)))
        
        if conditions:
            return Filter(must=conditions)
        
        return None

    def _create_query_vector(self, arch_depth: float, molar_width: float, landmarks: List[List[float]]) -> List[float]:
        """쿼리 벡터 생성"""
        self._validate_parameters(arch_depth, molar_width)
        processed_landmarks = self.preprocess_landmarks(landmarks)
        return [arch_depth, molar_width] + [v for xy in processed_landmarks for v in xy]

    def preprocess_landmarks(self, landmarks: List[List[float]]) -> List[List[float]]:
        """랜드마크 전처리"""
        if not landmarks:
            raise ValueError("landmarks는 비어있을 수 없습니다")
        
        # 랜드마크 정규화 또는 기타 전처리 로직
        # 현재는 단순히 입력을 그대로 반환
        processed = []
        for landmark in landmarks:
            if len(landmark) != 2:
                raise ValueError(f"각 랜드마크는 [x, y] 형태여야 합니다. 현재: {landmark}")
            processed.append([float(landmark[0]), float(landmark[1])])
        
        return processed

    def _validate_parameters(self, arch_depth: float, molar_width: float) -> None:
        """파라미터 유효성 검사"""
        if not isinstance(arch_depth, (int, float)) or arch_depth <= 0:
            raise ValueError("arch_depth는 양수여야 합니다")
        
        if not isinstance(molar_width, (int, float)) or molar_width <= 0:
            raise ValueError("molar_width는 양수여야 합니다")

    def _format_search_result(self, result: ScoredPoint) -> Dict:
        """검색 결과 포맷팅"""
        return {
            "template_id": result.id,
            "score": result.score,
            "payload": {
                "maxilla_madibular_type": result.payload.get("maxilla_madibular_type"),
                "arch_type": result.payload.get("arch_type"),
                "teeth_shape_type": result.payload.get("teeth_shape_type"),
                "teeth_height_type": result.payload.get("teeth_height_type"),
                "teeth_size_type": result.payload.get("teeth_size_type"),
                "arch_depth": result.payload.get("arch_depth"),
                "molar_width": result.payload.get("molar_width"),
                "landmarks": result.payload.get("landmarks"),
                "filename": result.payload.get("filename")
            }
        }


if __name__ == "__main__":
    print("Hello, World!")
    teeth_template_finder = TeethTemplateFinder()
    
    # 테스트 예시
    db_path = os.path.join(os.path.dirname(__file__), "templateDB")
    teeth_template_finder.start_template_finder(db_path)
    
    # 샘플 데이터로 테스트
    sample_landmarks = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    
    try:
        results = teeth_template_finder.find_template(
            arch_depth=10.5,
            molar_width=15.2,
            landmarks=sample_landmarks,
            top_k=3
        )
        print(f"Found {len(results)} templates")
    except Exception as e:
        print(f"Error: {e}")