import numpy as np
from typing import Optional, List

class VectorUtils:
    """벡터 관련 유틸리티 함수들을 제공하는 클래스"""

    EPSILON = 1e-10
    NEAR_PARALLEL_DOT_PRODUCT_THRESHOLD = 0.9
    STANDARD_BASIS = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    @staticmethod
    def rotate_vector_around_axis(vector: np.ndarray, axis: np.ndarray, angle_degrees: float) -> np.ndarray:
        """
        주어진 축을 중심으로 벡터를 회전시킵니다.

        Args:
            vector: 회전시킬 벡터 (numpy array, shape: (3,))
            axis: 회전축 벡터 (numpy array, shape: (3,))
            angle_degrees: 회전 각도 (도 단위)

        Returns:
            rotated_vector: 회전된 벡터 (numpy array, shape: (3,))
        """
        # 입력 검증
        if axis is None:
            raise ValueError("회전축이 None입니다.")

        # axis를 1차원 배열로 변환
        axis = np.asarray(axis).flatten()

        if len(axis) != 3:
            raise ValueError(f"회전축은 3차원 벡터여야 합니다. 현재 형태: {axis.shape}")

        # vector를 1차원 배열로 변환
        vector = np.asarray(vector).flatten()

        if len(vector) != 3:
            raise ValueError(f"벡터는 3차원 벡터여야 합니다. 현재 형태: {vector.shape}")

        # 각도를 라디안으로 변환
        angle_rad = np.radians(angle_degrees)

        # 회전축 정규화
        axis_norm = np.linalg.norm(axis)
        if axis_norm == 0:
            raise ValueError("회전축의 크기가 0입니다.")

        axis = axis / axis_norm

        # 로드리게스 회전 공식 사용
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

        # 외적 행렬 (skew-symmetric matrix)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])

        # 회전 행렬 계산: R = I + sin(θ)K + (1-cos(θ))K²
        I = np.eye(3)
        R = I + sin_angle * K + (1 - cos_angle) * np.dot(K, K)

        # 벡터 회전
        return np.dot(R, vector)

    @staticmethod
    def normalize_vector(vector: np.ndarray) -> np.ndarray:
        """
        벡터를 (3,) 형태로 정규화합니다.
        
        Args:
            vector: 입력 벡터 (다양한 shape 가능)
            
        Returns:
            정규화된 벡터 (3,) 형태
        """
        vector = np.asarray(vector).flatten()
        if len(vector) != 3:
            raise ValueError(f"벡터는 3차원이어야 합니다. 현재 shape: {vector.shape}")
        norm = np.linalg.norm(vector)
        if norm < VectorUtils.EPSILON:
            raise ValueError("영벡터는 정규화할 수 없습니다.")
        return (vector / norm).astype(np.float64)
    
    @staticmethod
    def orthogonalize_vector(
        vector: np.ndarray, 
        reference: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Gram-Schmidt 과정을 사용하여 벡터를 reference 벡터에 직교하도록 만듭니다.
        
        Args:
            vector: 직교화할 벡터 (3,)
            reference: 기준 벡터 (3,)
            
        Returns:
            직교화된 벡터 (3,) 또는 None (직교화 실패 시)
        """
        vector = VectorUtils.normalize_vector(vector)
        reference = VectorUtils.normalize_vector(reference)
        
        dot_product = np.dot(vector, reference)
        orthogonal = vector - dot_product * reference
        norm = np.linalg.norm(orthogonal)
        
        if norm < VectorUtils.EPSILON:
            return None
        return (orthogonal / norm).astype(np.float64)
    
    @staticmethod
    def get_vector_from_closest_intersection(
        intersection_points: np.ndarray, 
        center: np.ndarray
    ) -> np.ndarray:
        """
        교차점 중 중심에 가장 가까운 점으로부터 정규화된 벡터를 생성합니다.
        
        Args:
            intersection_points: 교차점 배열 (N, 3)
            center: 중심점 (3,)
            
        Returns:
            정규화된 벡터 (3,)
        """
        if len(intersection_points) == 0:
            raise ValueError("교차점이 없습니다.")
        
        center = np.asarray(center).flatten()
        if center.shape != (3,):
            raise ValueError(f"center는 (3,) 형태여야 합니다. 현재 shape: {center.shape}")
        
        distances = np.linalg.norm(intersection_points - center, axis=1)
        closest_point = intersection_points[np.argmin(distances)]
        vector = closest_point - center
        return VectorUtils.normalize_vector(vector)
    
    @staticmethod
    def find_orthogonal_from_unused_principal_evecs(
        principal_evecs: np.ndarray, 
        reference: np.ndarray,
        used_indices: List[int]
    ) -> Optional[np.ndarray]:
        """
        사용되지 않은 주축 벡터 중 reference와 직교하는 벡터를 찾습니다.
        
        Args:
            principal_evecs: 주축 벡터 행렬 (3, 3)
            reference: 기준 벡터 (3,)
            used_indices: 이미 사용된 인덱스 리스트
            
        Returns:
            직교 벡터 (3,) 또는 None
        """
        reference = VectorUtils.normalize_vector(reference)
        
        # 사용되지 않은 벡터만 시도
        for i in range(3):
            if i not in used_indices:
                candidate = VectorUtils.normalize_vector(principal_evecs[:, i])
                orthogonal = VectorUtils.orthogonalize_vector(candidate, reference)
                if orthogonal is not None:
                    return orthogonal
        
        return None
    
    @staticmethod
    def find_orthogonal_from_all_principal_evecs(
        principal_evecs: np.ndarray, 
        reference: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        모든 주축 벡터 중 reference와 직교하는 벡터를 찾습니다.
        
        Args:
            principal_evecs: 주축 벡터 행렬 (3, 3)
            reference: 기준 벡터 (3,)
            
        Returns:
            직교 벡터 (3,) 또는 None
        """
        reference = VectorUtils.normalize_vector(reference)
        
        # 모든 주축 벡터 시도
        for i in range(3):
            candidate = VectorUtils.normalize_vector(principal_evecs[:, i])
            orthogonal = VectorUtils.orthogonalize_vector(candidate, reference)
            if orthogonal is not None:
                return orthogonal
        
        return None
    
    @staticmethod
    def find_orthogonal_from_standard_basis(
        reference: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        표준 기저 벡터 중 reference와 가장 직교하는 벡터를 찾습니다.
        
        Args:
            reference: 기준 벡터 (3,)
            
        Returns:
            직교 벡터 (3,) 또는 None
        """
        reference = VectorUtils.normalize_vector(reference)
        
        best_candidate = None
        min_dot_product = float('inf')
        
        for basis_vec in VectorUtils.STANDARD_BASIS:
            dot_product = abs(np.dot(reference, basis_vec))
            if dot_product < min_dot_product:
                min_dot_product = dot_product
                best_candidate = basis_vec
        
        return VectorUtils.orthogonalize_vector(best_candidate, reference)
    
    @staticmethod
    def find_fallback_orthogonal(
        reference: np.ndarray
    ) -> np.ndarray:
        """
        최후의 수단으로 reference와 직교하는 벡터를 생성합니다.
        
        Args:
            reference: 기준 벡터 (3,)
            
        Returns:
            직교 벡터 (3,) - 항상 유효한 값 반환 보장
        """
        reference = VectorUtils.normalize_vector(reference)
        
        # reference의 성분을 기반으로 적절한 표준 기저 벡터 선택
        abs_ref = np.abs(reference)
        if abs_ref[0] < VectorUtils.NEAR_PARALLEL_DOT_PRODUCT_THRESHOLD:
            candidate = np.array([1.0, 0.0, 0.0])
        elif abs_ref[1] < VectorUtils.NEAR_PARALLEL_DOT_PRODUCT_THRESHOLD:
            candidate = np.array([0.0, 1.0, 0.0])
        else:
            candidate = np.array([0.0, 0.0, 1.0])
        
        orthogonal = VectorUtils.orthogonalize_vector(candidate, reference)
        if orthogonal is not None:
            return orthogonal
        
        # 완전히 실패한 경우 크로스 곱으로 생성
        # reference가 [1,0,0] 또는 [-1,0,0]에 가까우면 fallback = [0,1,0]
        # 그 외의 경우 fallback = [1,0,0]
        if abs_ref[0] < VectorUtils.NEAR_PARALLEL_DOT_PRODUCT_THRESHOLD:
            fallback = np.array([1.0, 0.0, 0.0])
        else:
            fallback = np.array([0.0, 1.0, 0.0])
        
        cross_result = np.cross(reference, fallback)
        cross_norm = np.linalg.norm(cross_result)
        
        # 크로스 곱이 영벡터가 아닌 경우 직교화 시도
        if cross_norm >= VectorUtils.EPSILON:
            orthogonal = VectorUtils.orthogonalize_vector(cross_result, reference)
            if orthogonal is not None:
                return orthogonal
        
        # 최후의 수단: reference와 가장 다른 표준 기저 벡터 사용
        # reference의 각 성분의 절댓값을 확인하여 가장 작은 성분의 축을 선택
        min_idx = np.argmin(abs_ref)
        result = np.zeros(3, dtype=np.float64)
        result[min_idx] = 1.0
        
        # 선택한 벡터가 reference와 평행한 경우 다른 축 선택
        if abs(np.dot(result, reference)) > VectorUtils.NEAR_PARALLEL_DOT_PRODUCT_THRESHOLD:
            # 다음으로 작은 성분의 축 선택
            sorted_indices = np.argsort(abs_ref)
            for idx in sorted_indices:
                candidate = np.zeros(3, dtype=np.float64)
                candidate[idx] = 1.0
                if abs(np.dot(candidate, reference)) < VectorUtils.NEAR_PARALLEL_DOT_PRODUCT_THRESHOLD:
                    result = candidate
                    break
        
        # 최종적으로 직교화 시도
        orthogonal = VectorUtils.orthogonalize_vector(result, reference)
        if orthogonal is not None:
            return orthogonal
        
        # 절대 실패하지 않도록 보장: reference와 크로스 곱으로 생성 가능한 벡터 찾기
        for basis_vec in VectorUtils.STANDARD_BASIS:
            cross_vec = np.cross(reference, basis_vec)
            cross_norm = np.linalg.norm(cross_vec)
            if cross_norm >= VectorUtils.EPSILON:
                return VectorUtils.normalize_vector(cross_vec)
        
        # 이론적으로 도달 불가능하지만 안전을 위해 기본값 반환
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
