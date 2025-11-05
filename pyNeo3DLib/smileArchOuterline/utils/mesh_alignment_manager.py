"""
메쉬 정렬 관련 기능을 담당하는 클래스
"""

import numpy as np
import pyvista as pv
from typing import Tuple, Optional, List
from .constants import AnalysisConstants
from .ray_caster import RayCaster


class MeshAlignmentManager:
    """메쉬 정렬을 관리하는 클래스"""
    
    # 상수 정의
    EPSILON = 1e-10
    NEAR_PARALLEL_DOT_PRODUCT_THRESHOLD = 0.9
    STANDARD_BASIS = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    
    def __init__(self):
        self.ray_caster = RayCaster()
        self.x_axis_intersection_count = AnalysisConstants.X_AXIS_INTERSECTION_COUNT
        self.y_axis_intersection_count = AnalysisConstants.Y_AXIS_INTERSECTION_COUNT
        self.z_axis_intersection_count = AnalysisConstants.Z_AXIS_INTERSECTION_COUNT
    
    # ==================== 벡터 유틸리티 메서드 ====================
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
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
        if norm < self.EPSILON:
            raise ValueError("영벡터는 정규화할 수 없습니다.")
        return (vector / norm).astype(np.float64)
    
    def _orthogonalize_vector(
        self, 
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
        vector = self._normalize_vector(vector)
        reference = self._normalize_vector(reference)
        
        dot_product = np.dot(vector, reference)
        orthogonal = vector - dot_product * reference
        norm = np.linalg.norm(orthogonal)
        
        if norm < self.EPSILON:
            return None
        return (orthogonal / norm).astype(np.float64)
    
    def _get_vector_from_closest_intersection(
        self, 
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
        return self._normalize_vector(vector)
    
    # ==================== 직교 벡터 찾기 메서드 ====================
    
    def _find_orthogonal_from_unused_principal_evecs(
        self, 
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
        reference = self._normalize_vector(reference)
        
        # 사용되지 않은 벡터만 시도
        for i in range(3):
            if i not in used_indices:
                candidate = self._normalize_vector(principal_evecs[:, i])
                orthogonal = self._orthogonalize_vector(candidate, reference)
                if orthogonal is not None:
                    return orthogonal
        
        return None
    
    def _find_orthogonal_from_all_principal_evecs(
        self, 
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
        reference = self._normalize_vector(reference)
        
        # 모든 주축 벡터 시도
        for i in range(3):
            candidate = self._normalize_vector(principal_evecs[:, i])
            orthogonal = self._orthogonalize_vector(candidate, reference)
            if orthogonal is not None:
                return orthogonal
        
        return None
    
    def _find_orthogonal_from_standard_basis(
        self, 
        reference: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        표준 기저 벡터 중 reference와 가장 직교하는 벡터를 찾습니다.
        
        Args:
            reference: 기준 벡터 (3,)
            
        Returns:
            직교 벡터 (3,) 또는 None
        """
        reference = self._normalize_vector(reference)
        
        best_candidate = None
        min_dot_product = float('inf')
        
        for basis_vec in self.STANDARD_BASIS:
            dot_product = abs(np.dot(reference, basis_vec))
            if dot_product < min_dot_product:
                min_dot_product = dot_product
                best_candidate = basis_vec
        
        return self._orthogonalize_vector(best_candidate, reference)
    
    def _find_fallback_orthogonal(
        self, 
        reference: np.ndarray
    ) -> np.ndarray:
        """
        최후의 수단으로 reference와 직교하는 벡터를 생성합니다.
        
        Args:
            reference: 기준 벡터 (3,)
            
        Returns:
            직교 벡터 (3,) - 항상 유효한 값 반환 보장
        """
        reference = self._normalize_vector(reference)
        
        # reference의 성분을 기반으로 적절한 표준 기저 벡터 선택
        abs_ref = np.abs(reference)
        if abs_ref[0] < self.NEAR_PARALLEL_DOT_PRODUCT_THRESHOLD:
            candidate = np.array([1.0, 0.0, 0.0])
        elif abs_ref[1] < self.NEAR_PARALLEL_DOT_PRODUCT_THRESHOLD:
            candidate = np.array([0.0, 1.0, 0.0])
        else:
            candidate = np.array([0.0, 0.0, 1.0])
        
        orthogonal = self._orthogonalize_vector(candidate, reference)
        if orthogonal is not None:
            return orthogonal
        
        # 완전히 실패한 경우 크로스 곱으로 생성
        # reference가 [1,0,0] 또는 [-1,0,0]에 가까우면 fallback = [0,1,0]
        # 그 외의 경우 fallback = [1,0,0]
        if abs_ref[0] < self.NEAR_PARALLEL_DOT_PRODUCT_THRESHOLD:
            fallback = np.array([1.0, 0.0, 0.0])
        else:
            fallback = np.array([0.0, 1.0, 0.0])
        
        cross_result = np.cross(reference, fallback)
        cross_norm = np.linalg.norm(cross_result)
        
        # 크로스 곱이 영벡터가 아닌 경우 직교화 시도
        if cross_norm >= self.EPSILON:
            orthogonal = self._orthogonalize_vector(cross_result, reference)
            if orthogonal is not None:
                return orthogonal
        
        # 최후의 수단: reference와 가장 다른 표준 기저 벡터 사용
        # reference의 각 성분의 절댓값을 확인하여 가장 작은 성분의 축을 선택
        min_idx = np.argmin(abs_ref)
        result = np.zeros(3, dtype=np.float64)
        result[min_idx] = 1.0
        
        # 선택한 벡터가 reference와 평행한 경우 다른 축 선택
        if abs(np.dot(result, reference)) > self.NEAR_PARALLEL_DOT_PRODUCT_THRESHOLD:
            # 다음으로 작은 성분의 축 선택
            sorted_indices = np.argsort(abs_ref)
            for idx in sorted_indices:
                candidate = np.zeros(3, dtype=np.float64)
                candidate[idx] = 1.0
                if abs(np.dot(candidate, reference)) < self.NEAR_PARALLEL_DOT_PRODUCT_THRESHOLD:
                    result = candidate
                    break
        
        # 최종적으로 직교화 시도
        orthogonal = self._orthogonalize_vector(result, reference)
        if orthogonal is not None:
            return orthogonal
        
        # 절대 실패하지 않도록 보장: reference와 크로스 곱으로 생성 가능한 벡터 찾기
        for basis_vec in self.STANDARD_BASIS:
            cross_vec = np.cross(reference, basis_vec)
            cross_norm = np.linalg.norm(cross_vec)
            if cross_norm >= self.EPSILON:
                return self._normalize_vector(cross_vec)
        
        # 이론적으로 도달 불가능하지만 안전을 위해 기본값 반환
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    
    # ==================== 축 결정 메서드 ====================
    
    def _determine_y_axis_from_ray_casting(
        self, 
        input_mesh: pv.PolyData, 
        center: np.ndarray, 
        principal_evecs: np.ndarray,
        used_indices: List[int]
    ) -> Tuple[np.ndarray, List[int]]:
        """
        레이 캐스팅 결과를 기반으로 Y축 벡터를 결정합니다.
        
        케이스 1: 교차점 개수 == 1개 → 가장 가까운 교차점으로 벡터 생성
        케이스 2: 교차점 개수 == 0개 → 해당 주축 벡터 사용
        케이스 3: 위 실패 시 → 사용되지 않은 주축 벡터 사용
        
        Args:
            input_mesh: 입력 메쉬
            center: 중심점 (3,)
            principal_evecs: 주축 벡터 행렬 (3, 3)
            used_indices: 이미 사용된 인덱스 리스트
            
        Returns:
            (Y축 벡터 (3,), 업데이트된 used_indices)
        """
        center = np.asarray(center).flatten()
        if center.shape != (3,):
            raise ValueError(f"center는 (3,) 형태여야 합니다. 현재 shape: {center.shape}")
        
        # 모든 축에 대해 레이캐스팅 수행
        intersection_results = [
            self.ray_caster.get_bidirectional_ray_points(
                input_mesh, center, principal_evecs[:, i]
            ) for i in range(3)
        ]

        # 케이스 1: 교차점 개수가 y_axis_intersection_count인 경우
        for i in range(3):
            if len(intersection_results[i]) == self.y_axis_intersection_count:
                evec_y = self._get_vector_from_closest_intersection(intersection_results[i], center)
                used_indices.append(i)
                return evec_y, used_indices
        
        # 케이스 2: 교차점이 0개인 경우
        for i in range(3):
            if len(intersection_results[i]) == 0:
                evec_y = self._normalize_vector(principal_evecs[:, i])
                used_indices.append(i)
                return evec_y, used_indices
        
        # 케이스 3: 백업 로직 - 사용되지 않은 주축 벡터 사용
        for i in range(3):
            if i not in used_indices:
                evec_y = self._normalize_vector(principal_evecs[:, i])
                used_indices.append(i)
                return evec_y, used_indices
        
        # 모든 벡터가 사용된 경우 첫 번째 주축 벡터 사용
        evec_y = self._normalize_vector(principal_evecs[:, 0])
        return evec_y, used_indices
    
    def _determine_z_axis_from_ray_casting(
        self, 
        input_mesh: pv.PolyData, 
        center: np.ndarray, 
        principal_evecs: np.ndarray,
        evec_y: np.ndarray,
        used_indices: List[int]
    ) -> Tuple[np.ndarray, List[int]]:
        """
        레이 캐스팅 결과를 기반으로 Z축 벡터를 결정합니다.
        
        케이스 1: 교차점 개수 == 2개 → 가장 가까운 교차점으로 벡터 생성 후 Y축과 직교화
        케이스 2: 케이스 1 실패 → 사용되지 않은 주축 벡터 중 Y축과 직교하는 벡터 선택
        케이스 3: 케이스 2 실패 → 모든 주축 벡터 중 Y축과 직교하는 벡터 선택
        케이스 4: 케이스 3 실패 → 표준 기저 벡터([1,0,0], [0,1,0], [0,0,1]) 중 Y축과 가장 직교하는 벡터 선택
        
        Args:
            input_mesh: 입력 메쉬
            center: 중심점 (3,)
            principal_evecs: 주축 벡터 행렬 (3, 3)
            evec_y: Y축 벡터 (3,)
            used_indices: 이미 사용된 인덱스 리스트
            
        Returns:
            (Z축 벡터 (3,), 업데이트된 used_indices)
        """
        center = np.asarray(center).flatten()
        if center.shape != (3,):
            raise ValueError(f"center는 (3,) 형태여야 합니다. 현재 shape: {center.shape}")
        
        evec_y = self._normalize_vector(evec_y)
        
        # 케이스 1: 교차점 개수가 z_axis_intersection_count (2개)인 경우
        for i in range(3):
            intersection_points = self.ray_caster.get_bidirectional_ray_points(
                input_mesh, center, principal_evecs[:, i]
            )
            
            if len(intersection_points) == self.z_axis_intersection_count:
                evec_z = self._get_vector_from_closest_intersection(intersection_points, center)
                
                # Y축과 직교하도록 조정
                orthogonal = self._orthogonalize_vector(evec_z, evec_y)
                if orthogonal is not None:
                    evec_z = orthogonal
                else:
                    # 직교화 실패 시 주축 벡터를 사용하여 직교 벡터 생성
                    candidate = self._normalize_vector(principal_evecs[:, i])
                    evec_z = self._orthogonalize_vector(candidate, evec_y)
                    if evec_z is None:
                        evec_z = candidate
                
                used_indices.append(i)
                return evec_z, used_indices
        
        # 케이스 2: 사용되지 않은 주축 벡터 중 Y축과 직교하는 벡터 선택
        evec_z = self._find_orthogonal_from_unused_principal_evecs(principal_evecs, evec_y, used_indices)
        if evec_z is not None:
            return evec_z, used_indices
        
        # 케이스 3: 모든 주축 벡터 중 Y축과 직교하는 벡터 선택
        evec_z = self._find_orthogonal_from_all_principal_evecs(principal_evecs, evec_y)
        if evec_z is not None:
            return evec_z, used_indices
        
        # 케이스 4: 표준 기저 벡터 중 Y축과 가장 직교하는 벡터 선택
        evec_z = self._find_orthogonal_from_standard_basis(evec_y)
        if evec_z is not None:
            return evec_z, used_indices
        
        # 최후의 수단 (fallback)
        evec_z = self._find_fallback_orthogonal(evec_y)
        return evec_z, used_indices
    
    def _determine_x_axis_from_cross_product(
        self, 
        evec_y: np.ndarray, 
        evec_z: np.ndarray,
        principal_evecs: np.ndarray,
        used_indices: List[int]
    ) -> np.ndarray:
        """
        Y축과 Z축 벡터로부터 X축 벡터를 결정합니다.
        
        케이스 1: evec_y × evec_z (외적)로 계산
        케이스 2: 외적이 영벡터 → 표준 기저 벡터 사용
        케이스 3: 케이스 2 실패 → 주축 벡터 사용
        케이스 4: 케이스 3 실패 → evec_y 성분 기반으로 표준 기저 벡터 선택
        
        Args:
            evec_y: Y축 벡터 (3,)
            evec_z: Z축 벡터 (3,)
            principal_evecs: 주축 벡터 행렬 (3, 3)
            used_indices: 이미 사용된 인덱스 리스트
            
        Returns:
            X축 벡터 (3,) - 항상 유효한 값 반환 보장
        """
        evec_y = self._normalize_vector(evec_y)
        evec_z = self._normalize_vector(evec_z)
        
        # 케이스 1: 외적으로 계산
        evec_x = np.cross(evec_y, evec_z)
        evec_x_norm = np.linalg.norm(evec_x)
        
        if evec_x_norm >= self.EPSILON:
            return self._normalize_vector(evec_x)
        
        # 케이스 2: 외적이 영벡터인 경우 표준 기저 벡터 사용
        evec_x = self._find_orthogonal_from_standard_basis(evec_y)
        if evec_x is not None:
            return evec_x
        
        # 케이스 3: 주축 벡터 사용 (사용되지 않은 것부터 시도)
        evec_x = self._find_orthogonal_from_unused_principal_evecs(principal_evecs, evec_y, used_indices)
        if evec_x is not None:
            return evec_x
        
        # 모든 주축 벡터 시도
        evec_x = self._find_orthogonal_from_all_principal_evecs(principal_evecs, evec_y)
        if evec_x is not None:
            return evec_x
        
        # 케이스 4: evec_y 성분 기반으로 표준 기저 벡터 선택
        return self._find_fallback_orthogonal(evec_y)
    
    # ==================== 공개 메서드 ====================
    
    def determine_alignment_axes(
        self,
        input_mesh: pv.PolyData, 
        center: np.ndarray, 
        principal_evecs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        메쉬의 주축 벡터를 분석하여 정렬에 사용할 각 축을 결정합니다.
        
        양방향 레이 캐스팅의 교차점 개수를 기준으로:
        - 4개 교차점: X축 정렬용
        - 1개 교차점: Y축 정렬용
        - 2개 교차점: Z축 정렬용
        
        벡터 결정 순서:
        1. Y축: 레이 캐스팅 결과 → 백업 로직
        2. Z축: 레이 캐스팅 결과 → 주축 벡터 → 표준 기저 벡터
        3. X축: Y축 × Z축 (외적) → 백업 로직
        
        Args:
            input_mesh: PyVista PolyData 객체
            center: 메쉬 중심점 (3,) 또는 (1, 3)
            principal_evecs: 주축 벡터 행렬 (3, 3)
            
        Returns:
            Tuple[evec_x (3,), evec_y (3,), evec_z (3,)] - 모든 벡터는 (3,) 형태로 반환
        """
        # 입력 검증 및 정규화
        center = np.asarray(center).flatten()
        if center.shape != (3,):
            raise ValueError(f"center는 (3,) 형태여야 합니다. 현재 shape: {center.shape}")
        
        if principal_evecs.shape != (3, 3):
            raise ValueError(f"principal_evecs는 (3, 3) 형태여야 합니다. 현재 shape: {principal_evecs.shape}")
        
        used_indices: List[int] = []
        
        # 1단계: Y축 결정
        evec_y, used_indices = self._determine_y_axis_from_ray_casting(
            input_mesh, center, principal_evecs, used_indices
        )
        
        # 2단계: Z축 결정
        evec_z, used_indices = self._determine_z_axis_from_ray_casting(
            input_mesh, center, principal_evecs, evec_y, used_indices
        )
        
        # 3단계: X축 결정 (Y축과 Z축의 외적)
        evec_x = self._determine_x_axis_from_cross_product(
            evec_y, evec_z, principal_evecs, used_indices
        )
        
        # 모든 벡터를 (3,) 형태로 보장
        return (
            self._normalize_vector(evec_x),
            self._normalize_vector(evec_y),
            self._normalize_vector(evec_z)
        )
    
    def align_mesh_to_global_coordinates(
        self,
        input_mesh: pv.PolyData, 
        evec_x: np.ndarray, 
        evec_y: np.ndarray, 
        evec_z: np.ndarray
    ) -> pv.PolyData:
        """
        주축 벡터를 사용하여 메쉬를 글로벌 좌표계로 정렬합니다.
        
        목표 정렬:
        - evec_x → 글로벌 X축 (1, 0, 0)
        - evec_y → 글로벌 Y축 (0, 1, 0)
        - evec_z → 글로벌 Z축 (0, 0, 1)
        
        Args:
            input_mesh: 정렬할 메쉬
            evec_x: X축 정렬용 주축 벡터 (3,)
            evec_y: Y축 정렬용 주축 벡터 (3,)
            evec_z: Z축 정렬용 주축 벡터 (3,)
            
        Returns:
            aligned_mesh: 정렬된 메쉬
        """
        # 벡터 정규화 및 검증
        evec_x = self._normalize_vector(evec_x)
        evec_y = self._normalize_vector(evec_y)
        evec_z = self._normalize_vector(evec_z)
        
        # 기저 행렬 구성 (3, 3)
        basis_matrix = np.array([evec_x, evec_y, evec_z], dtype=np.float64)
        
        # 역행렬 계산
        try:
            inverse_basis = np.linalg.inv(basis_matrix)
        except np.linalg.LinAlgError:
            raise ValueError("기저 행렬이 특이행렬입니다. 벡터들이 선형 독립인지 확인하세요.")
        
        # 목표 좌표계 정의 (3, 3)
        target_matrix = np.array([
            [1.0, 0.0, 0.0],  # X축
            [0.0, 1.0, 0.0],  # Y축
            [0.0, 0.0, 1.0]   # Z축
        ], dtype=np.float64)
        
        # 회전 행렬 계산 및 적용
        rotation_matrix = np.matmul(inverse_basis, target_matrix)
        aligned_vertices = np.matmul(input_mesh.points, rotation_matrix)

        aligned_mesh = input_mesh.copy()
        aligned_mesh.points = aligned_vertices.astype(np.float64)

        return aligned_mesh
    
    def filter_mesh_by_z_threshold(
        self, 
        aligned_mesh: pv.PolyData, 
        filtered_points: np.ndarray
    ) -> pv.PolyData:
        """
        Z 임계값을 기준으로 메쉬 필터링
        
        Args:
            aligned_mesh: 정렬된 메쉬
            filtered_points: 필터링된 포인트들 (N, 3)
            
        Returns:
            filtered_aligned_mesh: 필터링된 메쉬
        """
        if len(filtered_points) == 0:
            raise ValueError("filtered_points가 비어있습니다.")
        
        if filtered_points.shape[1] != 3:
            raise ValueError(f"filtered_points는 (N, 3) 형태여야 합니다. 현재 shape: {filtered_points.shape}")
        
        z_min_point = np.min(filtered_points[:, 2])
        mask = aligned_mesh.points[:, 2] > z_min_point
        filtered_aligned_mesh = aligned_mesh.extract_points(mask)

        # 가장 큰 덩어리만 추출
        largest_component = filtered_aligned_mesh.extract_largest()

        # 중복된 면이나 점이 있는 경우 제거
        filtered_largest_component = largest_component.clean()
        
        return filtered_largest_component

