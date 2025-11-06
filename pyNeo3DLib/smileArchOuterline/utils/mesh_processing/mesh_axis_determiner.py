import numpy as np
import pyvista as pv
from typing import Tuple, List
from pyNeo3DLib.smileArchOuterline.utils.common.constants import AnalysisConstants
from pyNeo3DLib.smileArchOuterline.utils.ray_casting.ray_caster import RayCaster
from pyNeo3DLib.smileArchOuterline.utils.common.vector_utils import VectorUtils


class MeshAxisDeterminer:
    """메쉬의 주축 벡터를 분석하여 정렬에 사용할 각 축을 결정하는 클래스"""

    def __init__(self):
        self.ray_caster = RayCaster()
        self.x_axis_intersection_count = AnalysisConstants.X_AXIS_INTERSECTION_COUNT
        self.y_axis_intersection_count = AnalysisConstants.Y_AXIS_INTERSECTION_COUNT
        self.z_axis_intersection_count = AnalysisConstants.Z_AXIS_INTERSECTION_COUNT

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
                evec_y = VectorUtils.get_vector_from_closest_intersection(intersection_results[i], center)
                used_indices.append(i)
                return evec_y, used_indices
        
        # 케이스 2: 교차점이 0개인 경우
        for i in range(3):
            if len(intersection_results[i]) == 0:
                evec_y = VectorUtils.normalize_vector(principal_evecs[:, i])
                used_indices.append(i)
                return evec_y, used_indices
        
        # 케이스 3: 백업 로직 - 사용되지 않은 주축 벡터 사용
        for i in range(3):
            if i not in used_indices:
                evec_y = VectorUtils.normalize_vector(principal_evecs[:, i])
                used_indices.append(i)
                return evec_y, used_indices
        
        # 모든 벡터가 사용된 경우 첫 번째 주축 벡터 사용
        evec_y = VectorUtils.normalize_vector(principal_evecs[:, 0])
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
        
        evec_y = VectorUtils.normalize_vector(evec_y)
        
        # 케이스 1: 교차점 개수가 z_axis_intersection_count (2개)인 경우
        for i in range(3):
            intersection_points = self.ray_caster.get_bidirectional_ray_points(
                input_mesh, center, principal_evecs[:, i]
            )
            
            if len(intersection_points) == self.z_axis_intersection_count:
                evec_z = VectorUtils.get_vector_from_closest_intersection(intersection_points, center)
                
                # Y축과 직교하도록 조정
                orthogonal = VectorUtils.orthogonalize_vector(evec_z, evec_y)
                if orthogonal is not None:
                    evec_z = orthogonal
                else:
                    # 직교화 실패 시 주축 벡터를 사용하여 직교 벡터 생성
                    candidate = VectorUtils.normalize_vector(principal_evecs[:, i])
                    evec_z = VectorUtils.orthogonalize_vector(candidate, evec_y)
                    if evec_z is None:
                        evec_z = candidate
                
                used_indices.append(i)
                return evec_z, used_indices
        
        # 케이스 2: 사용되지 않은 주축 벡터 중 Y축과 직교하는 벡터 선택
        evec_z = VectorUtils.find_orthogonal_from_unused_principal_evecs(principal_evecs, evec_y, used_indices)
        if evec_z is not None:
            return evec_z, used_indices
        
        # 케이스 3: 모든 주축 벡터 중 Y축과 직교하는 벡터 선택
        evec_z = VectorUtils.find_orthogonal_from_all_principal_evecs(principal_evecs, evec_y)
        if evec_z is not None:
            return evec_z, used_indices
        
        # 케이스 4: 표준 기저 벡터 중 Y축과 가장 직교하는 벡터 선택
        evec_z = VectorUtils.find_orthogonal_from_standard_basis(evec_y)
        if evec_z is not None:
            return evec_z, used_indices
        
        # 최후의 수단 (fallback)
        evec_z = VectorUtils.find_fallback_orthogonal(evec_y)
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
        evec_y = VectorUtils.normalize_vector(evec_y)
        evec_z = VectorUtils.normalize_vector(evec_z)
        
        # 케이스 1: 외적으로 계산
        evec_x = np.cross(evec_y, evec_z)
        evec_x_norm = np.linalg.norm(evec_x)
        
        if evec_x_norm >= VectorUtils.EPSILON:
            return VectorUtils.normalize_vector(evec_x)
        
        # 케이스 2: 외적이 영벡터인 경우 표준 기저 벡터 사용
        evec_x = VectorUtils.find_orthogonal_from_standard_basis(evec_y)
        if evec_x is not None:
            return evec_x
        
        # 케이스 3: 주축 벡터 사용 (사용되지 않은 것부터 시도)
        evec_x = VectorUtils.find_orthogonal_from_unused_principal_evecs(principal_evecs, evec_y, used_indices)
        if evec_x is not None:
            return evec_x
        
        # 모든 주축 벡터 시도
        evec_x = VectorUtils.find_orthogonal_from_all_principal_evecs(principal_evecs, evec_y)
        if evec_x is not None:
            return evec_x
        
        # 케이스 4: evec_y 성분 기반으로 표준 기저 벡터 선택
        return VectorUtils.find_fallback_orthogonal(evec_y)
    
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
            VectorUtils.normalize_vector(evec_x),
            VectorUtils.normalize_vector(evec_y),
            VectorUtils.normalize_vector(evec_z)
        )
