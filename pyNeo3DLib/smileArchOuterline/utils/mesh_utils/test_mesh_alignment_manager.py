"""
determine_alignment_axes 함수에 대한 포괄적인 테스트
다양한 경우의 수를 테스트하여 None 값이 반환되지 않도록 보장합니다.
"""

import unittest
from unittest.mock import Mock, patch
import numpy as np
import pyvista as pv
import sys
import os

# 상대 import와 절대 import 모두 지원
try:
    from .mesh_alignment_manager import MeshAlignmentManager
    from .constants import AnalysisConstants
except ImportError:
    # 직접 실행하는 경우: 프로젝트 루트를 sys.path에 추가
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # pyNeo3DLib/pyNeo3DLib/... 경로까지 올라가기
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from pyNeo3DLib.smileArchOuterline.utils.mesh_alignment_manager import MeshAlignmentManager
    from pyNeo3DLib.smileArchOuterline.utils.constants import AnalysisConstants


class TestDetermineAlignmentAxes(unittest.TestCase):
    """determine_alignment_axes 함수의 다양한 경우의 수를 테스트합니다."""
    
    def setUp(self):
        """테스트 준비"""
        self.manager = MeshAlignmentManager()
        self.mock_mesh = Mock(spec=pv.PolyData)
        self.center = np.array([0.0, 0.0, 0.0])
        
        # 기본 주축 벡터 (직교, 정규화됨)
        self.standard_evecs = np.array([
            [1.0, 0.0, 0.0],  # 첫 번째 주축
            [0.0, 1.0, 0.0],  # 두 번째 주축
            [0.0, 0.0, 1.0]   # 세 번째 주축
        ]).T
    
    def _create_intersection_points(self, num_points, offset=1.0):
        """교차점 생성 헬퍼 함수"""
        if num_points == 0:
            return np.array([]).reshape(0, 3)
        
        points = []
        for i in range(num_points):
            # 중심점에서 약간 떨어진 위치에 교차점 생성
            point = self.center + np.array([offset * (i + 1), 0.0, 0.0])
            points.append(point)
        
        return np.array(points)
    
    def test_case1_normal_y_and_z_determined(self):
        """케이스 1: Y축과 Z축이 정상적으로 결정되는 경우"""
        # 첫 번째 주축: Y축 (교차점 1개)
        # 두 번째 주축: Z축 (교차점 2개)
        # 세 번째 주축: 다른 값 (교차점 4개)
        
        intersection_counts = [
            AnalysisConstants.Y_AXIS_INTERSECTION_COUNT,  # 첫 번째 주축
            AnalysisConstants.Z_AXIS_INTERSECTION_COUNT,  # 두 번째 주축
            4  # 세 번째 주축
        ]
        
        def mock_get_bidirectional_ray_points(mesh, center, evec):
            idx = None
            for i in range(3):
                if np.allclose(evec, self.standard_evecs[:, i], atol=1e-6):
                    idx = i
                    break
            
            count = intersection_counts[idx] if idx is not None else 0
            return self._create_intersection_points(count)
        
        with patch.object(self.manager.ray_caster, 'get_bidirectional_ray_points', 
                         side_effect=mock_get_bidirectional_ray_points):
            evec_x, evec_y, evec_z = self.manager.determine_alignment_axes(
                self.mock_mesh, self.center, self.standard_evecs
            )
        
        # 모든 축이 None이 아니어야 함
        self.assertIsNotNone(evec_x, "X축이 None입니다")
        self.assertIsNotNone(evec_y, "Y축이 None입니다")
        self.assertIsNotNone(evec_z, "Z축이 None입니다")
        
        # 모든 축이 3차원 벡터여야 함
        self.assertEqual(len(evec_x), 3, "X축이 3차원 벡터가 아닙니다")
        self.assertEqual(len(evec_y), 3, "Y축이 3차원 벡터가 아닙니다")
        self.assertEqual(len(evec_z), 3, "Z축이 3차원 벡터가 아닙니다")
    
    def test_case2_y_fallback_zero_intersections(self):
        """케이스 2: Y축이 교차점 0개로 fallback되는 경우"""
        # 첫 번째 주축: 교차점 0개 (Y축 fallback)
        # 두 번째 주축: Z축 (교차점 2개)
        # 세 번째 주축: 다른 값
        
        intersection_counts = [0, AnalysisConstants.Z_AXIS_INTERSECTION_COUNT, 4]
        
        def mock_get_bidirectional_ray_points(mesh, center, evec):
            idx = None
            for i in range(3):
                if np.allclose(evec, self.standard_evecs[:, i], atol=1e-6):
                    idx = i
                    break
            
            count = intersection_counts[idx] if idx is not None else 0
            return self._create_intersection_points(count)
        
        with patch.object(self.manager.ray_caster, 'get_bidirectional_ray_points', 
                         side_effect=mock_get_bidirectional_ray_points):
            evec_x, evec_y, evec_z = self.manager.determine_alignment_axes(
                self.mock_mesh, self.center, self.standard_evecs
            )
        
        self.assertIsNotNone(evec_x)
        self.assertIsNotNone(evec_y)
        self.assertIsNotNone(evec_z)
    
    def test_case3_y_default_fallback(self):
        """케이스 3: Y축이 결정되지 않아 기본값을 사용하는 경우"""
        # 모든 주축이 Y축이나 Z축 조건을 만족하지 않음
        intersection_counts = [3, 3, 3]  # 모두 다른 값
        
        def mock_get_bidirectional_ray_points(mesh, center, evec):
            idx = None
            for i in range(3):
                if np.allclose(evec, self.standard_evecs[:, i], atol=1e-6):
                    idx = i
                    break
            
            count = intersection_counts[idx] if idx is not None else 0
            return self._create_intersection_points(count)
        
        with patch.object(self.manager.ray_caster, 'get_bidirectional_ray_points', 
                         side_effect=mock_get_bidirectional_ray_points):
            evec_x, evec_y, evec_z = self.manager.determine_alignment_axes(
                self.mock_mesh, self.center, self.standard_evecs
            )
        
        self.assertIsNotNone(evec_x)
        self.assertIsNotNone(evec_y)
        self.assertIsNotNone(evec_z)
    
    def test_case4_z_orthogonal_fallback(self):
        """케이스 4: Z축이 결정되지 않아 직교하는 주축을 선택하는 경우"""
        # 첫 번째 주축: Y축 (교차점 1개)
        # 나머지: Z축 조건 불만족
        intersection_counts = [AnalysisConstants.Y_AXIS_INTERSECTION_COUNT, 3, 3]
        
        def mock_get_bidirectional_ray_points(mesh, center, evec):
            idx = None
            for i in range(3):
                if np.allclose(evec, self.standard_evecs[:, i], atol=1e-6):
                    idx = i
                    break
            
            count = intersection_counts[idx] if idx is not None else 0
            return self._create_intersection_points(count)
        
        with patch.object(self.manager.ray_caster, 'get_bidirectional_ray_points', 
                         side_effect=mock_get_bidirectional_ray_points):
            evec_x, evec_y, evec_z = self.manager.determine_alignment_axes(
                self.mock_mesh, self.center, self.standard_evecs
            )
        
        self.assertIsNotNone(evec_x)
        self.assertIsNotNone(evec_y)
        self.assertIsNotNone(evec_z)
    
    def test_case5_parallel_vectors_retry(self):
        """케이스 5: Y축과 Z축이 평행하여 외적이 영벡터가 되는 경우"""
        # Y축과 Z축이 평행한 주축 벡터 사용
        parallel_evecs = np.array([
            [1.0, 0.0, 0.0],  # 첫 번째 주축
            [1.0, 0.0, 0.0],  # 두 번째 주축 (첫 번째와 평행)
            [0.0, 0.0, 1.0]   # 세 번째 주축
        ]).T
        
        intersection_counts = [
            AnalysisConstants.Y_AXIS_INTERSECTION_COUNT,
            AnalysisConstants.Z_AXIS_INTERSECTION_COUNT,
            3
        ]
        
        def mock_get_bidirectional_ray_points(mesh, center, evec):
            idx = None
            for i in range(3):
                if np.allclose(evec, parallel_evecs[:, i], atol=1e-6):
                    idx = i
                    break
            
            count = intersection_counts[idx] if idx is not None else 0
            return self._create_intersection_points(count)
        
        with patch.object(self.manager.ray_caster, 'get_bidirectional_ray_points', 
                         side_effect=mock_get_bidirectional_ray_points):
            evec_x, evec_y, evec_z = self.manager.determine_alignment_axes(
                self.mock_mesh, self.center, parallel_evecs
            )
        
        self.assertIsNotNone(evec_x)
        self.assertIsNotNone(evec_y)
        self.assertIsNotNone(evec_z)
    
    def test_case6_gram_schmidt_fallback(self):
        """케이스 6: 외적이 실패하여 Gram-Schmidt 과정을 사용하는 경우"""
        # Y축과 평행한 Z축 벡터
        gram_schmidt_evecs = np.array([
            [1.0, 0.0, 0.0],  # Y축
            [1.0, 0.001, 0.0],  # Y축과 거의 평행한 벡터
            [0.0, 0.0, 1.0]
        ]).T
        
        intersection_counts = [
            AnalysisConstants.Y_AXIS_INTERSECTION_COUNT,
            AnalysisConstants.Z_AXIS_INTERSECTION_COUNT,
            3
        ]
        
        def mock_get_bidirectional_ray_points(mesh, center, evec):
            idx = None
            for i in range(3):
                if np.allclose(evec, gram_schmidt_evecs[:, i], atol=1e-3):
                    idx = i
                    break
            
            count = intersection_counts[idx] if idx is not None else 0
            return self._create_intersection_points(count)
        
        with patch.object(self.manager.ray_caster, 'get_bidirectional_ray_points', 
                         side_effect=mock_get_bidirectional_ray_points):
            evec_x, evec_y, evec_z = self.manager.determine_alignment_axes(
                self.mock_mesh, self.center, gram_schmidt_evecs
            )
        
        self.assertIsNotNone(evec_x)
        self.assertIsNotNone(evec_y)
        self.assertIsNotNone(evec_z)
    
    def test_case7_final_principal_axis_search(self):
        """케이스 7: 모든 방법이 실패하여 주축 벡터 중에서 찾는 경우"""
        # Y축과 Z축이 평행한 경우 (외적이 영벡터가 되는 경우)
        parallel_evecs = np.array([
            [1.0, 0.0, 0.0],  # 첫 번째 주축
            [1.0, 0.0, 0.0],  # 두 번째 주축 (첫 번째와 평행)
            [0.0, 0.0, 1.0]   # 세 번째 주축
        ]).T
        
        intersection_counts = [
            AnalysisConstants.Y_AXIS_INTERSECTION_COUNT,
            AnalysisConstants.Z_AXIS_INTERSECTION_COUNT,
            3
        ]
        
        def mock_get_bidirectional_ray_points(mesh, center, evec):
            idx = None
            min_diff = float('inf')
            for i in range(3):
                diff = np.linalg.norm(evec - parallel_evecs[:, i])
                if diff < min_diff:
                    min_diff = diff
                    idx = i
            
            count = intersection_counts[idx] if idx is not None else 0
            return self._create_intersection_points(count)
        
        with patch.object(self.manager.ray_caster, 'get_bidirectional_ray_points', 
                         side_effect=mock_get_bidirectional_ray_points):
            evec_x, evec_y, evec_z = self.manager.determine_alignment_axes(
                self.mock_mesh, self.center, parallel_evecs
            )
        
        self.assertIsNotNone(evec_x)
        self.assertIsNotNone(evec_y)
        self.assertIsNotNone(evec_z)
        
        # evec_x가 nan이 아니어야 함
        self.assertFalse(np.any(np.isnan(evec_x)), "evec_x가 nan입니다")
        self.assertFalse(np.any(np.isnan(evec_y)), "evec_y가 nan입니다")
        self.assertFalse(np.any(np.isnan(evec_z)), "evec_z가 nan입니다")
    
    def test_case8_rotated_principal_axes(self):
        """케이스 8: 회전된 주축 벡터들"""
        # 45도 회전된 주축 벡터
        angle = np.pi / 4
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        rotated_evecs = np.array([
            [cos_a, sin_a, 0.0],
            [-sin_a, cos_a, 0.0],
            [0.0, 0.0, 1.0]
        ]).T
        
        intersection_counts = [
            AnalysisConstants.Y_AXIS_INTERSECTION_COUNT,
            AnalysisConstants.Z_AXIS_INTERSECTION_COUNT,
            3
        ]
        
        def mock_get_bidirectional_ray_points(mesh, center, evec):
            idx = None
            min_diff = float('inf')
            for i in range(3):
                diff = np.linalg.norm(evec - rotated_evecs[:, i])
                if diff < min_diff:
                    min_diff = diff
                    idx = i
            
            count = intersection_counts[idx] if idx is not None else 0
            return self._create_intersection_points(count)
        
        with patch.object(self.manager.ray_caster, 'get_bidirectional_ray_points', 
                         side_effect=mock_get_bidirectional_ray_points):
            evec_x, evec_y, evec_z = self.manager.determine_alignment_axes(
                self.mock_mesh, self.center, rotated_evecs
            )
        
        self.assertIsNotNone(evec_x)
        self.assertIsNotNone(evec_y)
        self.assertIsNotNone(evec_z)
    
    def test_case9_all_axes_zero_intersections(self):
        """케이스 9: 모든 주축의 교차점이 0개인 경우"""
        intersection_counts = [0, 0, 0]
        
        def mock_get_bidirectional_ray_points(mesh, center, evec):
            idx = None
            for i in range(3):
                if np.allclose(evec, self.standard_evecs[:, i], atol=1e-6):
                    idx = i
                    break
            
            count = intersection_counts[idx] if idx is not None else 0
            return self._create_intersection_points(count)
        
        with patch.object(self.manager.ray_caster, 'get_bidirectional_ray_points', 
                         side_effect=mock_get_bidirectional_ray_points):
            evec_x, evec_y, evec_z = self.manager.determine_alignment_axes(
                self.mock_mesh, self.center, self.standard_evecs
            )
        
        self.assertIsNotNone(evec_x)
        self.assertIsNotNone(evec_y)
        self.assertIsNotNone(evec_z)
    
    def test_case10_y_determined_z_not_determined(self):
        """케이스 10: Y축만 결정되고 Z축은 결정되지 않는 경우"""
        intersection_counts = [
            AnalysisConstants.Y_AXIS_INTERSECTION_COUNT,
            3,  # Z축 조건 불만족
            4   # Z축 조건 불만족
        ]
        
        def mock_get_bidirectional_ray_points(mesh, center, evec):
            idx = None
            for i in range(3):
                if np.allclose(evec, self.standard_evecs[:, i], atol=1e-6):
                    idx = i
                    break
            
            count = intersection_counts[idx] if idx is not None else 0
            return self._create_intersection_points(count)
        
        with patch.object(self.manager.ray_caster, 'get_bidirectional_ray_points', 
                         side_effect=mock_get_bidirectional_ray_points):
            evec_x, evec_y, evec_z = self.manager.determine_alignment_axes(
                self.mock_mesh, self.center, self.standard_evecs
            )
        
        self.assertIsNotNone(evec_x)
        self.assertIsNotNone(evec_y)
        self.assertIsNotNone(evec_z)
    
    def test_case11_orthogonality_check(self):
        """케이스 11: 반환된 축들이 직교하는지 확인"""
        intersection_counts = [
            AnalysisConstants.Y_AXIS_INTERSECTION_COUNT,
            AnalysisConstants.Z_AXIS_INTERSECTION_COUNT,
            3
        ]
        
        def mock_get_bidirectional_ray_points(mesh, center, evec):
            idx = None
            for i in range(3):
                if np.allclose(evec, self.standard_evecs[:, i], atol=1e-6):
                    idx = i
                    break
            
            count = intersection_counts[idx] if idx is not None else 0
            return self._create_intersection_points(count)
        
        with patch.object(self.manager.ray_caster, 'get_bidirectional_ray_points', 
                         side_effect=mock_get_bidirectional_ray_points):
            evec_x, evec_y, evec_z = self.manager.determine_alignment_axes(
                self.mock_mesh, self.center, self.standard_evecs
            )
        
        # X축과 Y축의 내적 (거의 0이어야 함)
        dot_xy = abs(np.dot(evec_x, evec_y))
        self.assertLess(dot_xy, 1e-6, f"X축과 Y축이 직교하지 않습니다 (내적: {dot_xy})")
        
        # X축과 Z축의 내적
        dot_xz = abs(np.dot(evec_x, evec_z))
        self.assertLess(dot_xz, 1e-6, f"X축과 Z축이 직교하지 않습니다 (내적: {dot_xz})")
        
        # Y축과 Z축의 내적
        dot_yz = abs(np.dot(evec_y, evec_z))
        self.assertLess(dot_yz, 1e-6, f"Y축과 Z축이 직교하지 않습니다 (내적: {dot_yz})")
    
    def test_case12_normalized_vectors(self):
        """케이스 12: 반환된 벡터들이 정규화되었는지 확인"""
        intersection_counts = [
            AnalysisConstants.Y_AXIS_INTERSECTION_COUNT,
            AnalysisConstants.Z_AXIS_INTERSECTION_COUNT,
            3
        ]
        
        def mock_get_bidirectional_ray_points(mesh, center, evec):
            idx = None
            for i in range(3):
                if np.allclose(evec, self.standard_evecs[:, i], atol=1e-6):
                    idx = i
                    break
            
            count = intersection_counts[idx] if idx is not None else 0
            return self._create_intersection_points(count)
        
        with patch.object(self.manager.ray_caster, 'get_bidirectional_ray_points', 
                         side_effect=mock_get_bidirectional_ray_points):
            evec_x, evec_y, evec_z = self.manager.determine_alignment_axes(
                self.mock_mesh, self.center, self.standard_evecs
            )
        
        # 모든 벡터의 크기가 1에 가까워야 함
        norm_x = np.linalg.norm(evec_x)
        norm_y = np.linalg.norm(evec_y)
        norm_z = np.linalg.norm(evec_z)
        
        self.assertAlmostEqual(norm_x, 1.0, places=6, msg=f"X축이 정규화되지 않았습니다 (크기: {norm_x})")
        self.assertAlmostEqual(norm_y, 1.0, places=6, msg=f"Y축이 정규화되지 않았습니다 (크기: {norm_y})")
        self.assertAlmostEqual(norm_z, 1.0, places=6, msg=f"Z축이 정규화되지 않았습니다 (크기: {norm_z})")


def run_all_tests():
    """모든 테스트를 실행하고 결과를 출력합니다."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestDetermineAlignmentAxes)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*70)
    print("테스트 결과 요약")
    print("="*70)
    print(f"총 테스트: {result.testsRun}")
    print(f"성공: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"실패: {len(result.failures)}")
    print(f"에러: {len(result.errors)}")
    
    if result.failures:
        print("\n실패한 테스트:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print("\n에러가 발생한 테스트:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)

