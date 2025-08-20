import numpy as np
from typing import Union, List, Tuple, cast
from numpy.typing import NDArray
from scipy.interpolate import make_interp_spline


class SplineGenerator:
    
    def __init__(self):
        pass

           
    def create_spline_curve(
        self,
        control_points: Union[List[Tuple[float, float, float]], NDArray[np.float64]], 
        num_points: int = 100
    ) -> NDArray[np.float64]:
        """3D 제어점들로부터 B-스플라인 곡선을 생성합니다.

        Args:
            control_points: 3D 제어점들의 배열 또는 리스트. 각 점은 (x, y, z) 좌표를 가짐.
                          shape: (N, 3) 여기서 N은 제어점의 개수
            num_points: 생성할 스플라인 곡선 위의 점들의 개수. 기본값: 100

        Returns:
            생성된 스플라인 곡선 위의 점들. shape: (num_points, 3)

        Raises:
            ValueError: 제어점이 4개 미만일 경우
            Exception: 스플라인 생성 중 발생하는 기타 오류
        """
        
        # 입력 데이터를 numpy 배열로 변환
        points_array = np.array(control_points, dtype=np.float64)
        
        if len(points_array) < 4:  # 최소 4개의 제어점 필요
            raise ValueError("최소 4개의 제어점이 필요합니다.")
            
        # # 정렬된 x 좌표를 기준으로 제어점 재정렬
        # sorted_indices = np.argsort(points_array[:, 0])
        # points_array = points_array[sorted_indices]
        
        try:
            t = np.linspace(0, 1, len(points_array))
            t_new = np.linspace(0, 1, num_points)
            
            splines = []
            for i in range(3):
                # 3차 스플라인 곡선
                spl = make_interp_spline(t, points_array[:, i], k=min(3, len(points_array)-1))
                splines.append(spl(t_new))
                
            return cast(NDArray[np.float64], np.column_stack(splines))
        except Exception as e:
            print("스플라인 생성 중 오류 발생:", str(e))
            raise 
    