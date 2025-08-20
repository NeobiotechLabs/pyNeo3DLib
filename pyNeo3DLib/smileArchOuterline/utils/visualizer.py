import numpy as np
import pyvista as pv


class VisualizeForTest:
    def __init__(self):
        """
        시각화를 위한 클래스 초기화
        
        입력: 없음
        출력: None
        """
        self.plotter = pv.Plotter(window_size=[800, 600])
        self.plotter.add_axes()
        
        # 카메라 위치 직접 설정
        self.plotter.camera_position = [0, -100, 0]  # 카메라 위치 [x, y, z]
        self.plotter.camera.focal_point = [0, 0, 0]   # 카메라가 바라보는 초점
        self.plotter.camera.up = [0, 0, 1]            # 카메라의 상단 방향
    
    def visualize_points_with_index(self, points: np.ndarray, color: str = 'red', point_size: int = 1) -> 'VisualizeForTest':
        """
        포인트 클라우드를 시각화하는 메서드. 각 포인트를 개별적으로 표시하고 인접한 포인트들 간에 선으로 연결됩니다.
        첫 번째 포인트는 크기 5, 마지막 포인트는 크기 10으로 표시됩니다.
        
        Args:
            points: 시각화할 포인트 클라우드 (numpy 배열, 형태: (n, 3))
            color: 포인트 색상 (기본값: 'red')
            point_size: 중간 포인트들의 크기 (기본값: 1)
        """
        # 각 포인트를 개별적으로 시각화
        for i, point in enumerate(points):
            # 첫 번째 포인트는 크기 5
            if i == 0:
                current_point_size = 5
            # 마지막 포인트는 크기 10
            elif i == len(points) - 1:
                current_point_size = 10
            # 나머지 포인트는 기본 크기
            else:
                current_point_size = point_size
                
            self.visualize_points(
                np.array([point], dtype=np.float32), 
                color=color, 
                point_size=current_point_size
            )
            
            # 인접 포인트와 연결선 그리기
            if i < len(points) - 1:
                self.visualize_line(
                    np.array([point], dtype=np.float32),
                    np.array([points[i + 1]], dtype=np.float32),
                    color='black',
                    line_width=1
                )
        
        return self
    
    def visualize_points(self, points: np.ndarray, color: str = 'red', point_size: int = 2) -> 'VisualizeForTest':
        """
        포인트 클라우드를 시각화하는 메서드
        
        Args:
            points: 시각화할 포인트 클라우드 (numpy 배열, 형태: (n, 3))
            color: 포인트 색상 (기본값: 'red')
            point_size: 포인트 크기 (기본값: 10)
            
        Returns:
            self: 메서드 체이닝을 위해 self 반환
        """
        # 포인트 클라우드를 PyVista PolyData로 변환 (float32로 변환하여 경고 방지)
        points_float32 = np.asarray(points, dtype=np.float32)
        point_cloud = pv.PolyData(points_float32)
        
        # 포인트 클라우드를 plotter에 추가
        self.plotter.add_mesh(point_cloud, color=color, point_size=point_size, render_points_as_spheres=True)
        
        return self
    
    def visualize_line(self, start_points: np.ndarray, end_points: np.ndarray, color: str = 'white', line_width: int = 3) -> 'VisualizeForTest':
        """
        여러 개의 선을 그리는 함수
        
        Args:
            start_points: 선의 시작점 좌표 (numpy 배열, 형태: (n, 3))
            end_points: 선의 끝점 좌표 (numpy 배열, 형태: (n, 3))
            color: 선 색상 (기본값: 'white')
            
        Returns:
            self: 메서드 체이닝을 위해 self 반환
        """
        # 각 점마다 선을 그림 (float32로 변환하여 경고 방지)
        start_points_float32 = np.asarray(start_points, dtype=np.float32)
        end_points_float32 = np.asarray(end_points, dtype=np.float32)
        
        for i in range(len(start_points_float32)):
            line = pv.Line(start_points_float32[i], end_points_float32[i])
            self.plotter.add_mesh(line, color=color, line_width=line_width)
        return self
    
    def visualize_mesh(self, mesh: pv.PolyData, color: str = 'white', opacity: float = 0.5) -> 'VisualizeForTest':
        """
        메쉬를 시각화하는 메서드
        
        Args:
            mesh: 시각화할 메쉬 (pyvista.PolyData 또는 다른 pyvista 객체)
            color: 메쉬 색상 (기본값: 'white')
            opacity: 메쉬 투명도 (기본값: 0.5)
            
        Returns:
            self: 메서드 체이닝을 위해 self 반환
        """
        self.plotter.add_mesh(mesh, color=color, opacity=opacity)
        return self
    
    def show(self) -> None:
        """
        시각화 결과를 화면에 표시
        
        입력: 없음
        출력: None
        """
        self.plotter.show() 