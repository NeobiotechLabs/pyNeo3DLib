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
        self.plotter.camera_position = [0, 0, -100]  # 카메라 위치 [x, y, z]
        self.plotter.camera.focal_point = [0, 0, 0]   # 카메라가 바라보는 초점
        self.plotter.camera.up = [0, 1, 0]            # 카메라의 상단 방향
    
    def visualize_points(self, points: np.ndarray, color: str = 'red', point_size: int = 10) -> 'VisualizeForTest':
        """
        포인트 클라우드를 시각화하는 메서드
        
        Args:
            points: 시각화할 포인트 클라우드 (numpy 배열, 형태: (n, 3))
            color: 포인트 색상 (기본값: 'red')
            point_size: 포인트 크기 (기본값: 10)
            
        Returns:
            self: 메서드 체이닝을 위해 self 반환
        """
        # 포인트 클라우드를 PyVista PolyData로 변환
        point_cloud = pv.PolyData(points)
        
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
        # 각 점마다 선을 그림
        
        for i in range(len(start_points)):
            line = pv.Line(start_points[i], end_points[i])
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