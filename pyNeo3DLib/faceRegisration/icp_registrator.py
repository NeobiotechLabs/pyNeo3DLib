"""
ICP 정합 모듈

이 모듈은 ICP(Iterative Closest Point) 알고리즘을 사용한 메쉬 정합을 담당합니다.
단일 책임 원칙(SRP)에 따라 ICP 정합 로직만을 캡슐화합니다.
"""
import numpy as np
import open3d as o3d
import copy
import time
from typing import Tuple, Optional
from dataclasses import dataclass

from pyNeo3DLib.fileLoader.mesh import Mesh
from pyNeo3DLib.faceRegisration.mesh_converter import MeshConverter
from pyNeo3DLib.faceRegisration.constants import (
    VisualizationConstants,
    ICPConstants
)


@dataclass
class ICPResult:
    """ICP 정합 결과를 담는 데이터 클래스"""
    transformed_mesh: Mesh
    transformation_matrix: np.ndarray
    fitness: float


class ICPRegistrator:
    """
    ICP 정합을 담당하는 클래스.
    
    단일 책임: ICP 알고리즘을 사용한 메쉬 정합
    
    이 클래스는 다음 기능을 제공합니다:
    - 3단계 ICP 정합 (coarse → medium → fine)
    - 시각화 지원
    - 정합 결과 반환
    """
    
    def __init__(self, visualization: bool = False):
        """
        ICPRegistrator 초기화
        
        Args:
            visualization: 시각화 활성화 여부
        """
        self.visualization = visualization
        self._visualizer: Optional[o3d.visualization.Visualizer] = None
    
    def register(
        self, 
        source_mesh: Mesh, 
        target_mesh: Mesh,
        remove_outliers: bool = True,
        nb_neighbors: int = 20,
        std_ratio: float = 2.0
    ) -> ICPResult:
        """
        소스 메쉬를 타겟 메쉬에 정합합니다.
        
        Args:
            source_mesh: 정합할 소스 메쉬
            target_mesh: 타겟 메쉬
            remove_outliers: 통계적 아웃라이어 제거 활성화 여부
            nb_neighbors: 아웃라이어 검출시 사용할 이웃 개수
            std_ratio: 표준편차 배수 (작을수록 엄격하게 제거)
            
        Returns:
            ICPResult: 정합 결과
        """
        # 메쉬를 포인트클라우드로 변환
        source_pcd = MeshConverter.mesh_to_pointcloud(source_mesh)
        target_pcd = MeshConverter.mesh_to_pointcloud(target_mesh)
        
        # 통계적 아웃라이어 제거 (전처리)
        if remove_outliers:
            source_pcd, _ = self._remove_statistical_outliers(
                source_pcd, nb_neighbors, std_ratio, "Source"
            )
            target_pcd, _ = self._remove_statistical_outliers(
                target_pcd, nb_neighbors, std_ratio, "Target"
            )
        
        # 색상 설정
        source_pcd.paint_uniform_color([1, 0, 0])  # 빨간색
        target_pcd.paint_uniform_color([0, 0, 1])  # 파란색
        
        # 시각화 설정
        if self.visualization:
            self._setup_visualizer()
            self._update_visualization(source_pcd, target_pcd)
        
        # 3단계 ICP 수행
        current_transform = np.eye(4)
        
        print("\nStarting 1st ICP registration (coarse)...")
        current_transform = self._run_icp_stage(
            source_pcd, target_pcd, current_transform,
            ICPConstants.DISTANCE_THRESHOLD_STAGE1, "1st"
        )
        
        print("Starting 2nd ICP registration (medium)...")
        current_transform = self._run_icp_stage(
            source_pcd, target_pcd, current_transform,
            ICPConstants.DISTANCE_THRESHOLD_STAGE2, "2nd"
        )
        
        print("Starting 3rd ICP registration (fine)...")
        current_transform, final_fitness = self._run_icp_stage(
            source_pcd, target_pcd, current_transform,
            ICPConstants.DISTANCE_THRESHOLD_STAGE3, "3rd",
            return_fitness=True
        )
        
        print(f"\n=== Registration completed ===")
        print(f"Final fitness: {final_fitness:.6f}")
        
        # 시각화 종료
        if self.visualization:
            self._close_visualizer()
        
        # 변환된 메쉬 생성
        transformed_mesh = copy.deepcopy(source_mesh)
        transformed_mesh.vertices = np.dot(
            source_mesh.vertices,
            current_transform[:3, :3].T
        ) + current_transform[:3, 3]
        
        return ICPResult(
            transformed_mesh=transformed_mesh,
            transformation_matrix=current_transform,
            fitness=final_fitness
        )
    
    def _remove_statistical_outliers(
        self,
        pcd: o3d.geometry.PointCloud,
        nb_neighbors: int,
        std_ratio: float,
        name: str = ""
    ) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
        """
        통계적 방법으로 아웃라이어를 제거합니다.
        
        각 포인트에서 k개의 이웃까지의 평균 거리를 계산하고,
        전체 평균 + std_ratio * 표준편차를 초과하는 포인트를 아웃라이어로 판정합니다.
        
        Args:
            pcd: 입력 포인트클라우드
            nb_neighbors: 이웃 개수 (k)
            std_ratio: 표준편차 배수 (작을수록 엄격하게 제거)
            name: 포인트클라우드 이름 (로깅용)
            
        Returns:
            Tuple[정제된 포인트클라우드, 아웃라이어 인덱스]
        """
        original_count = len(pcd.points)
        
        pcd_clean, inlier_indices = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
        
        outlier_count = original_count - len(pcd_clean.points)
        if outlier_count > 0:
            print(f"  [{name}] Removed {outlier_count} outliers "
                  f"({outlier_count/original_count*100:.1f}%) "
                  f"- {original_count} → {len(pcd_clean.points)} points")
        
        # 아웃라이어 인덱스 계산
        all_indices = set(range(original_count))
        inlier_set = set(inlier_indices)
        outlier_indices = np.array(list(all_indices - inlier_set))
        
        return pcd_clean, outlier_indices
    
    def _run_icp_stage(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        initial_transform: np.ndarray,
        distance_threshold: float,
        stage_name: str,
        return_fitness: bool = False
    ):
        """
        단일 ICP 단계를 수행합니다.
        
        Args:
            source: 소스 포인트클라우드
            target: 타겟 포인트클라우드
            initial_transform: 초기 변환 행렬
            distance_threshold: 거리 임계값
            stage_name: 단계 이름 (로깅용)
            return_fitness: fitness 값도 반환할지 여부
            
        Returns:
            변환 행렬 또는 (변환 행렬, fitness) 튜플
        """
        current_transform = initial_transform
        final_fitness = 0.0
        
        for iteration in range(ICPConstants.MAX_ITERATIONS):
            result = o3d.pipelines.registration.registration_icp(
                source, target,
                distance_threshold,
                current_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=ICPConstants.RELATIVE_FITNESS,
                    relative_rmse=ICPConstants.RELATIVE_RMSE,
                    max_iteration=1
                )
            )
            
            if iteration % ICPConstants.VISUALIZATION_INTERVAL == 0:
                print(f"  - ICP iteration {iteration}: fitness = {result.fitness:.6f}")
                
                if self.visualization:
                    source_temp = copy.deepcopy(source)
                    source_temp.transform(result.transformation)
                    self._update_visualization(source_temp, target)
            
            if np.allclose(result.transformation, current_transform, 
                          atol=ICPConstants.CONVERGENCE_TOLERANCE):
                print(f"  - ICP converged (iteration {iteration})")
                break
            
            current_transform = result.transformation
            final_fitness = result.fitness
        
        if return_fitness:
            return current_transform, final_fitness
        return current_transform
    
    def _setup_visualizer(self):
        """시각화 창을 설정합니다."""
        self._visualizer = o3d.visualization.Visualizer()
        self._visualizer.create_window(
            window_name='ICP Registration', 
            width=VisualizationConstants.WINDOW_WIDTH, 
            height=VisualizationConstants.WINDOW_HEIGHT
        )
        
        opt = self._visualizer.get_render_option()
        opt.background_color = np.asarray(VisualizationConstants.BACKGROUND_COLOR)
        opt.point_size = VisualizationConstants.POINT_SIZE
        
        ctr = self._visualizer.get_view_control()
        ctr.set_zoom(VisualizationConstants.CAMERA_ZOOM)
        ctr.set_front(VisualizationConstants.CAMERA_FRONT)
        ctr.set_up(VisualizationConstants.CAMERA_UP)
        
        self._visualizer.poll_events()
        self._visualizer.update_renderer()
        time.sleep(VisualizationConstants.INIT_DELAY)
    
    def _update_visualization(
        self, 
        source: o3d.geometry.PointCloud, 
        target: o3d.geometry.PointCloud
    ):
        """시각화를 업데이트합니다."""
        if self._visualizer is None:
            return
        
        self._visualizer.clear_geometries()
        self._visualizer.add_geometry(source)
        self._visualizer.add_geometry(target)
        
        ctr = self._visualizer.get_view_control()
        ctr.set_zoom(VisualizationConstants.CAMERA_ZOOM)
        ctr.set_front(VisualizationConstants.CAMERA_FRONT)
        ctr.set_up(VisualizationConstants.CAMERA_UP)
        
        self._visualizer.poll_events()
        self._visualizer.update_renderer()
        time.sleep(VisualizationConstants.ANIMATION_DELAY)
    
    def _close_visualizer(self):
        """시각화 창을 닫습니다."""
        if self._visualizer is None:
            return
        
        while True:
            if not self._visualizer.poll_events():
                break
            self._visualizer.update_renderer()
            time.sleep(VisualizationConstants.RENDER_LOOP_DELAY)
        
        self._visualizer = None

