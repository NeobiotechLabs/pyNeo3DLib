"""
CBCT-FaceScan 정합 파이프라인

설계 원칙:
- 모든 포인트 클라우드는 o3d.geometry.PointCloud 타입으로 통일
- 각 변환 함수: (입력 pcd) → (변환된 pcd, 변환 행렬) 패턴
- 단일 책임 원칙: 파이프라인은 각 모듈을 조율하는 역할만 담당

모듈 구조:
- utils.py: 유틸리티 함수 (변환, 포인트 클라우드 조작)
- cbct_processor.py: CBCT 데이터 처리 (표면 추출, 코 중심 추정)
- alignment_executor.py: 정합 실행 (초기 정렬, ICP)
- alignment_visualizer.py: 시각화
- result_types.py: 결과 데이터 타입
- transform_manager.py: 변환 행렬 관리

리팩토링 내역 (2026-01-30):
- TransformManager 도입으로 변환 행렬 관리 일원화
- run() 메서드를 단계별 메서드로 분해하여 가독성 향상
- 시각화 로직을 별도 메서드로 분리
- RefinementResult 타입 추가
- 에러 처리 및 검증 로직 강화
"""
import os
from typing import Optional, Tuple
import numpy as np
import open3d as o3d
import random

from ...config import AlignmentConfig
from ...utils import apply_transform, compute_translation_matrix
from ...processing import CBCTProcessor, GeometryProcessor
from ...core.alignment_executor import AlignmentExecutor
from ...visualization import AlignmentVisualizer
from ...registration import SurfaceRotationOptimizer, TransformManager
from ...types import (
    PipelineResult,
    FaceScanProcessResult,
    ICPAlignmentResult,
    RefinementResult,
    AlignmentStepResult,
)


class CBCTFaceScanAlignmentPipeline:
    """
    CBCT-FaceScan 정합 파이프라인 클래스 (Orchestrator)
    
    역할:
    - 각 처리 모듈의 조율
    - 파이프라인 단계 실행 순서 관리
    - 최종 결과 집계
    
    위임 대상:
    - CBCTProcessor: CBCT 데이터 처리
    - AlignmentExecutor: 정합 실행
    - AlignmentVisualizer: 시각화
    - GeometryProcessor: 메쉬/포인트 클라우드 처리
    """
    
    def __init__(self, config: Optional[AlignmentConfig] = None, random_seed: Optional[int] = None, visualize: bool = False, verbose: bool = True):
        """
        Args:
            config: 정합 설정 (None일 경우 기본값 사용)
            random_seed: 랜덤 시드 (재현 가능한 결과를 위해 설정, None이면 랜덤)
            visualize: 시각화 활성화 여부 (True면 단계별 시각화 표시)
            verbose: 상세 출력 여부 (True면 진행 상황 출력)
        """
        self.config = config if config is not None else AlignmentConfig()
        self.random_seed = random_seed
        self.visualize = visualize  # 시각화 설정 저장
        self.verbose = verbose      # Verbose 설정 저장
        
        # 랜덤 시드 설정 (재현 가능한 결과를 위해)
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
            o3d.utility.random.seed(random_seed)
        
        # 각 모듈 초기화
        self.cbct_processor = CBCTProcessor(self.config)
        self.alignment_executor = AlignmentExecutor(self.config)
        self.visualizer = AlignmentVisualizer(self.config.visualization, visualize, verbose)
        self.geometry_processor = GeometryProcessor()
        self.rotation_optimizer = SurfaceRotationOptimizer(
            visualizer=self.visualizer,
            geometry_processor=self.geometry_processor
        )
    
    # ==========================================================================
    # 유틸리티 메서드
    # ==========================================================================
    
    @staticmethod
    def _is_valid_transform(transform: np.ndarray) -> bool:
        """
        변환 행렬 유효성 검증
        
        Args:
            transform: 4x4 변환 행렬
        
        Returns:
            bool: 유효한 변환 행렬이면 True
        """
        if transform.shape != (4, 4):
            return False
        
        # 마지막 행이 [0, 0, 0, 1]인지 확인
        if not np.allclose(transform[3, :], [0, 0, 0, 1]):
            return False
        
        # 회전 부분의 행렬식이 1에 가까운지 확인 (직교 행렬)
        rotation = transform[:3, :3]
        det = np.linalg.det(rotation)
        if not np.isclose(det, 1.0, atol=0.1):
            return False
        
        return True
    
    @staticmethod
    def _validate_point_cloud(pcd: o3d.geometry.PointCloud, name: str, min_points: int = 100):
        """
        포인트 클라우드 검증
        
        Args:
            pcd: 검증할 포인트 클라우드
            name: 포인트 클라우드 이름 (에러 메시지용)
            min_points: 최소 포인트 개수
        
        Raises:
            ValueError: 포인트 클라우드가 유효하지 않을 경우
        """
        if pcd is None:
            raise ValueError(f"{name}: 포인트 클라우드가 None입니다.")
        
        if not pcd.has_points():
            raise ValueError(f"{name}: 포인트 클라우드가 비어있습니다.")
        
        num_points = len(pcd.points)
        if num_points < min_points:
            raise ValueError(
                f"{name}: 포인트 개수가 너무 적습니다. "
                f"(현재: {num_points}, 최소: {min_points})"
            )
    
    
    # ==========================================================================
    # Step 5: FaceScan 메쉬 로드 및 처리
    # ==========================================================================
    
    def _load_and_process_facescan(
        self,
        facescan_path: str,
        transform_matrix: np.ndarray,
        cbct_pcd: o3d.geometry.PointCloud,
        verbose: bool = True
    ) -> FaceScanProcessResult:
        """
        FaceScan 로드, 변환, 영역 필터링
        
        Args:
            facescan_path: FaceScan 파일 경로
            transform_matrix: 적용할 변환 행렬
            cbct_pcd: CBCT 포인트 클라우드 (영역 크기 기준)
            verbose: 상세 출력 여부
        
        Returns:
            FaceScanProcessResult: 처리 결과
        """
        if verbose:
            print("\n[Step 5] FaceScan 로드 및 변환")
            print("-" * 50)
        
        # 메쉬 로드 및 변환 적용
        mesh = self.geometry_processor.load_mesh(
            facescan_path,
            transform_matrix=transform_matrix,
            verbose=verbose
        )
        
        # 포인트 클라우드로 변환 및 코 끝 추출
        sampling_cfg = self.config.mesh_sampling
        pcd, nose_point = self.geometry_processor.extract_top_y_points(
            mesh,
            num_samples=sampling_cfg.num_samples,
            verbose=verbose
        )
        
        if verbose:
            print("\n[Step 6] FaceScan 영역 필터링")
            print("-" * 50)
        
        # CBCT 바운딩 박스 크기 계산
        bbox_extent = self.geometry_processor.calculate_bbox_extent(cbct_pcd, verbose=verbose)
        
        # 영역 필터링
        pcd_filtered = self.geometry_processor.filter_region_by_bbox(
            pcd,
            nose_point,
            bbox_extent,
            verbose=verbose
        )
        
        return FaceScanProcessResult(
            mesh=mesh,
            pcd=pcd,
            pcd_filtered=pcd_filtered,
            nose_point=nose_point,
            facescan_transform=transform_matrix
        )
    
    # ==========================================================================
    # Step 9: SDF 기반 표면 정제 (Z축 회전 탐색)
    # ==========================================================================
    
    def _refine_with_surface_rotation_search(
        self,
        pcd_cbct_full: o3d.geometry.PointCloud,
        facescan_mesh: o3d.geometry.TriangleMesh,
        facescan_nose_point: np.ndarray,
        distance_threshold: float = 5.0,
        rotation_range: Tuple[float, float] = (-15, 15),
        rotation_step: float = 1.0,
        downsample_voxel_size: float = 2.0,
        visualize: bool = False,
        verbose: bool = True
    ) -> Tuple[np.ndarray, float, float]:
        """
        SDF 기반 표면 필터링 + Z축 회전 탐색으로 최적 정합 찾기
        
        이 메서드는 SurfaceRotationOptimizer 모듈에 위임합니다.
        
        Args:
            pcd_cbct_full: CBCT 전체 포인트 클라우드
            facescan_mesh: FaceScan 메쉬
            facescan_nose_point: FaceScan 코 중심점 (회전 중심)
            distance_threshold: 표면 필터링 거리 임계값 (mm)
            rotation_range: Z축 회전 탐색 범위 (도 단위)
            rotation_step: 회전 탐색 간격 (도 단위)
            downsample_voxel_size: 다운샘플링 복셀 크기 (mm, 0이면 다운샘플링 안 함)
            visualize: 시각화 여부
            verbose: 상세 출력 여부
        
        Returns:
            Tuple[transform_matrix, best_angle, best_rmse]:
                - transform_matrix: 최적 회전 변환 행렬 (4x4)
                - best_angle: 최적 회전 각도 (도)
                - best_rmse: 최소 RMSE 값
        """
        return self.rotation_optimizer.optimize_rotation(
            pcd_cbct_full=pcd_cbct_full,
            facescan_mesh=facescan_mesh,
            facescan_nose_point=facescan_nose_point,
            distance_threshold=distance_threshold,
            rotation_range=rotation_range,
            rotation_step=rotation_step,
            downsample_voxel_size=downsample_voxel_size,
            visualize=visualize,
            verbose=verbose
        )
    
    # ==========================================================================
    # 시각화 단계별 메서드
    # ==========================================================================
    
    def _visualize_nose_points_if_needed(
        self,
        pcd_cbct: o3d.geometry.PointCloud,
        pcd_facescan: o3d.geometry.PointCloud,
        nose_cbct: np.ndarray,
        nose_facescan: np.ndarray,
        visualize: bool
    ):
        """코 정점 포인트 비교 시각화 (필요시)"""
        if visualize:
            self.visualizer.visualize_nose_points(
                pcd_cbct, pcd_facescan,
                nose_cbct, nose_facescan,
                "0. 코 정점 포인트 비교 (주황:CBCT 코, 하늘:FaceScan 코)"
            )
    
    def _visualize_initial_alignment_if_needed(
        self,
        pcd_cbct: o3d.geometry.PointCloud,
        pcd_facescan: o3d.geometry.PointCloud,
        visualize: bool
    ):
        """초기 정렬 결과 시각화 (필요시)"""
        if visualize:
            self.visualizer.visualize_alignment(
                pcd_cbct, pcd_facescan,
                "2. 초기 정렬 후 (빨강:CBCT, 초록:FaceScan)"
            )
    
    def _visualize_icp_alignment_if_needed(
        self,
        pcd_cbct: o3d.geometry.PointCloud,
        pcd_facescan: o3d.geometry.PointCloud,
        method: str,
        visualize: bool
    ):
        """ICP 정합 결과 시각화 (필요시)"""
        if visualize:
            self.visualizer.visualize_alignment(
                pcd_cbct, pcd_facescan,
                f"3. ICP 정합 후 (빨강:CBCT, 초록:FaceScan) - {method}"
            )
    
    def _visualize_before_refinement_if_needed(
        self,
        pcd_cbct_full: o3d.geometry.PointCloud,
        facescan_mesh: o3d.geometry.TriangleMesh,
        visualize: bool
    ):
        """정제 전 결과 시각화 (필요시)"""
        if visualize:
            print("\n[중간 시각화] CBCT 전체 볼륨 + FaceScan 메쉬 (정제 전)")
            self.visualizer.visualize_final_result(
                pcd_cbct_full,
                facescan_mesh,
                window_name="정제 전 정합 결과"
            )
    
    def _visualize_final_result_if_needed(
        self,
        pcd_cbct_full: o3d.geometry.PointCloud,
        facescan_mesh: o3d.geometry.TriangleMesh,
        best_angle: float,
        best_rmse: float,
        visualize: bool
    ):
        """최종 결과 시각화 (필요시)"""
        if visualize:
            print("\n[최종 시각화] CBCT 전체 볼륨 + FaceScan 메쉬 (정제 후)")
            self.visualizer.visualize_final_result(
                pcd_cbct_full,
                facescan_mesh,
                window_name=f"최종 정합 결과 (정제 후) - 각도: {best_angle:.1f}°, RMSE: {best_rmse:.3f}mm"
            )
    
    # ==========================================================================
    # 파이프라인 단계별 실행 메서드
    # ==========================================================================
    
    def _execute_cbct_processing(
        self,
        dicom_folder: str,
        verbose: bool
    ) -> Tuple[np.ndarray, o3d.geometry.PointCloud, o3d.geometry.PointCloud, np.ndarray]:
        """
        CBCT 데이터 처리 단계
        
        Args:
            dicom_folder: DICOM 폴더 경로
            verbose: 상세 출력 여부
        
        Returns:
            Tuple[nose_center, pcd_standard, pcd_full, transform_matrix]:
                - nose_center: 코 중심 (표준 좌표계)
                - pcd_standard: 필터링된 포인트 클라우드 (표준 좌표계)
                - pcd_full: 전체 포인트 클라우드 (표준 좌표계)
                - transform_matrix: RAI → 표준 변환 행렬
        
        Raises:
            FileNotFoundError: DICOM 폴더가 없을 경우
            ValueError: CBCT 처리 중 오류 발생 시
        """
        if verbose:
            print("\n[Step 2-4] CBCT 데이터 처리")
        
        # DICOM 폴더 존재 여부 확인
        if not os.path.exists(dicom_folder):
            raise FileNotFoundError(f"DICOM 폴더를 찾을 수 없습니다: {dicom_folder}")
        
        if not os.path.isdir(dicom_folder):
            raise ValueError(f"유효한 폴더가 아닙니다: {dicom_folder}")
        
        try:
            nose_center, pcd_standard, pcd_full, transform_matrix = (
                self.cbct_processor.process(dicom_folder, verbose=verbose)
            )
            
            # 결과 검증
            self._validate_point_cloud(pcd_standard, "CBCT 표준 포인트 클라우드")
            self._validate_point_cloud(pcd_full, "CBCT 전체 포인트 클라우드")
            
            if not self._is_valid_transform(transform_matrix):
                raise ValueError("유효하지 않은 CBCT 변환 행렬")
            
            return nose_center, pcd_standard, pcd_full, transform_matrix
            
        except Exception as e:
            raise ValueError(f"CBCT 처리 중 오류 발생: {e}") from e
    
    def _execute_facescan_processing(
        self,
        facescan_path: str,
        facescan_transform: np.ndarray,
        cbct_pcd: o3d.geometry.PointCloud,
        verbose: bool
    ) -> FaceScanProcessResult:
        """
        FaceScan 데이터 처리 단계
        
        Args:
            facescan_path: FaceScan 파일 경로
            facescan_transform: FaceScan → SmileArch 변환 행렬
            cbct_pcd: CBCT 포인트 클라우드 (영역 크기 기준)
            verbose: 상세 출력 여부
        
        Returns:
            FaceScanProcessResult: FaceScan 처리 결과
        
        Raises:
            FileNotFoundError: FaceScan 파일이 없을 경우
            ValueError: FaceScan 처리 중 오류 발생 시
        """
        if verbose:
            print("\n[Step 5-6] FaceScan 로드 및 처리")
        
        # FaceScan 파일 존재 여부 확인
        if not os.path.exists(facescan_path):
            raise FileNotFoundError(f"FaceScan 파일을 찾을 수 없습니다: {facescan_path}")
        
        try:
            result = self._load_and_process_facescan(
                facescan_path,
                facescan_transform,
                cbct_pcd,
                verbose
            )
            
            # 결과 검증
            self._validate_point_cloud(result.pcd, "FaceScan 포인트 클라우드")
            self._validate_point_cloud(result.pcd_filtered, "FaceScan 필터링 포인트 클라우드")
            
            return result
            
        except Exception as e:
            raise ValueError(f"FaceScan 처리 중 오류 발생: {e}") from e
    
    def _execute_alignment(
        self,
        pcd_cbct: o3d.geometry.PointCloud,
        pcd_facescan: o3d.geometry.PointCloud,
        verbose: bool
    ) -> Tuple[AlignmentStepResult, ICPAlignmentResult]:
        """
        정합 단계 (초기 정렬 + ICP)
        
        Args:
            pcd_cbct: CBCT 포인트 클라우드
            pcd_facescan: FaceScan 포인트 클라우드
            verbose: 상세 출력 여부
        
        Returns:
            Tuple[initial_result, icp_result]:
                - initial_result: 초기 정렬 결과
                - icp_result: ICP 정합 결과
        
        Raises:
            ValueError: 정합 중 오류 발생 시
        """
        if verbose:
            print("\n[Step 7-8] 정합 실행")
        
        # 입력 검증
        self._validate_point_cloud(pcd_cbct, "CBCT 정합용 포인트 클라우드")
        self._validate_point_cloud(pcd_facescan, "FaceScan 정합용 포인트 클라우드")
        
        try:
            initial_result, icp_result = self.alignment_executor.execute(
                pcd_cbct, pcd_facescan, verbose=verbose
            )
            
            # 결과 검증
            if icp_result.fitness < 0.1:
                print(f"경고: ICP Fitness가 낮습니다 ({icp_result.fitness:.6f}). 정합 품질이 좋지 않을 수 있습니다.")
            
            return initial_result, icp_result
            
        except Exception as e:
            raise ValueError(f"정합 중 오류 발생: {e}") from e
    
    def _execute_refinement(
        self,
        pcd_cbct_full: o3d.geometry.PointCloud,
        facescan_mesh: o3d.geometry.TriangleMesh,
        facescan_nose_point: np.ndarray,
        visualize: bool,
        verbose: bool
    ) -> RefinementResult:
        """
        SDF 기반 표면 정제 단계
        
        Args:
            pcd_cbct_full: CBCT 전체 포인트 클라우드
            facescan_mesh: FaceScan 메쉬
            facescan_nose_point: FaceScan 코 중심점
            visualize: 시각화 여부
            verbose: 상세 출력 여부
        
        Returns:
            RefinementResult: 정제 결과
        
        Raises:
            ValueError: 정제 중 오류 발생 시
        """
        if verbose:
            print("\n[Step 9] SDF 기반 표면 정제 (Z축 회전 탐색)")
        
        # 입력 검증
        self._validate_point_cloud(pcd_cbct_full, "CBCT 전체 포인트 클라우드", min_points=1000)
        
        if facescan_mesh is None or not facescan_mesh.has_triangles():
            raise ValueError("유효한 FaceScan 메쉬가 필요합니다.")
        
        try:
            refinement_transform, best_angle, best_rmse = self._refine_with_surface_rotation_search(
                pcd_cbct_full=pcd_cbct_full,
                facescan_mesh=facescan_mesh,
                facescan_nose_point=facescan_nose_point,
                distance_threshold=1.0,
                rotation_range=(-5, 5),
                rotation_step=0.2,
                visualize=visualize,
                verbose=verbose
            )
            
            # 결과 검증
            if not self._is_valid_transform(refinement_transform):
                raise ValueError("유효하지 않은 정제 변환 행렬")
            
            return RefinementResult(
                transform_matrix=refinement_transform,
                best_angle=best_angle,
                best_rmse=best_rmse
            )
            
        except Exception as e:
            raise ValueError(f"정제 중 오류 발생: {e}") from e
    
    def _build_pipeline_result(
        self,
        transform_manager: TransformManager,
        pcd_cbct_full_refined: o3d.geometry.PointCloud,
        facescan_result: FaceScanProcessResult,
        initial_result: AlignmentStepResult,
        icp_result: ICPAlignmentResult,
        refinement_result: RefinementResult
    ) -> PipelineResult:
        """
        파이프라인 결과 객체 생성
        
        Args:
            transform_manager: 변환 행렬 관리자
            pcd_cbct_full_refined: CBCT 전체 포인트 클라우드 (정제 후)
            facescan_result: FaceScan 처리 결과
            initial_result: 초기 정렬 결과
            icp_result: ICP 정합 결과
            refinement_result: 정제 결과
        
        Returns:
            PipelineResult: 전체 파이프라인 결과
        """
        # 변환 행렬 딕셔너리에 추가 정보 포함
        transforms_dict = transform_manager.to_dict()
        transforms_dict["refinement_angle"] = refinement_result.best_angle
        transforms_dict["refinement_rmse"] = refinement_result.best_rmse
        
        return PipelineResult(
            final_transform=transform_manager.get_final_transform(include_refinement=True),
            cbct_full_final=pcd_cbct_full_refined,
            facescan_process=facescan_result,
            initial_alignment=initial_result,
            icp_alignment=icp_result,
            refinement=refinement_result,
            transforms=transforms_dict
        )
    
    # ==========================================================================
    # 전체 파이프라인 실행
    # ==========================================================================
    
    def run(
        self,
        dicom_folder: str,
        facescan_path: str,
        facescan_laminate_result: np.ndarray
    ) -> np.ndarray:
        """
        전체 정합 파이프라인 실행
        
        Args:
            dicom_folder: DICOM 폴더 경로
            facescan_path: FaceScan 파일 경로
            facescan_laminate_result: FaceScan 정렬 변환 행렬
        
        Returns:
            np.ndarray: 최종 변환 행렬 (4x4) - CBCT RAI 좌표계 → FaceScan 좌표계
        
        Note:
            시각화와 verbose 설정은 __init__에서 설정한 값을 사용합니다.
            실행 중에 변경하려면: pipeline.visualizer.enabled = True/False
        """
        # __init__에서 설정한 값 사용
        visualize = self.visualize
        verbose = self.verbose
        print("=" * 60)
        print("CBCT-FaceScan 정합 파이프라인 시작")
        print("=" * 60)
        
        # 변환 행렬 관리자 초기화
        transform_manager = TransformManager()
        
        # ----------------------------------------------------------------------
        # Step 2-4: CBCT 데이터 처리
        # ----------------------------------------------------------------------
        nose_center_standard, pcd_cbct_standard, pcd_cbct_full_std, rai_transform = (
            self._execute_cbct_processing(dicom_folder, verbose)
        )
        transform_manager.rai_to_standard = rai_transform
        
        # ----------------------------------------------------------------------
        # Step 5-6: FaceScan 데이터 처리
        # ----------------------------------------------------------------------
        facescan_result = self._execute_facescan_processing(
            facescan_path,
            facescan_laminate_result,
            pcd_cbct_standard,
            verbose
        )
        
        # 시각화 1: 코 정점 비교
        self._visualize_nose_points_if_needed(
            pcd_cbct_standard,
            facescan_result.pcd_filtered,
            nose_center_standard,
            facescan_result.nose_point,
            visualize
        )
        
        # ----------------------------------------------------------------------
        # Step 7-8: 정합 실행 (초기 정렬 + ICP)
        # ----------------------------------------------------------------------
        initial_result, icp_result = self._execute_alignment(
            pcd_cbct_standard,
            facescan_result.pcd_filtered,
            verbose
        )
        
        transform_manager.initial_alignment = initial_result.transform_matrix
        transform_manager.icp = icp_result.transform_matrix
        
        # 시각화 2: 초기 정렬 후
        self._visualize_initial_alignment_if_needed(
            initial_result.aligned_pcd,
            facescan_result.pcd_filtered,
            visualize
        )
        
        # 시각화 3: ICP 정합 후
        self._visualize_icp_alignment_if_needed(
            icp_result.aligned_pcd,
            facescan_result.pcd_filtered,
            icp_result.method,
            visualize
        )
        
        # ----------------------------------------------------------------------
        # 중간 결과: 정제 전 전체 볼륨 변환
        # ----------------------------------------------------------------------
        accumulated_transform = transform_manager.get_accumulated_transform(include_refinement=False)
        pcd_cbct_full_aligned = apply_transform(pcd_cbct_full_std, accumulated_transform)
        
        # 시각화 4: 정제 전 결과
        self._visualize_before_refinement_if_needed(
            pcd_cbct_full_aligned,
            facescan_result.mesh,
            visualize
        )
        
        if verbose:
            print("\n" + "=" * 60)
            print("정제 전 변환 행렬 (CBCT RAI → FaceScan 좌표계)")
            print("=" * 60)
            print("적용 순서: RAI→표준 → 초기정렬 → ICP")
            print(transform_manager.get_final_transform(include_refinement=False))
            print("=" * 60)
        
        # ----------------------------------------------------------------------
        # Step 9: SDF 기반 표면 정제
        # ----------------------------------------------------------------------
        refinement_result = self._execute_refinement(
            pcd_cbct_full_aligned,
            facescan_result.mesh,
            facescan_result.nose_point,
            visualize,
            verbose
        )
        
        transform_manager.refinement = refinement_result.transform_matrix
        
        # 전체 볼륨에 정제 변환 적용
        pcd_cbct_full_refined = apply_transform(
            pcd_cbct_full_aligned,
            refinement_result.transform_matrix
        )
        
        # 시각화 5: 최종 결과 (정제 후)
        self._visualize_final_result_if_needed(
            pcd_cbct_full_refined,
            facescan_result.mesh,
            refinement_result.best_angle,
            refinement_result.best_rmse,
            visualize
        )
        
        # ----------------------------------------------------------------------
        # 최종 결과 출력
        # ----------------------------------------------------------------------
        if verbose:
            print("\n" + "=" * 60)
            print("정합 완료 (SDF 기반 Z축 회전 정제 포함)")
            print("=" * 60)
            print(f"최적 회전 각도: {refinement_result.best_angle:+.1f}°")
            print(f"최소 RMSE: {refinement_result.best_rmse:.3f} mm")
            print("=" * 60)
            
            # 변환 행렬 요약 출력
            transform_manager.print_summary(include_refinement=True)
        
        # ----------------------------------------------------------------------
        # 최종 변환 행렬 반환
        # ----------------------------------------------------------------------
        final_transform = transform_manager.get_final_transform(include_refinement=True)
        
        if verbose:
            print("\n" + "=" * 60)
            print("최종 변환 행렬 (CBCT RAI → FaceScan 좌표계)")
            print("=" * 60)
            print(final_transform)
            print("=" * 60)
        
        # 결과를 인스턴스 변수에 저장 (필요시 접근 가능)
        self.results = {
            'final_transform': final_transform,
            'transform_manager': transform_manager,
            'cbct_full_refined': pcd_cbct_full_refined,
            'facescan_result': facescan_result,
            'initial_result': initial_result,
            'icp_result': icp_result,
            'refinement_result': refinement_result,
        }
        
        return final_transform


# ==============================================================================
# 하위 호환성을 위한 유틸리티 함수 re-export
# ==============================================================================
from ...utils import np_to_pcd, pcd_to_np, apply_transform, compute_translation_matrix

__all__ = [
    "CBCTFaceScanAlignmentPipeline",
    "np_to_pcd",
    "pcd_to_np", 
    "apply_transform",
    "compute_translation_matrix",
]


