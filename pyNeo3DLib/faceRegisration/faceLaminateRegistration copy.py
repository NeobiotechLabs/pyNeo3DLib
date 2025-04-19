import numpy as np
from pyNeo3DLib.fileLoader.mesh import Mesh
import copy
from scipy.spatial import ConvexHull
import mediapipe as mp
from pyNeo3DLib.visualization.neovis import visualize_meshes
import cv2
import open3d as o3d
import time
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceMeshAnalyzer:
    def __init__(self):
        """MediaPipe Face Mesh 초기화"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        
        # 입술 랜드마크 인덱스
        self.outer_lips = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
                          291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
        self.inner_lips = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
                          308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
        
def normalize_and_flip_coordinates(points, image_size):
    """이미지 좌표를 UV 좌표로 변환 (Y축 반전 포함)"""
    w, h = image_size
    normalized = points.astype(float)
    normalized[:, 0] /= w
    normalized[:, 1] = 1.0 - (normalized[:, 1] / h)
    return normalized

def is_point_in_polygon(point, polygon, epsilon=1e-10):
    """점이 다각형 내부에 있는지 확인 (Ray Casting Algorithm)"""
    x, y = point
    n = len(polygon)
    inside = False

# 경계선 상의 점 처리
    def on_segment(p, q, r):
        if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
            q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
            d = (r[1] - p[1]) * (q[0] - p[0]) - (q[1] - p[1]) * (r[0] - p[0])
            if abs(d) < epsilon:
                return True
        return False

    # 경계선 상의 점 확인
    j = n - 1
    for i in range(n):
        if on_segment(polygon[i], [x, y], polygon[j]):
            return True
        j = i
    
    # Ray casting
    j = n - 1
    for i in range(n):
        if ((polygon[i, 1] > y) != (polygon[j, 1] > y) and
            (x < (polygon[j, 0] - polygon[i, 0]) * (y - polygon[i, 1]) /
            (polygon[j, 1] - polygon[i, 1]) + polygon[i, 0])):
            inside = not inside
        j = i
    
    return inside

def get_lip_landmarks(analyzer, image_path):
    """이미지에서 내부/외부 입술 랜드마크 추출"""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    results = analyzer.face_mesh.process(image_rgb)
    if not results.multi_face_landmarks:
        raise Exception("얼굴을 찾을 수 없습니다.")
    
    landmarks = results.multi_face_landmarks[0]
    
    # 내부 입술 좌표 추출
    inner_points = []
    for idx in analyzer.inner_lips:
        point = landmarks.landmark[idx]
        x, y = int(point.x * w), int(point.y * h)
        inner_points.append([x, y])
    
    # 외부 입술 좌표 추출
    outer_points = []
    for idx in analyzer.outer_lips:
        point = landmarks.landmark[idx]
        x, y = int(point.x * w), int(point.y * h)
        outer_points.append([x, y])
    
    return np.array(inner_points), np.array(outer_points)

def find_lip_regions(mesh, inner_uv_points, outer_uv_points, margin=0.0005):
    """내부와 외부 입술 영역 검출"""
    start_time = time.time()
    logger.info("입술 영역 검출 시작")
    
    # 메시 데이터 준비
    prep_start = time.time()
    triangles = np.asarray(mesh.triangles)
    triangle_uvs = np.asarray(mesh.triangle_uvs)
    logger.info(f"메시 데이터 준비 완료: {time.time() - prep_start:.4f}초")
    
    # 경계 상자 계산 (입술 영역의 최소/최대 좌표)
    inner_bbox_start = time.time()
    inner_min = np.min(inner_uv_points, axis=0) - margin
    inner_max = np.max(inner_uv_points, axis=0) + margin
    
    outer_min = np.min(outer_uv_points, axis=0) - margin
    outer_max = np.max(outer_uv_points, axis=0) + margin
    logger.info(f"경계 상자 계산 완료: {time.time() - inner_bbox_start:.4f}초")
    
    inner_triangles = set()
    outer_triangles = set()
    
    # 삼각형 처리
    triangle_start = time.time()
    total_triangles = len(triangle_uvs) // 3
    logger.info(f"총 {total_triangles}개의 삼각형 처리 시작")
    
    # 벡터화된 연산을 위한 준비
    # 모든 삼각형의 중심점을 한 번에 계산
    centers = np.zeros((total_triangles, 2))
    for i in range(total_triangles):
        idx = i * 3
        centers[i] = np.mean(triangle_uvs[idx:idx+3], axis=0)
    
    # 경계 상자 필터링 (빠른 사전 필터링)
    inner_mask = np.all((centers >= inner_min) & (centers <= inner_max), axis=1)
    outer_mask = np.all((centers >= outer_min) & (centers <= outer_max), axis=1)
    
    # 경계 상자 내 삼각형만 상세 검사
    inner_candidates = np.where(inner_mask)[0]
    outer_candidates = np.where(outer_mask)[0]
    
    logger.info(f"경계 상자 필터링 후 내부 후보: {len(inner_candidates)}개, 외부 후보: {len(outer_candidates)}개")
    
    # 내부 입술 영역 체크 (병렬 처리 가능)
    for idx in inner_candidates:
        center = centers[idx]
        if is_point_in_polygon(center, inner_uv_points):
            inner_triangles.add(idx)
    
    # 외부 입술 영역 체크 (병렬 처리 가능)
    for idx in outer_candidates:
        center = centers[idx]
        if is_point_in_polygon(center, outer_uv_points):
            outer_triangles.add(idx)
    
    logger.info(f"삼각형 처리 완료: {time.time() - triangle_start:.4f}초")
    logger.info(f"내부 입술 삼각형: {len(inner_triangles)}개, 외부 입술 삼각형: {len(outer_triangles)}개")
    
    # 버텍스 수집
    vertex_start = time.time()
    inner_vertices = set()
    outer_vertices = set()
    for triangle_idx in inner_triangles:
        inner_vertices.update(triangles[triangle_idx])
    for triangle_idx in outer_triangles:
        outer_vertices.update(triangles[triangle_idx])
    
    logger.info(f"버텍스 수집 완료: {time.time() - vertex_start:.4f}초")
    logger.info(f"내부 입술 버텍스: {len(inner_vertices)}개, 외부 입술 버텍스: {len(outer_vertices)}개")
    
    total_time = time.time() - start_time
    logger.info(f"입술 영역 검출 완료: 총 {total_time:.4f}초 소요")
    
    return list(inner_vertices), list(outer_vertices)

def visualize_both_lip_regions(mesh, inner_vertices, outer_vertices):
    """내부(빨강)와 외부(파랑) 입술 영역 시각화"""
    mesh_copy = o3d.geometry.TriangleMesh(mesh)
    
    # 색상 초기화 (회색)
    vertices = np.asarray(mesh.vertices)
    colors = np.ones((len(vertices), 3)) * 0.7
    
    # 영역별 색상 지정
    colors[outer_vertices] = [0, 0, 1]  # 파랑
    colors[inner_vertices] = [1, 0, 0]  # 빨강
    
    mesh_copy.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh_copy.compute_vertex_normals()
    
    # 시각화
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh_copy)
    
    # 렌더링 옵션 설정
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.8, 0.8, 0.8])
    # 음영 모드 설정 (버전에 따라 다른 옵션 사용)
    
    opt.light_on = True
    
    # 카메라 설정
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, 1])
    ctr.set_up([0, 1, 0])
    ctr.set_zoom(0.7)
    
    vis.run()
    vis.destroy_window()


class FaceLaminateRegistration:
    def __init__(self, face_path, laminate_path, visualization=False):
        self.face_smile_path = face_path
        self.laminate_path = laminate_path
        self.visualization = visualization
        # 변환 행렬 초기화 (4x4 행렬로 설정)
        self.transform_matrix = np.eye(4)

        self.__load_models()

    def __load_models(self):
        self.face_smile_mesh = Mesh.from_file(self.face_smile_path)
        self.laminate_mesh = Mesh.from_file(self.laminate_path)

        # z축 기준 180도 회전 행렬 생성
        rotation_matrix = np.array([
            [-1, 0, 0],  # x축 반전
            [0, -1, 0],  # y축 반전 
            [0, 0, 1]    # z축 유지
        ])
        
        # 4x4 변환 행렬로 확장
        rotation_matrix_4x4 = np.eye(4)
        rotation_matrix_4x4[0:3, 0:3] = rotation_matrix
        
        # transform_matrix에 회전 적용
        self.transform_matrix = np.dot(rotation_matrix_4x4, self.transform_matrix)
        
        # 정점들을 회전
        vertices = np.array(self.face_smile_mesh.vertices)
        self.face_smile_mesh.vertices = np.dot(vertices, rotation_matrix)
        
        print(self.face_smile_mesh.faces)
        print(self.laminate_mesh.faces)
        return self.face_smile_mesh, self.laminate_mesh
    
    def analyze_face_landmarks(self):
        analyzer = FaceMeshAnalyzer()

        try:
            face_smile_image_path = self.face_smile_path.replace(".obj", ".png")
            # 랜드마크 추출
            inner_points, outer_points = get_lip_landmarks(analyzer, face_smile_image_path)
            print(inner_points)

            image = cv2.imread(face_smile_image_path)
            h, w = image.shape[:2]
            inner_uv_points = normalize_and_flip_coordinates(inner_points, (w, h))
            outer_uv_points = normalize_and_flip_coordinates(outer_points, (w, h))

            o3d_mesh = o3d.io.read_triangle_mesh(self.face_smile_path)
            
            # 초기 상태 시각화
            initial_mesh = Mesh()
            initial_mesh.vertices = np.asarray(o3d_mesh.vertices)
            initial_mesh.faces = np.asarray(o3d_mesh.triangles)
            visualize_meshes([initial_mesh, self.laminate_mesh], ["Initial Face", "Laminate"], 
                            title="Initial state")
            
            # 180도 회전 적용
            vertices = np.asarray(o3d_mesh.vertices)
            rotation_matrix = np.array([
                [-1, 0, 0],  # x축 반전
                [0, -1, 0],  # y축 반전 
                [0, 0, 1]    # z축 유지
            ])
            rotated_vertices = np.dot(vertices, rotation_matrix)
            o3d_mesh.vertices = o3d.utility.Vector3dVector(rotated_vertices)
            
            # 회전 후 상태 시각화
            rotated_mesh = Mesh()
            rotated_mesh.vertices = np.asarray(o3d_mesh.vertices)
            rotated_mesh.faces = np.asarray(o3d_mesh.triangles)
            visualize_meshes([rotated_mesh, self.laminate_mesh], ["Rotated Face", "Laminate"], 
                            title="After rotation")
            
            # 입술 영역 검출 (회전된 메시에서)
            inner_vertices, outer_vertices = find_lip_regions(
                o3d_mesh, inner_uv_points, outer_uv_points
            )

            print(f"내부 입술 영역 버텍스 수: {len(inner_vertices)}")
            print(f"외부 입술 영역 버텍스 수: {len(outer_vertices)}")

            # Open3D 메시로 입술 영역 시각화 (회전된 메시에서)
            visualize_both_lip_regions(o3d_mesh, inner_vertices, outer_vertices)
            
            # 내부 입술 버텍스와 라미네이트 메시 정렬
            self.align_with_laminate_o3d(o3d_mesh, inner_vertices)
            
            # 최종 변환된 메시 시각화
            final_vertices = np.asarray(o3d_mesh.vertices)
            final_vertices_homogeneous = np.hstack((final_vertices, np.ones((final_vertices.shape[0], 1))))
            transformed_vertices = np.dot(final_vertices_homogeneous, self.transform_matrix.T)[:, :3]
            
            final_mesh = Mesh()
            final_mesh.vertices = transformed_vertices
            final_mesh.faces = np.asarray(o3d_mesh.triangles)
            
            visualize_meshes([final_mesh, self.laminate_mesh], ["Transformed Face", "Laminate"], 
                            title="Final transform")
            
        except Exception as e:
            print(e)
            logger.error(f"얼굴 랜드마크 분석 중 오류 발생: {e}")
        
        pass
        
    def align_with_laminate_o3d(self, o3d_mesh, inner_vertices):
        """Open3D 메시를 사용하여 내부 입술 버텍스의 무게 중심과 라미네이트 메시를 맞춤"""
        start_time = time.time()
        logger.info("라미네이트 메시와 내부 입술 정렬 시작 (Open3D)")
        
        # 내부 입술 버텍스의 무게 중심 계산 (Open3D 메시 사용)
        inner_vertices_coords = np.asarray(o3d_mesh.vertices)[inner_vertices]
        inner_centroid = np.mean(inner_vertices_coords, axis=0)
        logger.info(f"내부 입술 무게 중심 (Open3D): {inner_centroid}")
        
        # 입술 영역의 법선 벡터 계산
        o3d_mesh.compute_vertex_normals()
        normals = np.asarray(o3d_mesh.vertex_normals)
        lip_normal = np.mean(normals[inner_vertices], axis=0)
        lip_normal = lip_normal / np.linalg.norm(lip_normal)
        logger.info(f"입술 영역 평균 법선 벡터: {lip_normal}")
        
        # 입술이 +y 방향을 향하도록 회전
        target_direction = np.array([0, 1, 0])  # +y 방향
        rotation_axis = np.cross(lip_normal, target_direction)
        rotation_angle = np.arccos(np.dot(lip_normal, target_direction))
        
        # Rodrigues' rotation formula를 사용한 회전 행렬 생성
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                     [rotation_axis[2], 0, -rotation_axis[0]],
                     [-rotation_axis[1], rotation_axis[0], 0]])
        rotation_matrix = (np.eye(3) + np.sin(rotation_angle) * K + 
                         (1 - np.cos(rotation_angle)) * np.dot(K, K))
        
        # 4x4 회전 행렬로 확장
        rotation_matrix_4x4 = np.eye(4)
        rotation_matrix_4x4[:3, :3] = rotation_matrix
        
        # transform_matrix에 회전 적용
        self.transform_matrix = np.dot(rotation_matrix_4x4, self.transform_matrix)
        
        # 라미네이트 메시의 무게 중심 계산
        laminate_vertices = np.array(self.laminate_mesh.vertices)
        laminate_centroid = np.mean(laminate_vertices, axis=0)
        logger.info(f"라미네이트 무게 중심: {laminate_centroid}")
        
        # 이동 벡터 계산 (라미네이트는 고정하고 얼굴 메시를 이동)
        translation_vector = laminate_centroid - inner_centroid
        logger.info(f"이동 벡터: {translation_vector}")
        
        # 이동 행렬 생성
        translation_matrix = np.eye(4)
        translation_matrix[0:3, 3] = translation_vector
        
        # transform_matrix에 이동 적용 (누적)
        self.transform_matrix = np.dot(translation_matrix, self.transform_matrix)
        
        # ICP를 위한 포인트 클라우드 준비
        # 현재까지의 변환을 적용한 입술 영역 정점
        vertices = np.asarray(o3d_mesh.vertices)
        vertices_homogeneous = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
        transformed_vertices = np.dot(vertices_homogeneous, self.transform_matrix.T)[:, :3]
        
        # 입술 영역 포인트 클라우드 생성
        lip_pcd = o3d.geometry.PointCloud()
        lip_pcd.points = o3d.utility.Vector3dVector(transformed_vertices[inner_vertices])
        lip_pcd.estimate_normals()
        
        # 라미네이트 포인트 클라우드 생성
        laminate_pcd = o3d.geometry.PointCloud()
        laminate_pcd.points = o3d.utility.Vector3dVector(laminate_vertices)
        laminate_pcd.estimate_normals()
        
        # ICP 실행
        result = o3d.pipelines.registration.registration_icp(
            lip_pcd, laminate_pcd, 
            max_correspondence_distance=5.0,  # 대응점 최대 거리를 더 크게 설정
            init=np.eye(4),
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )
        
        # ICP 결과를 transform_matrix에 적용
        self.transform_matrix = np.dot(result.transformation, self.transform_matrix)
        
        logger.info(f"라미네이트 메시와 내부 입술 정렬 완료: {time.time() - start_time:.4f}초")
        logger.info(f"ICP fitness: {result.fitness}")
        logger.info(f"ICP RMSE: {result.inlier_rmse}")
        
        return self.transform_matrix

    def run_registration(self):
        visualize_meshes([self.face_smile_mesh, self.laminate_mesh], ["Face", "Laminate"], title="Face and Laminate")
        
        self.analyze_face_landmarks()



if __name__ == "__main__":
    face_laminate_registration = FaceLaminateRegistration("../../example/data/FaceScan/Smile/Smile.obj", "../../example/data/smile_arch_half.stl", visualization=True)
    face_laminate_registration.run_registration()