import open3d as o3d
import numpy as np
import trimesh
from .constants import MeshConversionConfig


class MeshConverter:
    """메쉬와 포인트클라우드 간 변환을 담당하는 클래스"""
    
    def __init__(self, 
                 default_sample_points: int = MeshConversionConfig.DEFAULT_SAMPLE_POINTS, 
                 random_seed: int = MeshConversionConfig.DEFAULT_RANDOM_SEED):
        """
        Args:
            default_sample_points (int): 기본 샘플링 포인트 수
            random_seed (int): 랜덤 시드 (기본값: 42, 결정론적 결과를 위해 고정값 사용)
                              None으로 설정하면 랜덤 시드를 설정하지 않음
        """
        self.default_sample_points = default_sample_points
        self.random_seed = random_seed
    
    
    def mesh_to_pointcloud(self, mesh: o3d.geometry.TriangleMesh, 
                          sample_points: int = None) -> o3d.geometry.PointCloud:
        """
        메쉬를 입력받아 o3d.geometry.PointCloud 타입의 포인트 클라우드를 생성
        
        Args:
            mesh (o3d.geometry.TriangleMesh): 입력 메쉬
            sample_points (int, optional): 샘플링할 포인트 수
            
        Returns:
            o3d.geometry.PointCloud: 생성된 포인트 클라우드
        """
        if sample_points is None:
            sample_points = self.default_sample_points
        
        # 랜덤 시드 설정 (매번 동일한 샘플링 결과를 얻기 위해)
        if self.random_seed is not None:
            o3d.utility.random.seed(self.random_seed)
            
        # 메쉬에서 균등하게 포인트 샘플링하여 포인트 클라우드 생성
        point_cloud = mesh.sample_points_uniformly(number_of_points=sample_points)
        
        return point_cloud
    
    def o3d_to_trimesh(self, o3d_mesh: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
        """
        Open3D 메쉬를 Trimesh로 변환
        
        Args:
            o3d_mesh (o3d.geometry.TriangleMesh): Open3D 메쉬
            
        Returns:
            trimesh.Trimesh: 변환된 Trimesh 메쉬
        """
        vertices = np.asarray(o3d_mesh.vertices)
        faces = np.asarray(o3d_mesh.triangles)
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    
    def trimesh_to_o3d(self, trimesh_mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
        """
        Trimesh를 Open3D 메쉬로 변환
        
        Args:
            trimesh_mesh (trimesh.Trimesh): Trimesh 메쉬
            
        Returns:
            o3d.geometry.TriangleMesh: 변환된 Open3D 메쉬
        """
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(trimesh_mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces)
        return o3d_mesh

