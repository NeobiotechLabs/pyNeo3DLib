"""
메쉬 형식 변환 모듈

이 모듈은 다양한 메쉬 형식 간의 변환을 담당합니다.
단일 책임 원칙(SRP)에 따라 변환 로직만을 캡슐화합니다.
"""
import numpy as np
import pyvista as pv
import open3d as o3d

from pyNeo3DLib.fileLoader.mesh import Mesh
from pyNeo3DLib.faceRegisration.constants import PointCloudConstants


class MeshConverter:
    """
    메쉬 형식 변환을 담당하는 클래스.
    
    단일 책임: 다양한 메쉬 형식 간의 변환
    
    지원하는 변환:
    - Mesh ↔ PyVista PolyData
    - Mesh ↔ Open3D TriangleMesh
    - Mesh → Open3D PointCloud
    """
    
    @staticmethod
    def mesh_to_pyvista(mesh: Mesh) -> pv.PolyData:
        """
        Mesh 객체를 PyVista PolyData로 변환합니다.
        
        Args:
            mesh: 변환할 Mesh 객체
            
        Returns:
            pv.PolyData: PyVista PolyData 객체
        """
        if isinstance(mesh, pv.PolyData):
            return mesh
        
        faces_pv = np.hstack([[3, *face] for face in mesh.faces])
        pv_mesh = pv.PolyData(mesh.vertices, faces_pv)
        
        # 노말 계산
        pv_mesh.compute_normals(inplace=True)
        
        return pv_mesh
    
    @staticmethod
    def pyvista_to_mesh(pv_mesh: pv.PolyData) -> Mesh:
        """
        PyVista PolyData를 Mesh 객체로 변환합니다.
        
        Args:
            pv_mesh: PyVista PolyData 객체
            
        Returns:
            Mesh: 변환된 Mesh 객체
        """
        mesh = Mesh()
        mesh.vertices = np.array(pv_mesh.points)
        
        # faces 변환 (PyVista는 [n, v0, v1, v2, ...] 형식)
        if len(pv_mesh.faces) > 0:
            faces_pv = pv_mesh.faces.reshape(-1, 4)[:, 1:4]
            mesh.faces = np.array(faces_pv)
        else:
            mesh.faces = np.array([]).reshape(0, 3)
        
        # 노말 계산
        pv_mesh.compute_normals(inplace=True)
        if pv_mesh.point_normals is not None:
            mesh.normals = np.array(pv_mesh.point_normals)
        
        return mesh
    
    @staticmethod
    def mesh_to_open3d(mesh: Mesh) -> o3d.geometry.TriangleMesh:
        """
        Mesh 객체를 Open3D TriangleMesh로 변환합니다.
        
        Args:
            mesh: 변환할 Mesh 객체
            
        Returns:
            o3d.geometry.TriangleMesh: Open3D TriangleMesh 객체
        """
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces)
        mesh_o3d.compute_vertex_normals()
        
        return mesh_o3d
    
    @staticmethod
    def open3d_to_mesh(mesh_o3d: o3d.geometry.TriangleMesh) -> Mesh:
        """
        Open3D TriangleMesh를 Mesh 객체로 변환합니다.
        
        Args:
            mesh_o3d: Open3D TriangleMesh 객체
            
        Returns:
            Mesh: 변환된 Mesh 객체
        """
        mesh = Mesh()
        mesh.vertices = np.asarray(mesh_o3d.vertices)
        mesh.faces = np.asarray(mesh_o3d.triangles)
        
        mesh_o3d.compute_vertex_normals()
        mesh.normals = np.asarray(mesh_o3d.vertex_normals)
        
        return mesh
    
    @staticmethod
    def mesh_to_pointcloud(
        mesh: Mesh, 
        downsample: bool = True,
        remove_outliers: bool = True
    ) -> o3d.geometry.PointCloud:
        """
        Mesh 객체를 Open3D PointCloud로 변환합니다.
        
        Args:
            mesh: 변환할 Mesh 객체
            downsample: 다운샘플링 여부
            remove_outliers: 이상치 제거 여부
            
        Returns:
            o3d.geometry.PointCloud: Open3D PointCloud 객체
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(mesh.vertices)
        
        # 노말 벡터 처리
        if mesh.normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(mesh.normals)
        else:
            # 임시 메쉬로부터 노말 계산
            temp_mesh = o3d.geometry.TriangleMesh()
            temp_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
            temp_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
            temp_mesh.compute_vertex_normals()
            pcd.normals = temp_mesh.vertex_normals
        
        # 노말 방향 추정 및 일관성 확인
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=PointCloudConstants.NORMAL_ESTIMATION_RADIUS, 
                max_nn=PointCloudConstants.NORMAL_ESTIMATION_MAX_NN
            )
        )
        pcd.orient_normals_consistent_tangent_plane(k=PointCloudConstants.ORIENT_NORMALS_K)
        
        # 이상치 제거
        if remove_outliers:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        # 다운샘플링
        if downsample:
            pcd = pcd.uniform_down_sample(
                every_k_points=PointCloudConstants.DOWNSAMPLE_EVERY_K_POINTS
            )
        
        return pcd

