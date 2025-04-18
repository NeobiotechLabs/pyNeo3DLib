import numpy as np
import struct
from pathlib import Path
import re

class Mesh:
    def __init__(self):
        self.vertices = None
        self.faces = None
        self.normals = None
        self.materials = {}  # MTL 파일에서 읽은 재질 정보
        self.face_materials = None  # 각 면의 재질 인덱스
    
    @classmethod
    def from_file(cls, file_path):
        """파일 확장자에 따라 적절한 메서드로 메시를 로드합니다."""
        path = Path(file_path)
        mesh = cls()
        
        if path.suffix.lower() == '.stl':
            return mesh._read_stl(file_path)
        elif path.suffix.lower() == '.obj':
            return mesh._read_obj(file_path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식입니다: {path.suffix}")
    
    def _read_stl(self, file_path):
        """STL 파일을 읽어옵니다. ASCII와 바이너리 형식 모두 지원합니다."""
        path = Path(file_path)
        vertices = []
        faces = []
        normals = []
        
        try:
            # 파일 형식 확인 (ASCII 또는 바이너리)
            with open(path, 'rb') as f:
                header = f.read(5)
                is_ascii = header.startswith(b'solid')
            
            if is_ascii:
                self._read_stl_ascii(path, vertices, faces, normals)
            else:
                self._read_stl_binary(path, vertices, faces, normals)
            
            if not vertices:
                raise ValueError("파일에서 정점을 찾을 수 없습니다.")
            
            self.vertices = np.array(vertices)
            self.faces = np.array(faces)
            self.normals = np.array(normals)
            
            return self
            
        except Exception as e:
            raise ValueError(f"STL 파일 읽기 실패: {str(e)}")
    
    def _read_stl_ascii(self, file_path, vertices, faces, normals):
        """ASCII 형식의 STL 파일을 읽어옵니다."""
        vertex_map = {}  # 정점 중복 제거를 위한 맵
        
        with open(file_path, 'r') as f:
            current_normal = None
            current_face = []
            
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if not parts:
                    continue
                
                if parts[0] == 'facet' and parts[1] == 'normal':
                    # 법선 벡터 읽기
                    current_normal = [float(parts[2]), float(parts[3]), float(parts[4])]
                    current_face = []
                
                elif parts[0] == 'vertex':
                    # 정점 읽기
                    vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                    
                    # 정점 중복 제거
                    vertex_key = tuple(vertex)
                    if vertex_key in vertex_map:
                        vertex_idx = vertex_map[vertex_key]
                    else:
                        vertex_idx = len(vertices)
                        vertices.append(vertex)
                        vertex_map[vertex_key] = vertex_idx
                    
                    current_face.append(vertex_idx)
                
                elif parts[0] == 'endloop':
                    # 면 완성
                    if len(current_face) == 3:
                        faces.append(current_face)
                        normals.append(current_normal)
                    elif len(current_face) == 4:
                        # 사각형을 삼각형 두 개로 분할
                        faces.append([current_face[0], current_face[1], current_face[2]])
                        faces.append([current_face[0], current_face[2], current_face[3]])
                        normals.extend([current_normal, current_normal])
    
    def _read_stl_binary(self, file_path, vertices, faces, normals):
        """바이너리 형식의 STL 파일을 읽어옵니다."""
        vertex_map = {}  # 정점 중복 제거를 위한 맵
        
        with open(file_path, 'rb') as f:
            # 헤더 건너뛰기
            f.seek(80)
            
            # 삼각형 개수 읽기
            num_triangles = struct.unpack('<I', f.read(4))[0]
            
            for _ in range(num_triangles):
                # 법선 벡터 읽기
                normal = list(struct.unpack('<fff', f.read(12)))
                
                # 정점 읽기
                face = []
                for _ in range(3):
                    vertex = list(struct.unpack('<fff', f.read(12)))
                    
                    # 정점 중복 제거
                    vertex_key = tuple(vertex)
                    if vertex_key in vertex_map:
                        vertex_idx = vertex_map[vertex_key]
                    else:
                        vertex_idx = len(vertices)
                        vertices.append(vertex)
                        vertex_map[vertex_key] = vertex_idx
                    
                    face.append(vertex_idx)
                
                # 속성 바이트 읽기 (사용하지 않음)
                f.read(2)
                
                faces.append(face)
                normals.append(normal)
    
    def _read_obj(self, file_path):
        """OBJ 파일을 읽어옵니다."""
        path = Path(file_path)
        vertices = []
        faces = []
        normals = []
        face_materials = []
        current_material = None
        
        # MTL 파일 찾기
        mtl_file = None
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('mtllib'):
                    mtl_name = line.split()[1]
                    mtl_file = path.parent / mtl_name
                    break
        
        # MTL 파일이 있다면 읽기
        if mtl_file and mtl_file.exists():
            self._read_mtl(mtl_file)
        
        try:
            with open(path, 'r') as f:
                for line in f:
                    if line.startswith('#'): continue
                    
                    values = line.strip().split()
                    if not values: continue
                    
                    if values[0] == 'v':
                        # 정점 데이터
                        vertices.append([float(x) for x in values[1:4]])
                        
                    elif values[0] == 'vn':
                        # 법선 벡터
                        normals.append([float(x) for x in values[1:4]])
                        
                    elif values[0] == 'usemtl':
                        # 재질 변경
                        current_material = values[1]
                        
                    elif values[0] == 'f':
                        # 면 데이터 처리
                        face = []
                        for v in values[1:]:
                            # f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3 형식 처리
                            vertex_idx = int(v.split('/')[0]) - 1  # OBJ는 1부터 시작
                            face.append(vertex_idx)
                        
                        if len(face) == 3:
                            faces.append(face)
                            face_materials.append(current_material)
                        elif len(face) == 4:
                            # 사각형을 삼각형 두 개로 분할
                            faces.append([face[0], face[1], face[2]])
                            faces.append([face[0], face[2], face[3]])
                            face_materials.extend([current_material, current_material])
            
            if not vertices:
                raise ValueError("파일에서 정점을 찾을 수 없습니다.")
            
            self.vertices = np.array(vertices)
            self.faces = np.array(faces)
            if normals:
                self.normals = np.array(normals)
            else:
                self._compute_normals()
            
            # 재질 정보 저장
            if face_materials:
                self.face_materials = face_materials
            
            return self
            
        except Exception as e:
            raise ValueError(f"OBJ 파일 읽기 실패: {str(e)}")
    
    def _read_mtl(self, mtl_file):
        """MTL 파일을 읽어옵니다."""
        current_material = None
        
        try:
            with open(mtl_file, 'r') as f:
                for line in f:
                    if line.startswith('#'): continue
                    
                    values = line.strip().split()
                    if not values: continue
                    
                    if values[0] == 'newmtl':
                        current_material = values[1]
                        self.materials[current_material] = {
                            'name': current_material,
                            'ambient': [0.2, 0.2, 0.2],
                            'diffuse': [0.8, 0.8, 0.8],
                            'specular': [1.0, 1.0, 1.0],
                            'shininess': 0.0,
                            'texture': None
                        }
                    
                    elif current_material is not None:
                        if values[0] == 'Ka':
                            self.materials[current_material]['ambient'] = [float(x) for x in values[1:4]]
                        elif values[0] == 'Kd':
                            self.materials[current_material]['diffuse'] = [float(x) for x in values[1:4]]
                        elif values[0] == 'Ks':
                            self.materials[current_material]['specular'] = [float(x) for x in values[1:4]]
                        elif values[0] == 'Ns':
                            self.materials[current_material]['shininess'] = float(values[1])
                        elif values[0] == 'map_Kd':
                            texture_path = mtl_file.parent / values[1]
                            if texture_path.exists():
                                self.materials[current_material]['texture'] = str(texture_path)
        
        except Exception as e:
            print(f"MTL 파일 읽기 실패: {str(e)}")
    
    def _compute_normals(self):
        """법선 벡터를 계산합니다."""
        if self.vertices is None or self.faces is None:
            return
        
        # 법선 벡터 초기화
        self.normals = np.zeros((len(self.vertices), 3))
        
        # 각 면에 대해 법선 벡터 계산
        for face in self.faces:
            v1 = self.vertices[face[0]]
            v2 = self.vertices[face[1]]
            v3 = self.vertices[face[2]]
            
            # 면의 법선 벡터 계산
            normal = np.cross(v2 - v1, v3 - v1)
            normal = normal / np.linalg.norm(normal)
            
            # 각 정점에 법선 벡터 추가
            for vertex_idx in face:
                self.normals[vertex_idx] += normal
        
        # 법선 벡터 정규화
        for i in range(len(self.normals)):
            norm = np.linalg.norm(self.normals[i])
            if norm > 0:
                self.normals[i] = self.normals[i] / norm

    def get_material(self, face_index):
        """면의 재질 정보를 반환합니다."""
        if self.face_materials is None or face_index >= len(self.face_materials):
            return None
        material_name = self.face_materials[face_index]
        return self.materials.get(material_name) 