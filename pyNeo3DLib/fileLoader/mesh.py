import numpy as np
import struct
from pathlib import Path
import re

class Mesh:
    def __init__(self):
        self.vertices = None
        self.faces = None
        self.normals = None
        self.materials = {}  # MTL file read material information
        self.face_materials = None  # Material index for each face
        self.uvs = None
        self.face_uvs = None  # UV index for each face

    @classmethod
    def from_file(cls, file_path):
        """Loads a mesh from a file based on its extension."""
        path = Path(file_path)
        mesh = cls()
        
        if path.suffix.lower() == '.stl':
            return mesh._read_stl(file_path)
        elif path.suffix.lower() == '.obj':
            return mesh._read_obj(file_path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def _read_stl(self, file_path):
        """Reads an STL file. Supports both ASCII and binary formats."""
        path = Path(file_path)
        vertices = []
        faces = []
        normals = []
        
        try:
            # Check file format (ASCII or binary)
            with open(path, 'rb') as f:
                header = f.read(5)
                is_ascii = header.startswith(b'solid')
            
            if is_ascii:
                self._read_stl_ascii(path, vertices, faces, normals)
            else:
                self._read_stl_binary(path, vertices, faces, normals)
            
            if not vertices:
                raise ValueError("No vertices found in the file.")
            
            self.vertices = np.array(vertices)
            self.faces = np.array(faces)
            self.normals = np.array(normals)
            
            return self
            
        except Exception as e:
            raise ValueError(f"Failed to read STL file: {str(e)}")
    
    def _read_stl_ascii(self, file_path, vertices, faces, normals):
        """Reads an ASCII format STL file."""
        vertex_map = {}  # Map for vertex deduplication
        
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
                    # Read normal vector
                    current_normal = [float(parts[2]), float(parts[3]), float(parts[4])]
                    current_face = []
                
                elif parts[0] == 'vertex':
                    # Read vertex
                    vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                    
                    # Remove duplicate vertices
                    vertex_key = tuple(vertex)
                    if vertex_key in vertex_map:
                        vertex_idx = vertex_map[vertex_key]
                    else:
                        vertex_idx = len(vertices)
                        vertices.append(vertex)
                        vertex_map[vertex_key] = vertex_idx
                    
                    current_face.append(vertex_idx)
                
                elif parts[0] == 'endloop':
                    # Complete face
                    if len(current_face) == 3:
                        faces.append(current_face)
                        normals.append(current_normal)
                    elif len(current_face) == 4:
                        # Split quad into two triangles
                        faces.append([current_face[0], current_face[1], current_face[2]])
                        faces.append([current_face[0], current_face[2], current_face[3]])
                        normals.extend([current_normal, current_normal])
    
    def _read_stl_binary(self, file_path, vertices, faces, normals):
        """Reads a binary format STL file."""
        vertex_map = {}  # Map for vertex deduplication
        
        with open(file_path, 'rb') as f:
            # Skip header
            f.seek(80)
            
            # Read number of triangles
            num_triangles = struct.unpack('<I', f.read(4))[0]
            
            for _ in range(num_triangles):
                # Read normal vector
                normal = list(struct.unpack('<fff', f.read(12)))
                
                # Read vertices
                face = []
                for _ in range(3):
                    vertex = list(struct.unpack('<fff', f.read(12)))
                    
                    # Remove duplicate vertices
                    vertex_key = tuple(vertex)
                    if vertex_key in vertex_map:
                        vertex_idx = vertex_map[vertex_key]
                    else:
                        vertex_idx = len(vertices)
                        vertices.append(vertex)
                        vertex_map[vertex_key] = vertex_idx
                    
                    face.append(vertex_idx)
                
                # Read attribute bytes (not used)
                f.read(2)
                
                faces.append(face)
                normals.append(normal)
    
    def _read_obj(self, file_path):
        """Reads an OBJ file."""
        path = Path(file_path)
        vertices = []
        faces = []
        face_uvs = []  # Store UV indices
        normals = []
        face_materials = []
        uvs = []
        current_material = None
        
        # Find MTL file
        mtl_file = None
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('mtllib'):
                    mtl_name = line.split()[1]
                    mtl_file = path.parent / mtl_name
                    break
        
        # Read MTL file if it exists
        if mtl_file and mtl_file.exists():
            self._read_mtl(mtl_file)
        
        try:
            with open(path, 'r') as f:
                for line in f:
                    if line.startswith('#'): continue
                    
                    values = line.strip().split()
                    if not values: continue
                    
                    if values[0] == 'v':
                        # Vertex data
                        vertices.append([float(x) for x in values[1:4]])

                    elif values[0] == 'vt':
                        # UV data
                        uvs.append([float(x) for x in values[1:3]])
                        
                    elif values[0] == 'vn':
                        # Normal vector
                        normals.append([float(x) for x in values[1:4]])
                        
                    elif values[0] == 'usemtl':
                        # Change material
                        current_material = values[1]
                        
                    elif values[0] == 'f':
                        # Process face data
                        face = []
                        face_uv = []
                        for v in values[1:]:
                            # Handle f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3 format
                            indices = v.split('/')
                            vertex_idx = int(indices[0]) - 1  # OBJ starts from 1
                            face.append(vertex_idx)
                            
                            # Store UV index if it exists
                            if len(indices) > 1 and indices[1]:
                                uv_idx = int(indices[1]) - 1
                                face_uv.append(uv_idx)
                            else:
                                face_uv.append(vertex_idx)  # Use vertex index if no UV
            
            if not vertices:
                raise ValueError("No vertices found in the file.")
            
            self.vertices = np.array(vertices)
            self.faces = np.array(faces)
            if normals:
                self.normals = np.array(normals)
            else:
                self._compute_normals()
            if uvs:
                self.uvs = np.array(uvs)
                self.face_uvs = np.array(face_uvs)

            # Store material information
            if face_materials:
                self.face_materials = face_materials
            
            return self
            
        except Exception as e:
            raise ValueError(f"Failed to read OBJ file: {str(e)}")
    
    def _read_mtl(self, mtl_file):
        """Reads an MTL file."""
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
            print(f"Failed to read MTL file: {str(e)}")
    
    def _compute_normals(self):
        """Computes normal vectors."""
        if self.vertices is None or self.faces is None:
            return
        
        # Initialize normal vectors
        self.normals = np.zeros((len(self.vertices), 3))
        
        # Calculate normal vector for each face
        for face in self.faces:
            v1 = self.vertices[face[0]]
            v2 = self.vertices[face[1]]
            v3 = self.vertices[face[2]]
            
            # Calculate face normal vector
            normal = np.cross(v2 - v1, v3 - v1)
            normal = normal / np.linalg.norm(normal)
            
            # Add normal vector to each vertex
            for vertex_idx in face:
                self.normals[vertex_idx] += normal
        
        # Normalize normal vectors
        for i in range(len(self.normals)):
            norm = np.linalg.norm(self.normals[i])
            if norm > 0:
                self.normals[i] = self.normals[i] / norm

    def get_material(self, face_index):
        """Returns the material information for a face."""
        if self.face_materials is None or face_index >= len(self.face_materials):
            return None
        material_name = self.face_materials[face_index]
        return self.materials.get(material_name)

    def extract_mesh_from_vertices(self, vertex_indices):
        """Creates a submesh from selected vertices.
        
        Args:
            vertex_indices: List of selected vertex indices
            
        Returns:
            Submesh composed of selected vertices
        """
        # Convert selected vertex indices to set (for faster lookup)
        vertex_set = set(vertex_indices)
        
        # Find faces composed of selected vertices
        selected_faces = []
        vertex_mapping = {}  # Original index -> new index mapping
        used_vertices = set()  # Actually used vertices
        used_uvs = set()  # Actually used UV coordinates
        
        # Filter faces: select faces containing at least one selected vertex
        for i, face in enumerate(self.faces):
            # Check if any vertex in the face is selected
            if any(v in vertex_set for v in face):
                # Add all vertices of the face to used vertices set
                for v in face:
                    used_vertices.add(v)
                selected_faces.append(i)
                # If UV coordinates exist, store UV indices for this face
                if self.face_uvs is not None:
                    for uv_idx in self.face_uvs[i]:
                        used_uvs.add(uv_idx)
        
        if not selected_faces:
            return None
        
        # Create new mesh
        submesh = Mesh()
        
        # Copy vertices and create index mapping
        used_vertices = sorted(list(used_vertices))
        for i, old_idx in enumerate(used_vertices):
            vertex_mapping[old_idx] = i
        
        # Create UV coordinate mapping (if UV coordinates exist)
        uv_mapping = {}
        if self.face_uvs is not None:
            used_uvs = sorted(list(used_uvs))
            for i, old_idx in enumerate(used_uvs):
                uv_mapping[old_idx] = i
        
        # Copy vertices to new mesh
        submesh.vertices = self.vertices[used_vertices]
        
        # Copy UV coordinates if they exist
        if self.uvs is not None and used_uvs:
            submesh.uvs = self.uvs[list(used_uvs)]
        
        # Transform face indices
        new_faces = []
        new_face_uvs = [] if self.face_uvs is not None else None
        
        for face_idx in selected_faces:
            old_face = self.faces[face_idx]
            new_face = [vertex_mapping[v] for v in old_face]
            new_faces.append(new_face)
            
            # Transform UV indices if UV coordinates exist
            if self.face_uvs is not None:
                old_face_uv = self.face_uvs[face_idx]
                new_face_uv = [uv_mapping[uv] for uv in old_face_uv]
                new_face_uvs.append(new_face_uv)
        
        submesh.faces = np.array(new_faces)
        if new_face_uvs:
            submesh.face_uvs = np.array(new_face_uvs)
        
        # Calculate normal vectors
        submesh._compute_normals()
        
        return submesh 