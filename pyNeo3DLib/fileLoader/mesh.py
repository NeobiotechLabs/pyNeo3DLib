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
        """Load mesh using appropriate method based on file extension."""
        path = Path(file_path)
        mesh = cls()
        
        if path.suffix.lower() == '.stl':
            return mesh._read_stl(file_path)
        elif path.suffix.lower() == '.obj':
            return mesh._read_obj(file_path)
        elif path.suffix.lower() == '.ply':
            return mesh._read_ply(file_path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def _read_stl(self, file_path):
        """Read STL file. Supports both ASCII and binary formats."""
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
        """Read ASCII format STL file."""
        vertex_map = {}  # Map for removing duplicate vertices
        
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
                        # Divide rectangle into two triangles
                        faces.append([current_face[0], current_face[1], current_face[2]])
                        faces.append([current_face[0], current_face[2], current_face[3]])
                        normals.extend([current_normal, current_normal])
    
    def _read_stl_binary(self, file_path, vertices, faces, normals):
        """Read binary format STL file."""
        vertex_map = {}  # Map for removing duplicate vertices
        
        with open(file_path, 'rb') as f:
            # Skip header
            f.seek(80)
            
            # Read triangle count
            num_triangles = struct.unpack('<I', f.read(4))[0]
            
            for _ in range(num_triangles):
                # Read normal vector
                normal = list(struct.unpack('<fff', f.read(12)))
                
                # Read vertex
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
        """Read OBJ file."""
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
        
        # Read MTL file if exists
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
                        # Material change
                        current_material = values[1]
                        
                    elif values[0] == 'f':
                        # Face data processing
                        face = []
                        face_uv = []
                        for v in values[1:]:
                            # f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3 format processing
                            indices = v.split('/')
                            vertex_idx = int(indices[0]) - 1  # OBJ starts from 1
                            face.append(vertex_idx)
                            
                            # Store UV index if exists
                            if len(indices) > 1 and indices[1]:
                                uv_idx = int(indices[1]) - 1
                                face_uv.append(uv_idx)
                            else:
                                face_uv.append(vertex_idx)  # Use vertex index if UV is not available
                        
                        if len(face) == 3:
                            faces.append(face)
                            face_uvs.append(face_uv)
                            face_materials.append(current_material)
                        elif len(face) == 4:
                            # Divide rectangle into two triangles
                            faces.append([face[0], face[1], face[2]])
                            faces.append([face[0], face[2], face[3]])
                            face_uvs.append([face_uv[0], face_uv[1], face_uv[2]])
                            face_uvs.append([face_uv[0], face_uv[2], face_uv[3]])
                            face_materials.extend([current_material, current_material])
            
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
    
    def _read_ply(self, file_path):
        """Read PLY file using Open3D for better compatibility."""
        path = Path(file_path)
        
        try:
            import open3d as o3d
            
            # Load PLY file using Open3D
            print(f"Loading PLY file using Open3D: {file_path}")
            mesh_o3d = o3d.io.read_triangle_mesh(str(path))
            
            if len(mesh_o3d.vertices) == 0:
                print("Open3D returned empty mesh, trying manual parsing")
                raise ValueError("No vertices found in PLY file")
            
            # Convert Open3D mesh to our format
            self.vertices = np.asarray(mesh_o3d.vertices)
            self.faces = np.asarray(mesh_o3d.triangles, dtype=np.int32)  # Ensure integer type
            
            print(f"Open3D successfully loaded: {len(self.vertices)} vertices, {len(self.faces)} faces")
            
            if mesh_o3d.has_vertex_normals():
                self.normals = np.asarray(mesh_o3d.vertex_normals)
                print("Loaded vertex normals from PLY file")
            else:
                self._compute_normals()
                print("Computed vertex normals")
                
            # Try to load UV coordinates manually (Open3D doesn't handle PLY UVs well)
            print("Attempting UV coordinate loading from PLY file...")
            uv_loaded = self._try_load_ply_uvs_simple(file_path)
            print(f"UV loading result: {uv_loaded}")
            
            if uv_loaded:
                print(f"Loaded {len(self.uvs)} UV coordinates from PLY file")
                # Ensure face_uvs are integers and same as faces for PLY
                if hasattr(self, 'uvs') and self.uvs is not None:
                    self.face_uvs = np.array(self.faces, dtype=np.int32)
            else:
                print("UV coordinate loading failed from PLY file")
                print("WARNING: No UV coordinates available - lip detection may not work correctly")
                # Don't create default UVs as they were incorrect
                # self._try_create_default_uvs()  # Disabled - use actual texture UV coordinates only
            
            # Final safety check: ensure face_uvs exists and is integer type
            if not hasattr(self, 'face_uvs') or self.face_uvs is None:
                print("Warning: face_uvs not set, creating from faces")
                self.face_uvs = np.array(self.faces, dtype=np.int32)
            else:
                # Ensure face_uvs is integer type
                if self.face_uvs.dtype != np.int32:
                    print(f"Warning: Converting face_uvs from {self.face_uvs.dtype} to int32")
                    self.face_uvs = self.face_uvs.astype(np.int32)
            
            print(f"Final check: faces dtype={self.faces.dtype}, face_uvs dtype={self.face_uvs.dtype}")
            
            return self
            
        except ImportError:
            print("Open3D not available, falling back to simple PLY parsing")
            return self._read_ply_simple_fallback(file_path)
        except Exception as e:
            print(f"Open3D PLY loading failed: {e}")
            print("Falling back to simple PLY parsing")  
            return self._read_ply_simple_fallback(file_path)
    
    def _try_load_ply_uvs_simple(self, file_path):
        """Simple UV coordinate loader for binary PLY files."""
        print("Attempting to load UV coordinates from PLY file...")
        try:
            with open(file_path, 'rb') as f:
                # Read header
                header_lines = []
                while True:
                    line = f.readline().decode('utf-8', errors='ignore').strip()
                    header_lines.append(line)
                    if 'end_header' in line:
                        break
                
                print(f"PLY Header lines: {header_lines[:10]}")  # Print first 10 header lines
                
                # Check format and find UV properties
                is_binary = any('format binary' in line for line in header_lines)
                vertex_count = 0
                uv_indices = {'u_idx': -1, 'v_idx': -1}
                
                # Parse vertex properties to find UV coordinates
                in_vertex_section = False
                prop_index = 0
                
                for line in header_lines:
                    if line.startswith('element vertex'):
                        vertex_count = int(line.split()[-1])
                        in_vertex_section = True
                        prop_index = 0  # Reset property index
                        print(f"Found vertex count: {vertex_count}")
                    elif line.startswith('element') and not line.startswith('element vertex'):
                        in_vertex_section = False  # End of vertex properties
                    elif line.startswith('property') and in_vertex_section:
                        print(f"Vertex property {prop_index}: {line}")
                        if 'texture_u' in line:
                            uv_indices['u_idx'] = prop_index
                            print(f"Found U coordinate at index {prop_index}")
                        elif 'texture_v' in line:
                            uv_indices['v_idx'] = prop_index
                            print(f"Found V coordinate at index {prop_index}")
                        prop_index += 1
                
                print(f"UV indices found: {uv_indices}")
                print(f"Is binary: {is_binary}, Vertex count: {vertex_count}")
                
                if vertex_count > 0 and uv_indices['u_idx'] >= 0 and uv_indices['v_idx'] >= 0:
                    print(f"Found UV coordinates at indices {uv_indices['u_idx']}, {uv_indices['v_idx']}")
                    
                    if is_binary:
                        # Read binary UV coordinates (8 floats per vertex)
                        uvs = []
                        print(f"Reading {vertex_count} UV coordinates from binary PLY (8 floats format)...")
                        
                        try:
                            for i in range(vertex_count):
                                # Read exactly 8 floats: x,y,z,nx,ny,nz,texture_u,texture_v
                                vertex_data = struct.unpack('<8f', f.read(32))  # 8 * 4 bytes = 32 bytes
                                
                                # Extract UV coordinates using the found indices
                                u = vertex_data[uv_indices['u_idx']]
                                v = vertex_data[uv_indices['v_idx']]
                                uvs.append([u, v])
                                
                                # Progress report
                                if i % 20000 == 0 and i > 0:
                                    print(f"Read {i}/{vertex_count} UV coordinates... (u={u:.3f}, v={v:.3f})")
                                    
                        except Exception as e:
                            print(f"Error reading UV coordinates at vertex {i}: {e}")
                            print(f"Successfully read {len(uvs)} UV coordinates")
                        
                        if uvs and len(uvs) > 0:
                            print(f"Loaded {len(uvs)} UV coordinates")
                            print(f"UV sample: {uvs[:3]}")
                            self.uvs = np.array(uvs)
                            # Ensure face_uvs are integers (PLY files sometimes have float indices)
                            self.face_uvs = np.array(self.faces, dtype=np.int32)
                            return True
                        else:
                            print("No UV coordinates could be read")
                else:
                    print("No UV coordinate properties found in PLY header")
                            
        except Exception as e:
            print(f"Could not load UV coordinates: {e}")
            
        return False
    
    def _try_create_default_uvs(self):
        """Create default UV coordinates based on vertex positions."""
        if hasattr(self, 'vertices') and self.vertices is not None and len(self.vertices) > 0:
            print("Creating default UV coordinates from vertex positions...")
            
            # Create UV coordinates based on XY projection
            vertices = self.vertices
            min_x, max_x = np.min(vertices[:, 0]), np.max(vertices[:, 0])
            min_y, max_y = np.min(vertices[:, 1]), np.max(vertices[:, 1])
            
            # Avoid division by zero
            range_x = max_x - min_x if max_x != min_x else 1.0
            range_y = max_y - min_y if max_y != min_y else 1.0
            
            # Normalize to 0-1 range
            uvs = []
            for vertex in vertices:
                u = (vertex[0] - min_x) / range_x
                v = (vertex[1] - min_y) / range_y
                uvs.append([u, v])
            
            self.uvs = np.array(uvs)
            self.face_uvs = np.array(self.faces, dtype=np.int32)
            
            print(f"Created {len(uvs)} default UV coordinates")
            print(f"UV range: U({np.min(self.uvs[:, 0]):.3f}-{np.max(self.uvs[:, 0]):.3f}), V({np.min(self.uvs[:, 1]):.3f}-{np.max(self.uvs[:, 1]):.3f})")
            return True
        else:
            print("Cannot create default UVs - no vertices available")
            return False
    
    def _read_ply_simple_fallback(self, file_path):
        """Super simple PLY fallback - create basic mesh from vertices only."""
        print("Using simple fallback PLY loader")
        
        try:
            # Create a basic mesh with vertices from the binary data
            with open(file_path, 'rb') as f:
                # Skip header
                while True:
                    line = f.readline().decode('utf-8', errors='ignore').strip()
                    if 'end_header' in line:
                        break
                
                # Try to read some vertices (assuming standard x,y,z,nx,ny,nz,u,v format)
                vertices = []
                uvs = []
                
                try:
                    for i in range(min(50000, 113066)):  # Read up to 50k vertices
                        vertex_data = struct.unpack('<8f', f.read(32))  # 8 floats
                        x, y, z, nx, ny, nz, u, v = vertex_data
                        vertices.append([x, y, z])
                        uvs.append([u, v])
                        
                        if i % 10000 == 0:
                            print(f"Read {i} vertices...")
                            
                except:
                    print(f"Read {len(vertices)} vertices before error")
                
                if len(vertices) > 100:
                    self.vertices = np.array(vertices)
                    self.uvs = np.array(uvs)
                    
                    # Create simple triangular faces
                    faces = []
                    for i in range(0, len(vertices) - 2, 3):
                        faces.append([i, i+1, i+2])
                    
                    self.faces = np.array(faces, dtype=np.int32)
                    self.face_uvs = np.array(faces, dtype=np.int32)
                    
                    # Compute normals
                    self._compute_normals()
                    
                    print(f"Simple PLY loader: {len(vertices)} vertices, {len(faces)} faces, {len(uvs)} UVs")
                    return self
                    
        except Exception as e:
            print(f"Simple PLY fallback failed: {e}")
            
        raise ValueError("Could not load PLY file with any method")
    
        # This function is no longer used - removed complex manual parser
    
        # Removed all complex PLY parsing functions - using Open3D instead
    
    def _read_mtl(self, mtl_file):
        """Read MTL file."""
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
        """Calculate normal vectors."""
        if self.vertices is None or self.faces is None:
            return
        
        # Initialize normal vectors
        self.normals = np.zeros((len(self.vertices), 3))
        
        # Calculate normal vectors for each face
        for face in self.faces:
            v1 = self.vertices[face[0]]
            v2 = self.vertices[face[1]]
            v3 = self.vertices[face[2]]
            
            # Calculate normal vector for the face
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
        """Return material information for a face."""
        if self.face_materials is None or face_index >= len(self.face_materials):
            return None
        material_name = self.face_materials[face_index]
        return self.materials.get(material_name)

    def extract_mesh_from_vertices(self, vertex_indices):
        """Create a partial mesh from selected vertices.
        
        Args:
            vertex_indices: List of selected vertex indices
            
        Returns:
            Partial mesh composed of selected vertices
        """
        # Convert selected vertex indices to a set for faster search
        vertex_set = set(vertex_indices)
        
        # Find faces containing selected vertices
        selected_faces = []
        vertex_mapping = {}  # Original index -> New index mapping
        used_vertices = set()  # Actual used vertices
        used_uvs = set()  # Actual used UV coordinates
        
        # Face filtering: Select faces containing selected vertices
        for i, face in enumerate(self.faces):
            # Check if selected vertices are in the face
            if any(v in vertex_set for v in face):
                # Add all vertices of the face to the used vertices set
                for v in face:
                    used_vertices.add(v)
                selected_faces.append(i)
                # Store UV index if exists
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
        
        # Copy UV coordinates if exists
        if self.uvs is not None and used_uvs:
            submesh.uvs = self.uvs[list(used_uvs)]
        
        # Transform face indices
        new_faces = []
        new_face_uvs = [] if self.face_uvs is not None else None
        
        for face_idx in selected_faces:
            old_face = self.faces[face_idx]
            new_face = [vertex_mapping[v] for v in old_face]
            new_faces.append(new_face)
            
            # Transform UV indices if exists
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