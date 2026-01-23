import cv2
import numpy as np
import open3d as o3d
from retinaface import RetinaFace
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass


@dataclass
class FaceAlignmentResult:
    """Face alignment result data class using Open3D"""
    front_plane: o3d.geometry.TriangleMesh
    front_texture: o3d.geometry.Image
    front_plane_params: Dict
    right_plane: Optional[o3d.geometry.TriangleMesh] = None
    right_texture: Optional[o3d.geometry.Image] = None
    right_plane_params: Optional[Dict] = None
    left_plane: Optional[o3d.geometry.TriangleMesh] = None
    left_texture: Optional[o3d.geometry.Image] = None
    left_plane_params: Optional[Dict] = None
    

class FaceDetector:
    """Face detection class"""
    
    def detect_face_landmarks(self, image_path: str) -> Optional[Dict]:
        """Detect face landmarks using RetinaFace"""
        try:
            result = RetinaFace.detect_faces(image_path)
            if not result:
                print("No face detected.")
                return None
            
            # Use first detected face
            face_data = list(result.values())[0]
            landmarks = face_data['landmarks']
            
            return {
                'left_eye': landmarks['left_eye'],
                'right_eye': landmarks['right_eye'],
                'nose': landmarks['nose'],
                'mouth_left': landmarks['mouth_left'],
                'mouth_right': landmarks['mouth_right']
            }
        except Exception as e:
            print(f"Error during face detection: {e}")
            return None


class ImageAligner:
    """Image alignment class"""
    
    def align_face_horizontal(self, image: np.ndarray, left_eye: Tuple[float, float], 
                            right_eye: Tuple[float, float]) -> Tuple[np.ndarray, float, np.ndarray]:
        """Align face horizontally based on eyes"""
        # Calculate angle between eyes
        delta_x = right_eye[0] - left_eye[0]
        delta_y = right_eye[1] - left_eye[1]
        angle_rad = np.arctan2(delta_y, delta_x)
        angle_deg = np.degrees(angle_rad)
        
        print(f"[DEBUG] Original angle: {angle_deg:.2f} degrees")
        print(f"[DEBUG] Left eye: {left_eye}, Right eye: {right_eye}")
        
        # Adjust if angle is too large (>90 degrees indicates potential issue)
        if abs(angle_deg) > 90:
            # Eyes might be swapped, try swapping order
            delta_x = left_eye[0] - right_eye[0]
            delta_y = left_eye[1] - right_eye[1]
            angle_rad = np.arctan2(delta_y, delta_x)
            angle_deg = np.degrees(angle_rad)
            print(f"[DEBUG] Angle after swapping eyes: {angle_deg:.2f} degrees")
        
        # Further adjustment if angle is still large
        if angle_deg > 90:
            angle_deg -= 180
        elif angle_deg < -90:
            angle_deg += 180
            
        # Limit angle to ±45 degrees
        if angle_deg > 45:
            angle_deg = 45
        elif angle_deg < -45:
            angle_deg = -45
            
        print(f"[DEBUG] Final rotation angle: {angle_deg:.2f} degrees")
        
        # Rotate image
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)
        aligned_image = cv2.warpAffine(image, rotation_matrix, (w, h))
        
        return aligned_image, angle_deg, rotation_matrix
    
    def transform_landmarks(self, landmarks: Dict, rotation_matrix: np.ndarray) -> Dict:
        """Apply rotation transformation to landmarks"""
        transformed = {}
        for key, (x, y) in landmarks.items():
            # Convert to homogeneous coordinates and apply rotation
            point = np.array([x, y, 1])
            transformed_point = rotation_matrix @ point
            transformed[key] = (transformed_point[0], transformed_point[1])
        return transformed


class Face3DVisualizer:
    """3D visualization class using Open3D"""
    
    def __init__(self, target_eye_to_mouth_distance: float = 65.0):
        self.target_eye_to_mouth_distance = target_eye_to_mouth_distance
    
    def create_front_face_plane(self, image: np.ndarray, landmarks: Dict) -> Tuple[o3d.geometry.TriangleMesh, o3d.geometry.Image, Dict]:
        """Create 3D plane for front face in XZ plane using Open3D"""
        mouth_left = np.array(landmarks['mouth_left'])
        mouth_right = np.array(landmarks['mouth_right'])
        mouth_center = (mouth_left + mouth_right) / 2
        
        # 눈-입 거리 기반 스케일링으로 변경
        left_eye = np.array(landmarks['left_eye'])
        right_eye = np.array(landmarks['right_eye'])
        eye_center = (left_eye + right_eye) / 2
        eye_to_mouth_distance_pixels = np.linalg.norm(mouth_center - eye_center)
        
        if eye_to_mouth_distance_pixels > 0:
            scale_factor = self.target_eye_to_mouth_distance / eye_to_mouth_distance_pixels
        else:
            scale_factor = 1.0
        
        h, w = image.shape[:2]
        plane_width = w * scale_factor
        plane_height = h * scale_factor
        
        half_width = plane_width / 2
        half_height = plane_height / 2
        
        points = np.array([
            [-half_width, 0, -half_height],  # 0: bottom-left
            [half_width, 0, -half_height],   # 1: bottom-right  
            [half_width, 0, half_height],    # 2: top-right
            [-half_width, 0, half_height]    # 3: top-left
        ])
        
        # Reverse winding order to make the plane face the +Y direction.
        triangles = np.array([[0, 2, 1], [0, 3, 2]])
        
        plane_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(points),
            o3d.utility.Vector3iVector(triangles)
        )
        plane_mesh.compute_vertex_normals()
        
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                texture_image_rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
                print(f"[DEBUG] Front texture has alpha channel. Alpha range: {texture_image_rgba[:,:,3].min()}-{texture_image_rgba[:,:,3].max()}")
                transparent_pixels = np.sum(texture_image_rgba[:,:,3] < 255)
                print(f"[DEBUG] Number of transparent pixels: {transparent_pixels}/{texture_image_rgba.shape[0]*texture_image_rgba.shape[1]}")
            else:
                texture_image_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
                print(f"[DEBUG] Front texture converted to RGBA (no original alpha)")
        else:
            texture_image_rgba = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
            print(f"[DEBUG] Front texture converted from grayscale to RGBA")
            
        o3d_texture = o3d.geometry.Image(texture_image_rgba)    
        plane_mesh.textures = [o3d_texture]
        
        # Define UV coordinates for the 4 vertices of the quad.
        uv_map = {
            0: [0.0, 1.0],  # UV for vertex 0 (bottom-left)
            1: [1.0, 1.0],  # UV for vertex 1 (bottom-right)
            2: [1.0, 0.0],  # UV for vertex 2 (top-right)
            3: [0.0, 0.0],  # UV for vertex 3 (top-left)
        }

        # Build the triangle_uvs list dynamically based on the triangle vertex order
        triangle_uvs = []
        for tri in triangles:
            triangle_uvs.extend([uv_map[tri[0]], uv_map[tri[1]], uv_map[tri[2]]])

        plane_mesh.triangle_uvs = o3d.utility.Vector2dVector(np.array(triangle_uvs))
        plane_mesh.triangle_material_ids = o3d.utility.IntVector(np.zeros(len(triangles), dtype=np.int32))
        
        mouth_center_3d_x = (mouth_center[0] - w/2) * scale_factor
        mouth_center_3d_z = -(mouth_center[1] - h/2) * scale_factor
        
        plane_mesh.translate((-mouth_center_3d_x, 0, -mouth_center_3d_z), relative=True)
        
        # 눈-입 거리 3D 계산 (이미 위에서 계산됨)
        eye_to_mouth_distance_3d = eye_to_mouth_distance_pixels * scale_factor
        
        print(f"[DEBUG] Front face - Eye center: ({eye_center[0]:.2f}, {eye_center[1]:.2f})")
        print(f"[DEBUG] Front face - Mouth center: ({mouth_center[0]:.2f}, {mouth_center[1]:.2f})")
        print(f"[DEBUG] Front face - Eye to mouth distance (pixels): {eye_to_mouth_distance_pixels:.2f}")
        print(f"[DEBUG] Front face - Target eye-mouth distance: {self.target_eye_to_mouth_distance:.1f}mm")
        print(f"[DEBUG] Front face - Calculated scale factor: {scale_factor:.4f}")
        print(f"[DEBUG] Front face - Eye to mouth distance (3D): {eye_to_mouth_distance_3d:.2f}mm")
        
        plane_params = {
            'width': plane_width,
            'height': plane_height,
            'scale_factor': scale_factor,
            'mouth_center_3d': (0, 0, 0),
            'target_eye_to_mouth_distance': self.target_eye_to_mouth_distance,
            'face_height': plane_height,
            'eye_to_mouth_distance_pixels': eye_to_mouth_distance_pixels,
            'eye_to_mouth_distance_3d': eye_to_mouth_distance_3d
        }
        
        return plane_mesh, o3d_texture, plane_params
    
    def create_right_face_plane(self, image: np.ndarray, landmarks: Dict, front_plane_params: Dict) -> Tuple[o3d.geometry.TriangleMesh, o3d.geometry.Image, Dict]:
        """Create 3D plane for right face in YZ plane (facing -X direction)"""
        return self._create_side_face_plane(image, landmarks, front_plane_params, "RIGHT")
    
    def create_left_face_plane(self, image: np.ndarray, landmarks: Dict, front_plane_params: Dict) -> Tuple[o3d.geometry.TriangleMesh, o3d.geometry.Image, Dict]:
        """Create 3D plane for left face in YZ plane (facing +X direction)"""
        return self._create_side_face_plane(image, landmarks, front_plane_params, "LEFT")

    def _create_side_face_plane(self, image: np.ndarray, landmarks: Dict, front_plane_params: Dict, 
                               side_name: str) -> Tuple[o3d.geometry.TriangleMesh, o3d.geometry.Image, Dict]:
        """Creates a side face plane directly in the YZ plane with correct orientation."""
        mouth_left = np.array(landmarks['mouth_left'])
        mouth_right = np.array(landmarks['mouth_right'])
        mouth_center = (mouth_left + mouth_right) / 2
        
        h, w = image.shape[:2]
        
        front_face_height = front_plane_params['face_height']
        target_eye_to_mouth_distance = front_plane_params['target_eye_to_mouth_distance']
        front_eye_to_mouth_distance_3d = front_plane_params['eye_to_mouth_distance_3d']

        # 측면 얼굴에서 눈~입 거리 계산
        left_eye = np.array(landmarks['left_eye'])
        right_eye = np.array(landmarks['right_eye'])
        eye_center = (left_eye + right_eye) / 2
        side_eye_to_mouth_distance_pixels = np.linalg.norm(mouth_center - eye_center)
        
        print(f"[DEBUG] {side_name} face - Eye center: ({eye_center[0]:.2f}, {eye_center[1]:.2f})")
        print(f"[DEBUG] {side_name} face - Mouth center: ({mouth_center[0]:.2f}, {mouth_center[1]:.2f})")
        print(f"[DEBUG] {side_name} face - Eye to mouth distance (pixels): {side_eye_to_mouth_distance_pixels:.2f}")
        print(f"[DEBUG] Front face - Eye to mouth distance (3D): {front_eye_to_mouth_distance_3d:.2f}")

        # 개선된 스케일 계산: 측면 얼굴의 눈~입 거리를 정면 얼굴의 눈~입 거리에 맞춤
        if side_eye_to_mouth_distance_pixels > 0:
            scale_factor = front_eye_to_mouth_distance_3d / side_eye_to_mouth_distance_pixels
        else:
            # fallback: 기존 방식 사용
            scale_factor = front_face_height / h
            print(f"[WARNING] {side_name} face - Eye to mouth distance is 0, using fallback scale")
        
        print(f"[DEBUG] {side_name} face - Calculated scale factor: {scale_factor:.4f}")
        
        plane_width = w * scale_factor  # This is depth in Y direction
        plane_height = h * scale_factor  # This is height in Z direction
        
        half_width = plane_width / 2
        half_height = plane_height / 2
        
        # Create plane directly in YZ plane
        points = np.array([
            [0, -half_width, -half_height],   # 0: bottom-back
            [0, half_width, -half_height],    # 1: bottom-front
            [0, half_width, half_height],     # 2: top-front
            [0, -half_width, half_height]     # 3: top-back
        ])

        # Define UV coordinates for the 4 vertices of the quad.
        # (0,0) is top-left of the texture image.
        uv_map = {
            0: [0.0, 1.0],  # UV for vertex 0 (bottom-left)
            1: [1.0, 1.0],  # UV for vertex 1 (bottom-right)
            2: [1.0, 0.0],  # UV for vertex 2 (top-right)
            3: [0.0, 0.0],  # UV for vertex 3 (top-left)
        }

        if side_name == "RIGHT":
            # Right plane faces inwards (-X). Reverse winding order to flip normals.
            triangles = np.array([[0, 2, 1], [0, 3, 2]])
            offset_x = 0.1 #mouth_width_3d / 2.0
        else:  # LEFT
            # Left plane faces inwards (+X). Standard winding order.
            triangles = np.array([[0, 1, 2], [0, 2, 3]])
            offset_x = -0.1 #mouth_width_3d / 2.0
        
        # Apply the X offset to position the plane correctly
        points[:, 0] = offset_x
        
        plane_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(points),
            o3d.utility.Vector3iVector(triangles)
        )
        
        # Build the triangle_uvs list dynamically based on the triangle vertex order
        triangle_uvs = []
        for tri in triangles:
            triangle_uvs.extend([uv_map[tri[0]], uv_map[tri[1]], uv_map[tri[2]]])
        
        if len(image.shape) == 3:
            texture_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            texture_image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        o3d_texture = o3d.geometry.Image(texture_image_rgb)
        plane_mesh.textures = [o3d_texture]
        
        plane_mesh.triangle_uvs = o3d.utility.Vector2dVector(np.array(triangle_uvs))
        plane_mesh.triangle_material_ids = o3d.utility.IntVector(np.zeros(len(triangles), dtype=np.int32))
        
        # Align mouth center. For side view: image_x -> 3D_Y, image_y -> 3D_Z
        mouth_center_y_on_plane = (mouth_center[0] - w/2) * scale_factor
        mouth_center_z_on_plane = -(mouth_center[1] - h/2) * scale_factor
        
        # Translate to align the mouth to the YZ plane of the origin
        translation_vector = np.array([0, -mouth_center_y_on_plane, -mouth_center_z_on_plane])
        plane_mesh.translate(translation_vector, relative=True)
        plane_mesh.compute_vertex_normals()
        
        plane_params = {
            'width': plane_width,
            'height': plane_height,
            'scale_factor': scale_factor
        }
        
        return plane_mesh, o3d_texture, plane_params


class FaceAlignment3D:
    """Main face 3D alignment class"""
    
    def __init__(self, front_image_path: str = "photo/hk1.jpg", 
                 right_image_path: str = "photo/hk3.jpg", 
                 left_image_path: str = "photo/hk2.jpg"):
        self.front_image_path = front_image_path
        self.right_image_path = right_image_path
        self.left_image_path = left_image_path
        self.face_detector = FaceDetector()
        self.image_aligner = ImageAligner()
        self.visualizer = Face3DVisualizer()
        
    def run_registration(self, visualize=False) -> Optional[Tuple[np.ndarray, FaceAlignmentResult]]:
        """
        Aligns faces and returns a transformation matrix and a FaceAlignmentResult object
        containing separate Open3D meshes for front, left, and right planes.
        """
        # 1. Process front face to get the initial rotation matrix
        front_image = cv2.imread(self.front_image_path, cv2.IMREAD_UNCHANGED)
        if front_image is None:
            print(f"Cannot load front image: {self.front_image_path}")
            return None
        
        landmarks = self.face_detector.detect_face_landmarks(self.front_image_path)
        if landmarks is None:
            print("Front face landmarks not detected")
            return None
            
        _, _, rotation_matrix = self.image_aligner.align_face_horizontal(
            front_image, landmarks['left_eye'], landmarks['right_eye']
        )
        
        # 2. Perform the full alignment process to get the 3D planes
        alignment_result = self.align_faces()
        if alignment_result is None:
            return None

        # 3. Optionally visualize the result
        if visualize:
            self.visualize_result(alignment_result)

        # 4. Return the rotation matrix and the result object with separate planes
        return rotation_matrix, alignment_result

    def align_faces(self) -> Optional[FaceAlignmentResult]:
        """Align front, right, and left faces in 3D space"""
        # 1. Process front face
        front_result = self.process_front_face()
        if front_result is None:
            return None
            
        # 2. Process right face, passing all front plane parameters
        right_result = self.process_right_face(front_result.front_plane_params)
        if right_result is None:
            print("Right face processing failed, continuing without right face")
        
        # 3. Process left face, passing all front plane parameters
        left_result = self.process_left_face(front_result.front_plane_params)
        if left_result is None:
            print("Left face processing failed, continuing without left face")
        
        # Combine results
        result = FaceAlignmentResult(
            front_plane=front_result.front_plane,
            front_texture=front_result.front_texture,
            front_plane_params=front_result.front_plane_params,
            right_plane=right_result[0] if right_result else None,
            right_texture=right_result[1] if right_result else None,
            right_plane_params=right_result[2] if right_result else None,
            left_plane=left_result[0] if left_result else None,
            left_texture=left_result[1] if left_result else None,
            left_plane_params=left_result[2] if left_result else None
        )
        
        # Rotate all planes 180 degrees around the Z-axis to face +Y direction
        # center = np.array([0, 0, 0])
        # R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, np.pi))
        
        # if result.front_plane:
        #     result.front_plane.rotate(R, center=center)
        # if result.right_plane:
        #     result.right_plane.rotate(R, center=center)
        # if result.left_plane:
        #     result.left_plane.rotate(R, center=center)
            
        return result
    
    def process_front_face(self) -> Optional[FaceAlignmentResult]:
        """Process the front face"""
        # Load image
        image = cv2.imread(self.front_image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Cannot load front image: {self.front_image_path}")
            return None
        
        print(f"Front image loaded: {self.front_image_path}")
        
        # Detect face landmarks
        landmarks = self.face_detector.detect_face_landmarks(self.front_image_path)
        if landmarks is None:
            return None
            
        print("Front face landmarks detected")
        
        # Landmark 내용물 출력
        print("=== Front Face Landmarks ===")
        for key, (x, y) in landmarks.items():
            print(f"  {key}: ({x:.2f}, {y:.2f})")
        print("============================")
        
        # Align face horizontally based on eyes
        aligned_image, rotation_angle, rotation_matrix = self.image_aligner.align_face_horizontal(
            image, landmarks['left_eye'], landmarks['right_eye']
        )
        
        print(f"Front face horizontally aligned (rotation: {rotation_angle:.2f} degrees)")
        
        # Transform landmarks using rotation matrix
        aligned_landmarks = self.image_aligner.transform_landmarks(landmarks, rotation_matrix)
        
        print("Front face transformed landmarks calculated")
        
        # 정렬된 Landmark 내용물 출력
        print("=== Front Face Landmarks (Aligned) ===")
        for key, (x, y) in aligned_landmarks.items():
            print(f"  {key}: ({x:.2f}, {y:.2f})")
        print("=====================================")
        
        # Create 3D plane
        front_plane, front_texture, plane_params = self.visualizer.create_front_face_plane(
            aligned_image, aligned_landmarks
        )
        
        print("Front face 3D plane created")
        
        return FaceAlignmentResult(
            front_plane=front_plane,
            front_texture=front_texture,
            front_plane_params=plane_params
        )
    
    def process_right_face(self, front_plane_params: Dict) -> Optional[Tuple[o3d.geometry.TriangleMesh, o3d.geometry.Image, Dict]]:
        """Process the right face"""
        # Load image
        image = cv2.imread(self.right_image_path)
        if image is None:
            print(f"Cannot load right image: {self.right_image_path}")
            return None
        
        print(f"Right image loaded: {self.right_image_path}")
        
        # Detect face landmarks
        landmarks = self.face_detector.detect_face_landmarks(self.right_image_path)
        if landmarks is None:
            return None
            
        print("Right face landmarks detected")
        
        # Landmark 내용물 출력
        print("=== Right Face Landmarks ===")
        for key, (x, y) in landmarks.items():
            print(f"  {key}: ({x:.2f}, {y:.2f})")
        print("============================")
        
        
        # For side faces, we don't need horizontal alignment based on eyes
        aligned_image = image
        aligned_landmarks = landmarks
        
        print("Right face: skipping horizontal alignment (side view)")
        
        # Create 3D plane in YZ plane (facing +X direction)
        right_plane, right_texture, plane_params = self.visualizer.create_right_face_plane(
            aligned_image, aligned_landmarks, front_plane_params
        )
        
        print("Right face 3D plane created")
        
        return right_plane, right_texture, plane_params
    
    def process_left_face(self, front_plane_params: Dict) -> Optional[Tuple[o3d.geometry.TriangleMesh, o3d.geometry.Image, Dict]]:
        """Process the left face"""
        # Load image
        image = cv2.imread(self.left_image_path)
        if image is None:
            print(f"Cannot load left image: {self.left_image_path}")
            return None
        
        print(f"Left image loaded: {self.left_image_path}")
        
        # Flip left image horizontally to make it face the same direction as right face
        image = cv2.flip(image, 1)  # 1 means horizontal flip
        print("Left image: applied horizontal flip")
        
        # Detect face landmarks on the original image before flipping
        landmarks = self.face_detector.detect_face_landmarks(self.left_image_path)  
        if landmarks is None:
            return None
            
        # Also flip the landmarks to match the flipped image
        h, w = image.shape[:2]
        flipped_landmarks = {}
        for key, (x, y) in landmarks.items():
            # Flip X coordinate: new_x = width - old_x
            flipped_landmarks[key] = (w - x, y)
        
        print("Left face landmarks detected and flipped")
        
        # Landmark 내용물 출력 (원본)
        print("=== Left Face Landmarks (Original) ===")
        for key, (x, y) in landmarks.items():
            print(f"  {key}: ({x:.2f}, {y:.2f})")
        print("=====================================")
        
        # Landmark 내용물 출력 (플립된 후)
        print("=== Left Face Landmarks (Flipped) ===")
        for key, (x, y) in flipped_landmarks.items():
            print(f"  {key}: ({x:.2f}, {y:.2f})")
        print("====================================")
        
        # For side faces, we don't need horizontal alignment based on eyes
        aligned_image = image
        aligned_landmarks = flipped_landmarks
        
        print("Left face: skipping horizontal alignment (side view)")
        
        # Create 3D plane in YZ plane (now facing same direction as right face)
        left_plane, left_texture, plane_params = self.visualizer.create_left_face_plane(
            aligned_image, aligned_landmarks, front_plane_params
        )
        
        print("Left face 3D plane created")
        
        return left_plane, left_texture, plane_params
    
    def visualize_result(self, result: FaceAlignmentResult):
        """Visualize result in 3D using Open3D"""
        
        # Check texture alpha values
        if result.front_texture:
            texture_array = np.asarray(result.front_texture)
            print(f"[DEBUG] Front texture shape: {texture_array.shape}")
            if len(texture_array.shape) == 3 and texture_array.shape[2] == 4:
                alpha_channel = texture_array[:,:,3]
                print(f"[DEBUG] Alpha channel range: {alpha_channel.min()}-{alpha_channel.max()}")
                transparent_count = np.sum(alpha_channel < 255)
                total_pixels = alpha_channel.shape[0] * alpha_channel.shape[1]
                print(f"[DEBUG] Transparent pixels: {transparent_count}/{total_pixels} ({transparent_count/total_pixels*100:.1f}%)")
            else:
                print(f"[DEBUG] No alpha channel in front texture")
        
        geometries = []
        if result.front_plane:
            geometries.append(result.front_plane)
        if result.right_plane:
            geometries.append(result.right_plane)
        if result.left_plane:
            geometries.append(result.left_plane)

        if not geometries:
            print("No geometries to visualize.")
            return

        # Add coordinate axes
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50, origin=[0, 0, 0])
        geometries.append(axes)

        # Add origin marker (mouth position)  
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5)
        sphere.paint_uniform_color([1.0, 0.0, 0.0]) # Red
        sphere.translate((0,0,0))
        geometries.append(sphere)
        
        o3d.visualization.draw_geometries(geometries, window_name="Face Alignment: All faces")


# Test execution
if __name__ == "__main__":
    # Test with all three images
    face_aligner = FaceAlignment3D(
        # front_image_path="../../example/data/photo/hk1.jpg",
        # right_image_path="../../example/data/photo/hk2.jpg",
        # left_image_path="../../example/data/photo/hk3.jpg"
        
        front_image_path="../../example/data/photo/su11.png",
        right_image_path="../../example/data/photo/su2.png",
        left_image_path="../../example/data/photo/su3.png"
    )
    
    # Perform face alignment
    rotation_matrix, alignment_result = face_aligner.run_registration(visualize=True)
    
    if rotation_matrix is not None:
        print("\n=== Face Alignment Successful ===")
        print("Returned a rotation matrix and a FaceAlignmentResult object.")
        print(f"Rotation Matrix:\n{rotation_matrix}")
        print(f"Alignment Result:\n{alignment_result}")
        
        
        if alignment_result.front_plane:
            print(f"- Front plane has {len(alignment_result.front_plane.vertices)} vertices.")
        if alignment_result.right_plane:
            print(f"- Right plane has {len(alignment_result.right_plane.vertices)} vertices.")
        if alignment_result.left_plane:
            print(f"- Left plane has {len(alignment_result.left_plane.vertices)} vertices.")
            
    else:
        print("Face alignment failed.")
