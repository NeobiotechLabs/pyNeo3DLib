import os
import imageio
import numpy as np
import cv2 
import mediapipe as mp 
import open3d as o3d


class FacePhotoRegistration:
    def __init__(self, photo_path, visualization=False):
        self.photo_path = photo_path
        self.visualization = visualization
        
        self.mp_face_mesh = mp.solutions.face_mesh
        # Mediapipe FaceMesh에서 사용할 랜드마크 인덱스 정의
        # (https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png 참고)
        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144] # 왼쪽 눈 주변
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380] # 오른쪽 눈 주변
        self.LIPS_OUTER_INDICES = list(set(i for tup in self.mp_face_mesh.FACEMESH_LIPS for i in tup)) # 전체 외부 입술
        self.MOUTH_CORNER_LEFT_INDEX = 61 # 또는 78 - 왼쪽 입꼬리 (이미지 기준 왼쪽)
        self.MOUTH_CORNER_RIGHT_INDEX = 291 # 또는 308 - 오른쪽 입꼬리 (이미지 기준 오른쪽)

        self.TARGET_MOUTH_WIDTH_3D = 55.0 # mm 단위로 가정
    
    def run_registration(self):
        img_data_raw_rgb = self.__load_and_prepare_image()
        
        with self.mp_face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
            ) as face_mesh_processor:
            initial_landmarks = self.__get_landmark_coords(img_data_raw_rgb, face_mesh_processor)
            
            if not initial_landmarks:
                print("초기 랜드마크 감지 실패, 원본 이미지를 그대로 사용합니다.")
                aligned_landmarks = {
                    "lips_center": (img_data_raw_rgb.shape[1]/2, img_data_raw_rgb.shape[0]/2),
                    "mouth_left": (img_data_raw_rgb.shape[1]/4, img_data_raw_rgb.shape[0]/2),
                    "mouth_right": (3*img_data_raw_rgb.shape[1]/4, img_data_raw_rgb.shape[0]/2)
                }
                img_data_aligned_for_rotation = img_data_raw_rgb
                # M_eye_align은 이미 항등 행렬로 초기화되어 있음
                height_before_rot90 = img_data_raw_rgb.shape[0]
            else:
                # 3. 양 눈 수평 정렬을 위한 이미지 회전
                img_data_aligned_for_rotation, _, M_eye_align = self.__rotate_image_to_align_eyes(
                    img_data_raw_rgb, initial_landmarks["eye_left"], initial_landmarks["eye_right"]
                )
                height_before_rot90 = img_data_aligned_for_rotation.shape[0]
                
                # 4. 회전된 이미지에서 랜드마크 재감지
                aligned_landmarks = self.__get_landmark_coords(img_data_aligned_for_rotation.copy(), face_mesh_processor)
                if not aligned_landmarks:
                    print("정렬된 이미지에서 랜드마크 재감지 실패, 원본 이미지를 그대로 사용합니다.")
                    aligned_landmarks = initial_landmarks
                    img_data_aligned_for_rotation = img_data_raw_rgb
                    # 이 경우, 눈 정렬이 실패했으므로 M_eye_align을 항등 행렬로 리셋하고, 높이도 원본 이미지 높이로.
                    M_eye_align = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
                    height_before_rot90 = img_data_raw_rgb.shape[0]
        
        final_texture_image, p_x_size, p_z_size, tex_u, tex_v = \
            self.__calculate_plane_and_texture_parameters(
                img_data_aligned_for_rotation, 
                aligned_landmarks
            )
            
        print(f"p_x_size: {p_x_size}, p_z_size: {p_z_size}, tex_u: {tex_u}, tex_v: {tex_v}")
        
        # 텍스처 이미지를 클래스 변수로 저장
        self.final_texture_image = final_texture_image
        
        # p_x_size = 100
        # p_z_size = 100
        # tex_u = 0.55
        # tex_v = 0.55
        
        image_plane = self.__create_and_position_image_plane(p_x_size, p_z_size, tex_u, tex_v)
        
        if(self.visualization):
            self.__visualize_image_plane(image_plane)
        
        return M_eye_align, image_plane
    
    
        
    def __load_and_prepare_image(self):
        img_data_raw = imageio.v2.imread(self.photo_path)
        if img_data_raw.ndim == 2: # Grayscale to RGB
            img_data_raw = cv2.cvtColor(img_data_raw, cv2.COLOR_GRAY2RGB)
        elif img_data_raw.shape[2] == 4: # RGBA to RGB
            img_data_raw = cv2.cvtColor(img_data_raw, cv2.COLOR_RGBA2RGB)
        return img_data_raw
    
    def __get_landmark_coords(self, img_data_raw_rgb, face_mesh_processor):
        try:
            results = face_mesh_processor.process(img_data_raw_rgb)
        except Exception as e:
            print(f"Mediapipe 처리 중 오류: {e}")
            return None
            
        if not results.multi_face_landmarks:
            print("얼굴 랜드마크를 찾지 못했습니다.")
            return None

        h, w, _ = img_data_raw_rgb.shape
        landmarks = results.multi_face_landmarks[0].landmark # 첫 번째 감지된 얼굴 사용

        def get_avg_coords(indices):
            xs = [landmarks[i].x * w for i in indices if i < len(landmarks)]
            ys = [landmarks[i].y * h for i in indices if i < len(landmarks)]
            if not xs or not ys: return None
            return np.mean(xs), np.mean(ys)

        def get_single_coord(index):
            if index < len(landmarks):
                return landmarks[index].x * w, landmarks[index].y * h
            return None

        eye_left_center = get_avg_coords(self.LEFT_EYE_INDICES)
        eye_right_center = get_avg_coords(self.RIGHT_EYE_INDICES)
        lips_center = get_avg_coords(self.LIPS_OUTER_INDICES)
        mouth_corner_left = get_single_coord(self.MOUTH_CORNER_LEFT_INDEX)
        mouth_corner_right = get_single_coord(self.MOUTH_CORNER_RIGHT_INDEX)

        if not all([eye_left_center, eye_right_center, lips_center, mouth_corner_left, mouth_corner_right]):
            print("주요 랜드마크 중 일부를 감지하지 못했습니다.")
            return None
            
        # print(f"[DEBUG] Raw Landmarks: EyeL{eye_left_center}, EyeR{eye_right_center}, LipC{lips_center}, MouthL{mouth_corner_left}, MouthR{mouth_corner_right}")

        return {
            "eye_left": eye_left_center,
            "eye_right": eye_right_center,
            "lips_center": lips_center,
            "mouth_left": mouth_corner_left,
            "mouth_right": mouth_corner_right
        }
        
    def __rotate_image_to_align_eyes(self, image, eye_left, eye_right):
        """양 눈을 수평으로 정렬하도록 이미지를 회전합니다."""
        (ex1, ey1) = eye_left # 왼쪽 눈 (x1, y1)
        (ex2, ey2) = eye_right # 오른쪽 눈 (x2, y2)
        print(f"[DEBUG] Eye Coords: Left({ex1:.1f},{ey1:.1f}), Right({ex2:.1f},{ey2:.1f})")

        # 눈 사이의 각도 계산 (delta_y, delta_x). OpenCV 이미지 좌표계 (y는 아래로 증가)
        delta_x = ex2 - ex1
        delta_y = ey2 - ey1 # y2가 y1보다 크면 오른쪽 눈이 왼쪽 눈보다 아래에 있음
        angle_rad = np.arctan2(delta_y, delta_x)
        original_angle_deg = np.degrees(angle_rad)
        
        print(f"[DEBUG] 눈 벡터의 X축과의 각도 (arctan2(dy,dx)): {original_angle_deg:.2f} 도")

        angle_for_rotation = original_angle_deg

        print(f"[DEBUG] 이미지 회전에 적용될 각도 (이론상): {angle_for_rotation:.2f} 도")

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle_for_rotation, 1.0)
        rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
        
        return rotated_image, angle_for_rotation, M
    
    def __calculate_plane_and_texture_parameters(
        self,
        aligned_image_for_texture_base, 
        landmarks_on_rotated_image
        ):
        """
        최종 텍스처 이미지와 평면 크기(X,Z), 텍스처 좌표(u,v)를 계산합니다.
        (90도 회전 없이 이미지 비율을 그대로 사용)
        """
        lips_center_px = landmarks_on_rotated_image["lips_center"]
        mouth_left_px = landmarks_on_rotated_image["mouth_left"]
        mouth_right_px = landmarks_on_rotated_image["mouth_right"]

        # 디버깅: 최종 사용될 텍스처 이미지에 랜드마크 점 그리기
        final_image_for_texture = aligned_image_for_texture_base.copy()
        cv2.circle(final_image_for_texture, (int(lips_center_px[0]), int(lips_center_px[1])), 5, (0, 255, 0), -1) # 초록
        cv2.circle(final_image_for_texture, (int(mouth_left_px[0]), int(mouth_left_px[1])), 3, (255, 0, 0), -1)   # 파랑
        cv2.circle(final_image_for_texture, (int(mouth_right_px[0]), int(mouth_right_px[1])), 3, (0, 0, 255), -1)   # 빨강

        pixel_mouth_width = np.linalg.norm(np.array(mouth_right_px) - np.array(mouth_left_px))
        print(f"[DEBUG] 감지된 입 너비 (픽셀): {pixel_mouth_width:.2f}")

        # 최종 텍스처 이미지의 크기 (회전 없음)
        final_h_tex, final_w_tex, _ = final_image_for_texture.shape
        print(f"[DEBUG] 최종 텍스처 이미지 크기 (W, H): ({final_w_tex}, {final_h_tex})")

        # --- 스케일링 로직: 회전 없는 버전에 맞게 수정 ---
        # 입 너비(이미지의 수평)를 기준으로 3D 평면의 X축 크기(너비)를 계산
        if pixel_mouth_width > 1e-3:
            target_plane_x_size = (final_w_tex / pixel_mouth_width) * self.TARGET_MOUTH_WIDTH_3D
        else:
            print("[WARNING] 입 너비가 0에 가깝습니다. 임의의 스케일(100mm)로 평면 너비를 설정합니다.")
            target_plane_x_size = 100.0
        print(f"[DEBUG] 계산된 목표 평면 X 크기 (3D): {target_plane_x_size:.2f}")

        # 평면의 Z 크기(높이)를 최종 텍스처 이미지의 가로세로 비율에 맞게 조정
        if final_w_tex > 1e-3: # 0으로 나누기 방지
            target_plane_z_size = target_plane_x_size * (final_h_tex / float(final_w_tex))
        else:
            target_plane_z_size = target_plane_x_size # 너비가 0이면 정사각형으로 가정 (오류 방지)
        print(f"[DEBUG] 계산된 목표 평면 Z 크기 (3D, 비율 유지): {target_plane_z_size:.2f}")
        # --- 스케일링 로직 수정 끝 ---

        # 입술 중심의 픽셀 좌표 (회전이 없으므로 변환 불필요)
        final_lips_center_x_in_texture = lips_center_px[0]
        final_lips_center_y_in_texture = lips_center_px[1]
        print(f"[DEBUG] 최종 텍스처 기준 입 중심 픽셀: ({final_lips_center_x_in_texture:.2f}, {final_lips_center_y_in_texture:.2f})")

        # 픽셀 좌표를 텍스처 좌표 (u,v)로 변환 (0.0 ~ 1.0 범위)
        # 픽셀 중심을 고려하여 +0.5
        u_coord = (final_lips_center_x_in_texture + 0.5) / final_w_tex if final_w_tex > 0 else 0.5
        v_coord = (final_lips_center_y_in_texture + 0.5) / final_h_tex if final_h_tex > 0 else 0.5
        
        u_coord = np.clip(u_coord, 0.0, 1.0)
        v_coord = np.clip(v_coord, 0.0, 1.0)
        print(f"[DEBUG] 최종 입 중심 텍스처 좌표 (u,v): ({u_coord:.4f}, {v_coord:.4f})")
        
        if np.isnan(u_coord) or np.isnan(v_coord):
            print("[WARNING] u 또는 v가 NaN입니다! 기본값(0.5, 0.5)으로 설정합니다.")
            u_coord = 0.5
            v_coord = 0.5

        # 반환값 순서: final_texture_image, plane_x_size, plane_z_size, u, v
        return final_image_for_texture, target_plane_x_size, target_plane_z_size, u_coord, v_coord

    def __create_and_position_image_plane(self, p_x_size, p_z_size, tex_u, tex_v):
        """
        Open3D를 사용하여 이미지 평면을 생성하고 위치를 조정합니다.
        
        Args:
            p_x_size: 평면의 X축 크기
            p_z_size: 평면의 Z축 크기  
            tex_u: 텍스처 U 좌표 (입술 중심)
            tex_v: 텍스처 V 좌표 (입술 중심)
        
        Returns:
            o3d.geometry.TriangleMesh: 위치가 조정된 평면 메시
        """
        # 평면 메시를 직접 생성 (2개의 삼각형으로 사각형 구성)
        vertices = np.array([
            [-p_x_size/2, 0, -p_z_size/2],  # 좌하단
            [p_x_size/2, 0, -p_z_size/2],   # 우하단  
            [p_x_size/2, 0, p_z_size/2],    # 우상단
            [-p_x_size/2, 0, p_z_size/2]    # 좌상단
        ])
        
        # 두 개의 삼각형으로 사각형 구성
        triangles = np.array([
            [0, 1, 2],  # 첫 번째 삼각형
            [0, 2, 3]   # 두 번째 삼각형
        ])
        
        # TriangleMesh 생성
        plane_mesh = o3d.geometry.TriangleMesh()
        plane_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        plane_mesh.triangles = o3d.utility.Vector3iVector(triangles)
        
        # 법선 벡터 계산 (조명용)
        plane_mesh.compute_vertex_normals()
        
        print(f"[DEBUG] 생성된 평면 크기: X={p_x_size:.2f}, Z={p_z_size:.2f}")
        print(f"[DEBUG] 평면 정점 좌표:")
        for i, vertex in enumerate(vertices):
            print(f"  정점 {i}: ({vertex[0]:.2f}, {vertex[1]:.2f}, {vertex[2]:.2f})")
        
        # 평면의 로컬 좌표계에서 입술 중심의 위치 계산
        # u는 좌->우 (0->1), v는 상->하 (0->1) 기준
        # X축: u=0 -> -half_x, u=1 -> +half_x
        # Z축: v=0 -> +half_z (텍스처 상단), v=1 -> -half_z (텍스처 하단)
        plane_i_half = p_x_size / 2.0
        plane_j_half = p_z_size / 2.0
        
        # u, v 좌표를 평면의 로컬 x, z 좌표로 변환
        # 텍스처가 좌우반전되었으므로, 입 중심의 u좌표 또한 (1.0 - u)로 반전하여 계산해야 합니다.
        local_x = plane_i_half * ((1.0 - tex_u) - 0.5) * 2
        local_z = plane_j_half * (0.5 - tex_v) * 2
        
        mouth_center_on_plane_local = np.array([local_x, 0.0, local_z])
        print(f"[DEBUG] 이동 전 계산된 입 중심 3D 월드 좌표 (평면 로컬): {mouth_center_on_plane_local}")
        print(f"[DEBUG] 텍스처 좌표 (u,v): ({tex_u:.4f}, {tex_v:.4f})")
        
        if np.any(np.isnan(mouth_center_on_plane_local)):
            print("[WARNING] mouth_center_on_plane_local에 NaN 포함. 이동 벡터 (0,0,0)으로 설정합니다.")
            translation_vector = np.array([0.0, 0.0, 0.0])
        else:
            translation_vector = -mouth_center_on_plane_local  # 이 점을 원점으로 옮기기 위한 벡터
        
        print(f"[DEBUG] 이미지 평면 이동 벡터: {translation_vector}")
        print(f"[DEBUG] 이동 전 평면 중심: {plane_mesh.get_center()}")
        
        # 평면 이동
        plane_mesh.translate(translation_vector)
        
        print(f"[DEBUG] 이동 후 평면 중심: {plane_mesh.get_center()}")
        
        # 경계 상자 정보 출력
        bbox = plane_mesh.get_axis_aligned_bounding_box()
        bbox_min = bbox.get_min_bound()
        bbox_max = bbox.get_max_bound()
        bbox_size = bbox_max - bbox_min
        print(f"[DEBUG] 이동 후 평면 경계 상자:")
        print(f"  최소값: ({bbox_min[0]:.2f}, {bbox_min[1]:.2f}, {bbox_min[2]:.2f})")
        print(f"  최대값: ({bbox_max[0]:.2f}, {bbox_max[1]:.2f}, {bbox_max[2]:.2f})")
        print(f"  크기: ({bbox_size[0]:.2f}, {bbox_size[1]:.2f}, {bbox_size[2]:.2f})")
        
        # 평면이 너무 작거나 큰지 확인
        max_size = np.max(bbox_size)
        if max_size < 1.0:
            print(f"[WARNING] 평면이 매우 작습니다 (최대 크기: {max_size:.2f}). 시각화 시 보이지 않을 수 있습니다.")
        elif max_size > 1000.0:
            print(f"[WARNING] 평면이 매우 큽니다 (최대 크기: {max_size:.2f}). 시각화 시 카메라 조정이 필요할 수 있습니다.")
        
        # 텍스처 이미지가 있다면 UV 좌표 설정 및 텍스처 적용
        if hasattr(self, 'final_texture_image'):
            try:
                print(f"[DEBUG] 텍스처 이미지 크기: {self.final_texture_image.shape}")
                
                # UV 좌표를 메시에 적용
                plane_mesh.triangle_uvs = o3d.utility.Vector2dVector(
                    np.array([
                        # 이미지가 좌우 반전되는 문제를 해결하기 위해 U 좌표(첫 번째 값)를 뒤집습니다. (0.0 -> 1.0, 1.0 -> 0.0)
                        [1.0, 1.0], [0.0, 1.0], [0.0, 0.0],  # 첫 번째 삼각형의 UV
                        [1.0, 1.0], [0.0, 0.0], [1.0, 0.0]   # 두 번째 삼각형의 UV
                    ])
                )
                print(f"[DEBUG] UV 좌표 설정 완료")
                
                # 텍스처 이미지를 Open3D 이미지로 직접 변환
                print(f"[DEBUG] 텍스처를 메모리에서 직접 적용")
                
                # 텍스처 이미지 전처리
                texture_array = self.final_texture_image.copy()
                print(f"[DEBUG] 원본 텍스처 데이터 타입: {texture_array.dtype}")
                print(f"[DEBUG] 원본 텍스처 값 범위: {texture_array.min()} ~ {texture_array.max()}")
                
                # 데이터 타입이 float이고 범위가 0-1이면 0-255로 변환
                if texture_array.dtype == np.float64 or texture_array.dtype == np.float32:
                    if texture_array.max() <= 1.0:
                        texture_array = (texture_array * 255).astype(np.uint8)
                        print("[DEBUG] 텍스처를 0-1 범위에서 0-255 범위로 변환")
                elif texture_array.dtype != np.uint8:
                    texture_array = texture_array.astype(np.uint8)
                    print(f"[DEBUG] 텍스처를 {texture_array.dtype}로 변환")
                
                print(f"[DEBUG] 최종 텍스처 데이터 타입: {texture_array.dtype}")
                print(f"[DEBUG] 최종 텍스처 값 범위: {texture_array.min()} ~ {texture_array.max()}")
                
                # RGB를 그대로 사용 (BGR 변환 제거)
                texture_image = o3d.geometry.Image(texture_array)
                
                print(f"[DEBUG] 텍스처 이미지 생성 성공")
                img_array = np.asarray(texture_image)
                print(f"[DEBUG] 텍스처 이미지 크기: {img_array.shape}")
                print(f"[DEBUG] 텍스처 이미지 데이터 타입: {img_array.dtype}")
                print(f"[DEBUG] 텍스처 이미지 값 범위: {img_array.min()} ~ {img_array.max()}")
                
                # 텍스처를 메시에 적용
                plane_mesh.textures = [texture_image]
                
                # triangle_material_ids 설정 (각 삼각형이 어떤 텍스처를 사용할지)
                plane_mesh.triangle_material_ids = o3d.utility.IntVector([0, 0])  # 두 삼각형 모두 첫 번째 텍스처 사용
                
                print(f"[DEBUG] 텍스처가 메시에 적용됨")
                print(f"[DEBUG] 메시 텍스처 개수: {len(plane_mesh.textures)}")
                print(f"[DEBUG] 메시 UV 좌표 개수: {len(plane_mesh.triangle_uvs)}")
                print(f"[DEBUG] 메시 재질 ID 개수: {len(plane_mesh.triangle_material_ids)}")
                
            except Exception as e:
                print(f"[ERROR] 텍스처 처리 중 오류: {e}")
                import traceback
                traceback.print_exc()
                # 오류 시 파란색으로 표시
                plane_mesh.paint_uniform_color([0.0, 0.0, 1.0])
                
        else:
            print("[DEBUG] 텍스처 이미지가 없음 - 회색으로 표시")
            # 텍스처가 없으면 색상으로 대체
            plane_mesh.paint_uniform_color([0.8, 0.8, 0.8])  # 회색
        
        return plane_mesh
    
    def __visualize_image_plane(self, image_plane):
        """
        Open3D를 사용하여 이미지 평면을 시각화합니다.
        """
        try:
            print("[DEBUG] 시각화 시작...")
            print(f"[DEBUG] 메시 정점 수: {len(image_plane.vertices)}")
            print(f"[DEBUG] 메시 삼각형 수: {len(image_plane.triangles)}")
            print(f"[DEBUG] 메시에 텍스처가 있는지: {hasattr(image_plane, 'textures') and len(image_plane.textures) > 0}")
            
            # 좌표축 추가 (크기를 메시에 맞게 조정)
            bbox = image_plane.get_axis_aligned_bounding_box()
            bbox_size = bbox.get_max_bound() - bbox.get_min_bound()
            coord_size = 50 #max(np.max(bbox_size) * 0.3, 20.0)  # 메시 크기의 30% 또는 최소 20
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=coord_size)
            
            print("[DEBUG] 시각화 창 생성 중...")
            
            # 시각화 설정을 더 명시적으로 구성
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="Face Photo Registration - Image Plane", width=1200, height=800)
            
            # 기하학적 객체들 추가
            vis.add_geometry(image_plane)
            vis.add_geometry(coordinate_frame)
            
            # 렌더링 옵션 설정
            render_option = vis.get_render_option()
            render_option.background_color = np.array([0.2, 0.2, 0.2])  # 어두운 회색 배경
            render_option.light_on = True
            render_option.mesh_show_back_face = True  # 뒷면도 보이게
            render_option.mesh_show_wireframe = False  # 와이어프레임 끄기
            
            # 텍스처 렌더링 활성화
            try:
                if hasattr(image_plane, 'textures') and len(image_plane.textures) > 0:
                    print("[DEBUG] 텍스처 렌더링 모드 활성화")
                    # 텍스처가 있을 때는 텍스처를 우선 렌더링
                    render_option.mesh_color_option = o3d.visualization.MeshColorOption.Color
                else:
                    print("[DEBUG] 단색 렌더링 모드")
                    render_option.mesh_color_option = o3d.visualization.MeshColorOption.Default
            except Exception as render_e:
                print(f"[DEBUG] 렌더링 옵션 설정 중 오류: {render_e}")
                render_option.mesh_color_option = o3d.visualization.MeshColorOption.Default
            
            # 카메라 위치 조정 (평면이 전체적으로 보이도록)
            try:
                bbox_center = bbox.get_center()
                max_dimension = np.max(bbox_size)
                
                print(f"[DEBUG] 메시 경계 상자 크기: {bbox_size}")
                print(f"[DEBUG] 메시 중심: {bbox_center}")
                print(f"[DEBUG] 최대 치수: {max_dimension}")
                
                # 뷰 컨트롤 설정
                view_control = vis.get_view_control()
                
                # 평면이 XZ 평면에 있으므로 Y축 위에서 내려다보도록 설정
                view_control.set_front([0, -1, -0.5])  # 약간 비스듬히 내려다보기
                view_control.set_lookat(bbox_center)
                view_control.set_up([0, 0, 1])  # Z축이 위쪽
                
                # 적절한 거리로 줌 설정
                view_control.set_zoom(0.4)
                
            except Exception as cam_e:
                print(f"[DEBUG] 카메라 설정 중 오류: {cam_e}")
                import traceback
                traceback.print_exc()
            
            print("[DEBUG] 시각화 창이 생성되었습니다. 창을 닫으려면 ESC 키를 누르거나 창을 닫으세요.")
            
            # 시각화 실행 (창이 닫힐 때까지 대기)
            vis.run()
            vis.destroy_window()
            
            print("[DEBUG] 시각화 종료")
            
        except Exception as e:
            print(f"[ERROR] 시각화 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            
            # 대안: 간단한 시각화 시도
            print("[DEBUG] 대안 시각화 방법 시도...")
            try:
                # 좌표축 크기 재계산
                bbox = image_plane.get_axis_aligned_bounding_box()
                bbox_size = bbox.get_max_bound() - bbox.get_min_bound()
                coord_size = max(np.max(bbox_size) * 0.3, 20.0)
                coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=coord_size)
                
                o3d.visualization.draw_geometries(
                    [image_plane, coordinate_frame],
                    window_name="Face Photo Registration - Simple View",
                    width=1200,
                    height=800,
                    mesh_show_back_face=True
                )
            except Exception as e2:
                print(f"[ERROR] 대안 시각화도 실패: {e2}")
                
                # 최종 대안: 메시 정보만 출력
                print("[INFO] 시각화 실패. 메시 정보:")
                print(f"  - 정점 수: {len(image_plane.vertices)}")
                print(f"  - 삼각형 수: {len(image_plane.triangles)}")
                bbox = image_plane.get_axis_aligned_bounding_box()
                print(f"  - 경계 상자: min={bbox.get_min_bound()}, max={bbox.get_max_bound()}")
                print(f"  - 중심점: {image_plane.get_center()}")
                
                if hasattr(image_plane, 'textures') and len(image_plane.textures) > 0:
                    print(f"  - 텍스처 개수: {len(image_plane.textures)}")
                    print(f"  - UV 좌표 개수: {len(image_plane.triangle_uvs) if hasattr(image_plane, 'triangle_uvs') else 0}")
                    
                # 텍스처 이미지를 별도로 저장하여 확인할 수 있도록
                if hasattr(self, 'final_texture_image'):
                    debug_texture_path = os.path.join(os.path.dirname(self.photo_path), "debug_texture_check.png")
                    texture_bgr = cv2.cvtColor(self.final_texture_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(debug_texture_path, texture_bgr)
                    print(f"  - 디버그용 텍스처 저장됨: {debug_texture_path}")

# def show_image_plane(image_plane):
#     plt.imshow(image_plane)
#     plt.show()
    
if __name__ == "__main__":
    try:
        print("[INFO] 얼굴 사진 등록 시작...")
        photo_path = "../../example/data/photo/face10.jpg"
        
        # 파일 존재 확인
        if not os.path.exists(photo_path):
            print(f"[ERROR] 사진 파일을 찾을 수 없습니다: {photo_path}")
            print(f"[INFO] 현재 작업 디렉토리: {os.getcwd()}")
            exit(1)
        
        print(f"[INFO] 사진 파일 로드: {photo_path}")
        face_photo_registration = FacePhotoRegistration(photo_path, visualization=True)
        M_eye_align, image_plane = face_photo_registration.run_registration()
        
        print(f"\n[INFO] 처리 완료!")
        print(f"[INFO] 변환 행렬:\n{M_eye_align}")
        print(f"[INFO] 메시 정점 개수: {len(image_plane.vertices)}")
        print(f"[INFO] 메시 삼각형 개수: {len(image_plane.triangles)}")
        
    except Exception as e:
        print(f"[ERROR] 메인 처리 중 오류: {e}")
        import traceback
        traceback.print_exc()
