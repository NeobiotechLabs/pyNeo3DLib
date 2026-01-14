import mediapipe as mp
import cv2
import numpy as np
from PIL import Image
import os


class MouthEraserForFace:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 입술 내부 랜드마크 인덱스 (MediaPipe Face Mesh의 입술 내부 좌표)
        # 윗입술 내부와 아랫입술 내부를 연결하는 폐곡선
        self.LIPS_INNER = [
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
            308, 324, 318, 402, 317, 14, 87, 178, 88, 95
        ]
        
    def erase_mouth(self, image_path):
        """
        이미지에서 입술 내부를 투명하게 만들어 저장합니다.
        
        Args:
            image_path (str): 입력 이미지 경로 (jpg, png)
        
        Returns:
            numpy.ndarray: 투명 처리된 이미지
        """
        try:
            # 이미지 읽기
            image_rgba = self.__read_image_file(image_path)
            if image_rgba is None:
                print(f"이미지를 읽을 수 없습니다: {image_path}")
                return None
            
            # 입술 영역 찾기
            mouth_points = self.__find_mouth(image_rgba)
            if mouth_points is None:
                print("입술을 찾을 수 없습니다.")
                return None
            
            # 디버그 모드: 입술 랜드마크 시각화
            # 입술 내부를 투명하게 만들기
            result_image = self.__make_mouth_transparent(image_rgba, mouth_points)
            
            return result_image
            
        except Exception as e:
            print(f"오류가 발생했습니다: {str(e)}")
            return False
    
    def __find_mouth(self, image_rgba):
        """
        이미지에서 입술 내부 좌표를 찾습니다.
        
        Args:
            image_rgba (numpy.ndarray): RGBA 이미지 배열
            
        Returns:
            numpy.ndarray or None: 입술 내부 좌표 배열 또는 None
        """
        try:
            # RGBA를 RGB로 변환 (MediaPipe는 RGB를 사용)
            image_rgb = cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2RGB)
            
            # MediaPipe로 얼굴 랜드마크 감지
            results = self.face_mesh.process(image_rgb)
            
            if not results.multi_face_landmarks:
                return None
            
            # 첫 번째 얼굴의 랜드마크 사용
            face_landmarks = results.multi_face_landmarks[0]
            
            # 이미지 크기
            height, width = image_rgba.shape[:2]
            
            # 입술 내부 좌표 추출
            mouth_points = []
            for idx in self.LIPS_INNER:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                mouth_points.append([x, y])
            
            return np.array(mouth_points, dtype=np.int32)
            
        except Exception as e:
            print(f"입술 찾기 중 오류: {str(e)}")
            return None
    
    def __make_mouth_transparent(self, image_rgba, mouth_points):
        """
        입술 내부 영역을 투명하게 만듭니다.
        
        Args:
            image_rgba (numpy.ndarray): RGBA 이미지 배열
            mouth_points (numpy.ndarray): 입술 내부 좌표 배열
            
        Returns:
            numpy.ndarray: 투명 처리된 이미지
        """
        result_image = image_rgba.copy()
        
        # 입술 내부 영역에 대한 마스크 생성
        mask = np.zeros(image_rgba.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [mouth_points], 255)
        
        # 마스크가 적용된 영역의 알파 채널을 0으로 설정 (완전 투명)
        result_image[mask == 255, 3] = 50
        
        return result_image
    
    def __save_debug_image(self, image_rgba, mouth_points, debug_path):
        """
        입술 랜드마크를 시각화한 디버그 이미지를 저장합니다.
        
        Args:
            image_rgba (numpy.ndarray): RGBA 이미지 배열
            mouth_points (numpy.ndarray): 입술 내부 좌표 배열
            debug_path (str): 디버그 이미지 저장 경로
        """
        try:
            # 이미지 복사
            debug_image = image_rgba.copy()
            
            # RGB로 변환 (OpenCV 그리기용)
            debug_rgb = cv2.cvtColor(debug_image, cv2.COLOR_RGBA2RGB)
            
            # 입술 랜드마크 점들을 빨간색으로 표시
            for point in mouth_points:
                cv2.circle(debug_rgb, tuple(point), 2, (255, 0, 0), -1)
            
            # 입술 내부 영역을 선으로 연결
            cv2.polylines(debug_rgb, [mouth_points], True, (0, 255, 0), 1)
            
            # 입술 내부 영역을 반투명하게 채우기
            mask = np.zeros(debug_rgb.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [mouth_points], 255)
            debug_rgb[mask == 255] = debug_rgb[mask == 255] * 0.7 + np.array([0, 0, 255]) * 0.3
            
            # RGBA로 다시 변환
            debug_rgba = cv2.cvtColor(debug_rgb.astype(np.uint8), cv2.COLOR_RGB2RGBA)
            debug_rgba[:, :, 3] = debug_image[:, :, 3]  # 원본 알파 채널 유지
            
            # 디버그 이미지 저장
            debug_pil = Image.fromarray(debug_rgba, 'RGBA')
            debug_pil.save(debug_path, 'PNG')
            
            print(f"디버그 이미지가 저장되었습니다: {debug_path}")
            
        except Exception as e:
            print(f"디버그 이미지 저장 오류: {str(e)}")
    
    def __read_image_file(self, image_path):
        """
        이미지 파일을 RGBA 형식으로 읽습니다.
        PLY/OBJ 파일인 경우 같은 디렉토리에서 텍스처 파일을 찾습니다.
        
        Args:
            image_path (str): 이미지 파일 경로 또는 3D 메시 파일 경로
            
        Returns:
            numpy.ndarray or None: RGBA 이미지 배열 또는 None
        """
        try:
            if not os.path.exists(image_path):
                print(f"파일이 존재하지 않습니다: {image_path}")
                return None
            
            # PLY/OBJ 파일인 경우 텍스처 파일 경로를 찾음
            actual_image_path = image_path
            if image_path.lower().endswith('.ply'):
                actual_image_path = self.__find_texture_from_ply(image_path)
            elif image_path.lower().endswith('.obj'):
                actual_image_path = self.__find_texture_from_obj(image_path)
            
            if actual_image_path is None:
                print(f"텍스처 파일을 찾을 수 없습니다: {image_path}")
                return None
            
            print(f"텍스처 이미지 로드: {actual_image_path}")
            
            # PIL로 이미지 읽기
            with Image.open(actual_image_path) as img:
                # RGBA로 변환 (이미 RGBA면 그대로, RGB면 알파 채널 추가)
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                
                # numpy 배열로 변환
                image_array = np.array(img)
                
                return image_array
                
        except Exception as e:
            print(f"이미지 읽기 오류: {str(e)}")
            return None
    
    def __find_texture_from_ply(self, ply_path):
        """
        PLY 파일에서 텍스처 파일 경로를 찾습니다.
        
        Args:
            ply_path (str): PLY 파일 경로
            
        Returns:
            str or None: 텍스처 파일 경로 또는 None
        """
        try:
            texture_filename = None
            
            # PLY 헤더에서 TextureFile 정보 읽기
            with open(ply_path, 'rb') as f:
                for line in f:
                    try:
                        line_str = line.decode('utf-8').strip()
                    except:
                        break  # 바이너리 데이터 시작
                    
                    if line_str.startswith('comment TextureFile'):
                        texture_filename = line_str.replace('comment TextureFile', '').strip()
                        break
                    elif line_str == 'end_header':
                        break
            
            if texture_filename:
                # PLY 파일과 같은 디렉토리에서 텍스처 파일 찾기
                ply_dir = os.path.dirname(ply_path)
                texture_path = os.path.join(ply_dir, texture_filename)
                
                if os.path.exists(texture_path):
                    return texture_path
                else:
                    print(f"텍스처 파일이 존재하지 않습니다: {texture_path}")
            
            # TextureFile 정보가 없으면 같은 이름의 이미지 파일 찾기
            ply_dir = os.path.dirname(ply_path)
            ply_name = os.path.splitext(os.path.basename(ply_path))[0]
            
            for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
                texture_path = os.path.join(ply_dir, ply_name + ext)
                if os.path.exists(texture_path):
                    return texture_path
            
            return None
            
        except Exception as e:
            print(f"PLY 텍스처 파일 찾기 오류: {str(e)}")
            return None
    
    def __find_texture_from_obj(self, obj_path):
        """
        OBJ 파일에서 텍스처 파일 경로를 찾습니다.
        
        Args:
            obj_path (str): OBJ 파일 경로
            
        Returns:
            str or None: 텍스처 파일 경로 또는 None
        """
        try:
            obj_dir = os.path.dirname(obj_path)
            obj_name = os.path.splitext(os.path.basename(obj_path))[0]
            
            # MTL 파일에서 텍스처 찾기
            mtl_path = os.path.join(obj_dir, obj_name + '.mtl')
            if os.path.exists(mtl_path):
                with open(mtl_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip().startswith('map_Kd'):
                            texture_filename = line.strip().split()[-1]
                            texture_path = os.path.join(obj_dir, texture_filename)
                            if os.path.exists(texture_path):
                                return texture_path
            
            # 같은 이름의 이미지 파일 찾기
            for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
                texture_path = os.path.join(obj_dir, obj_name + ext)
                if os.path.exists(texture_path):
                    return texture_path
            
            return None
            
        except Exception as e:
            print(f"OBJ 텍스처 파일 찾기 오류: {str(e)}")
            return None
    

