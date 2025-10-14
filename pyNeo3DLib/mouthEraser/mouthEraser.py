import mediapipe as mp
import cv2
import numpy as np
from PIL import Image
import os


class MouthEraser:
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
        
    def erase_mouth(self, image_path, target_path=None, debug=False):
        """
        이미지에서 입술 내부를 투명하게 만들어 저장합니다.
        
        Args:
            image_path (str): 입력 이미지 경로 (jpg, png)
            target_path (str): 출력 이미지 경로 (png)
            debug (bool): 디버그 모드 (입술 랜드마크를 시각화)
        
        Returns:
            bool: 성공 여부
        """
        try:
            if target_path is None:
                target_path = image_path
                
            # 이미지 읽기
            image_rgba = self.__read_image_file(image_path)
            if image_rgba is None:
                print(f"이미지를 읽을 수 없습니다: {image_path}")
                return False
            
            # 입술 영역 찾기
            mouth_points = self.__find_mouth(image_rgba)
            if mouth_points is None:
                print("입술을 찾을 수 없습니다.")
                return False
            
            # 디버그 모드: 입술 랜드마크 시각화
            if debug:
                self.__save_debug_image(image_rgba, mouth_points, target_path.replace('.png', '_debug.png'))
            
            # 입술 내부를 투명하게 만들기
            result_image = self.__make_mouth_transparent(image_rgba, mouth_points)
            
            # 결과 이미지 저장
            success = self.__write_image_file(result_image, target_path)
            
            if success:
                print(f"입술이 지워진 이미지가 저장되었습니다: {target_path}")
            
            return success
            
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
        
        Args:
            image_path (str): 이미지 파일 경로
            
        Returns:
            numpy.ndarray or None: RGBA 이미지 배열 또는 None
        """
        try:
            if not os.path.exists(image_path):
                print(f"파일이 존재하지 않습니다: {image_path}")
                return None
            
            # PIL로 이미지 읽기
            with Image.open(image_path) as img:
                # RGBA로 변환 (이미 RGBA면 그대로, RGB면 알파 채널 추가)
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                
                # numpy 배열로 변환
                image_array = np.array(img)
                
                return image_array
                
        except Exception as e:
            print(f"이미지 읽기 오류: {str(e)}")
            return None
    
    def __write_image_file(self, image_data, target_path):
        """
        이미지 데이터를 PNG 파일로 저장합니다.
        
        Args:
            image_data (numpy.ndarray): RGBA 이미지 배열
            target_path (str): 저장할 파일 경로
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            # 디렉토리가 없으면 생성
            target_dir = os.path.dirname(target_path)
            if target_dir and not os.path.exists(target_dir):
                os.makedirs(target_dir)
            
            # PIL Image로 변환하여 저장
            image = Image.fromarray(image_data, 'RGBA')
            image.save(target_path, 'PNG')
            
            return True
            
        except Exception as e:
            print(f"이미지 저장 오류: {str(e)}")
            return False


if __name__ == "__main__":
    mouth_eraser = MouthEraser()
    # 디버그 모드로 실행하여 입술 랜드마크 확인
    mouth_eraser.erase_mouth("../../example/data/photo/su1.png", "../../example/data/photo/su1_erased.png", debug=True)
    # mouth_eraser.erase_mouth("../../example/data/photo/Smile.png", "../../example/data/photo/Smile.png", debug=True)