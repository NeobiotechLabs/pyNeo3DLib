import json
import numpy as np
import asyncio
import os
from PIL import Image
import io
import base64
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path

# Lazy import: 실제 사용 시점에만 import (mediapipe 의존성 회피)
# from pyNeo3DLib.iosRegistration.iosLaminateRegistration import IOSLaminateRegistration
# from pyNeo3DLib.faceRegisration.faceLaminateRegistration import FaceLaminateRegistration  # mediapipe 필요
# from pyNeo3DLib.faceRegisration.facePhotoRegistration import FacePhotoRegistration  # mediapipe 필요
# from pyNeo3DLib.faceRegisration.facesRegistration import FacesRegistration  # mediapipe 필요
# from pyNeo3DLib.bowRegistration.iosBowRegistration import IOSBowRegistration
# from pyNeo3DLib.condyleFinder.condyleFinder import CondyleFinder
# from pyNeo3DLib.smileArchOuterline.core import analyze_upper_IOS_scandata
# from pyNeo3DLib.faceRegisration.faceAlign import FaceAlignment3D  # mediapipe 필요
# from pyNeo3DLib.goldenProportion.goldenProportionFinder import GoldenProportionFinder
# from pyNeo3DLib.mouthEraser.mouthEraser import MouthEraser  # mediapipe 필요


class RegistrationConstants:
    """등록 작업에 사용되는 상수들"""
    LAMINATE_PATH = os.path.join(os.path.dirname(__file__), "smile_arch_half.stl")
    CENTERPIN_PATH = os.path.join(os.path.dirname(__file__), "center_pin.stl")
    TOTAL_PROGRESS_STEPS = 8
    WEBSOCKET_SLEEP_DURATION = 0.1
    IDENTITY_MATRIX = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


@dataclass
class IOSData:
    """IOS 데이터 구조"""
    sub_type: str  # "smileArch", "upper", "lower"
    path: str
    
    def __post_init__(self):
        """데이터 검증"""
        valid_subtypes = {"smileArch", "upper", "lower"}
        if self.sub_type not in valid_subtypes:
            raise ValueError(f"Invalid IOS subType: {self.sub_type}. Must be one of {valid_subtypes}")
        
        # 경로가 비어있을 수 있음 - 실제 사용 시에만 검증
        # if not self.path:
        #     raise ValueError("IOS path cannot be empty")
    
    def is_valid_for_processing(self) -> bool:
        """실제 처리에 사용할 수 있는지 확인"""
        return bool(self.path and self.path.strip())


@dataclass  
class FaceScanData:
    """FaceScan 데이터 구조"""
    sub_type: str  # "faceSmile", "faceRest", "faceRetraction"
    path: str
    
    def __post_init__(self):
        """데이터 검증"""
        valid_subtypes = {"faceSmile", "faceRest", "faceRetraction"}
        if self.sub_type not in valid_subtypes:
            raise ValueError(f"Invalid FaceScan subType: {self.sub_type}. Must be one of {valid_subtypes}")
        
        # 경로가 비어있을 수 있음 - 실제 사용 시에만 검증
        # if not self.path:
        #     raise ValueError("FaceScan path cannot be empty")
    
    def is_valid_for_processing(self) -> bool:
        """실제 처리에 사용할 수 있는지 확인"""
        return bool(self.path and self.path.strip())


@dataclass
class CBCTData:
    """CBCT 데이터 구조"""
    path: str
    
    def __post_init__(self):
        """데이터 검증"""
        # 경로가 비어있을 수 있음 - 실제 사용 시에만 검증
        # if not self.path:
        #     raise ValueError("CBCT path cannot be empty")
        pass
    
    def is_valid_for_processing(self) -> bool:
        """실제 처리에 사용할 수 있는지 확인"""
        return bool(self.path and self.path.strip())


@dataclass
class SmileArchBowData:
    """SmileArch Bow 데이터 구조"""
    path: str
    
    def __post_init__(self):
        """데이터 검증"""
        # 경로가 비어있을 수 있음 - 실제 사용 시에만 검증
        # if not self.path:
        #     raise ValueError("SmileArch Bow path cannot be empty")
        pass
    
    def is_valid_for_processing(self) -> bool:
        """실제 처리에 사용할 수 있는지 확인"""
        return bool(self.path and self.path.strip())


@dataclass
class RegistrationConfig:
    """등록 작업 설정을 담는 데이터 클래스"""
    ios_data: List[IOSData]
    facescan_data: List[FaceScanData] 
    cbct_data: CBCTData
    smilearch_bow_data: Optional[SmileArchBowData] = None
    
    def __post_init__(self):
        """설정 검증"""
        if not self.ios_data:
            raise ValueError("At least one IOS data is required")
        
        if not self.facescan_data:
            raise ValueError("At least one FaceScan data is required")
        
        # 필수 IOS subType 확인
        ios_subtypes = {ios.sub_type for ios in self.ios_data}
        required_subtypes = {"smileArch"}
        missing_subtypes = required_subtypes - ios_subtypes
        if missing_subtypes:
            raise ValueError(f"Missing required IOS subTypes: {missing_subtypes}")
        
        # 필수 FaceScan subType 확인  
        facescan_subtypes = {face.sub_type for face in self.facescan_data}
        required_face_subtypes = {"faceSmile"}
        missing_face_subtypes = required_face_subtypes - facescan_subtypes
        if missing_face_subtypes:
            raise ValueError(f"Missing required FaceScan subTypes: {missing_face_subtypes}")
    
    def get_ios_by_subtype(self, sub_type: str) -> Optional[IOSData]:
        """subType으로 IOS 데이터 찾기"""
        for ios in self.ios_data:
            if ios.sub_type == sub_type:
                return ios
        return None
    
    def get_facescan_by_subtype(self, sub_type: str) -> Optional[FaceScanData]:
        """subType으로 FaceScan 데이터 찾기"""
        for face in self.facescan_data:
            if face.sub_type == sub_type:
                return face
        return None


@dataclass
class ProgressEvent:
    """진행 상황 이벤트를 나타내는 데이터 클래스"""
    type: str
    progress: float
    message: str
    
    def __str__(self) -> str:
        return f"ProgressEvent(type={self.type}, progress={self.progress}, message={self.message})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def get_json(self) -> Dict[str, Any]:
        """JSON 형태로 변환"""
        return {
            "type": self.type,
            "progress": self.progress,
            "message": self.message
        }


class ProgressReporter:
    """WebSocket을 통한 진행 상황 보고를 담당하는 클래스"""
    
    def __init__(self, websocket=None, total_steps: int = RegistrationConstants.TOTAL_PROGRESS_STEPS):
        self.websocket = websocket
        self.total_steps = total_steps
        self.current_step = 0
    
    async def report_progress(self, message: str) -> None:
        """진행 상황을 보고합니다"""
        if self.websocket is not None:
            progress = (self.current_step / self.total_steps) * 100
            event = ProgressEvent("progress", progress, message)
            await self.websocket.send_json(event.get_json())
            await asyncio.sleep(RegistrationConstants.WEBSOCKET_SLEEP_DURATION)
        self.current_step += 1
    
    async def report_completion(self, result: Any = None) -> None:
        """완료 상황을 보고합니다"""
        if self.websocket is not None:
            completion_event = ProgressEvent("progress", 100, "All registration completed")
            await self.websocket.send_json(completion_event.get_json())
            
            if result is not None:
                result_event = ProgressEvent("result", 100, result)
                await self.websocket.send_json(result_event.get_json())
            
            await asyncio.sleep(RegistrationConstants.WEBSOCKET_SLEEP_DURATION)
    

class ConfigParser:
    """설정 파싱 및 검증을 담당하는 클래스"""
    
    @staticmethod
    def parse_json(json_string: str) -> Dict[str, Any]:
        """JSON 문자열을 파싱합니다"""
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        except Exception as e:
            raise ValueError(f"Failed to parse JSON: {e}")
    
    @staticmethod
    def create_config(json_data: Dict[str, Any]) -> RegistrationConfig:
        """JSON 데이터로부터 RegistrationConfig 객체를 생성합니다"""
        try:
            # IOS 데이터 파싱
            ios_list = json_data.get("ios", [])
            if not isinstance(ios_list, list):
                raise ValueError("'ios' must be a list")
            
            ios_data = []
            for ios_item in ios_list:
                if not isinstance(ios_item, dict):
                    raise ValueError("Each IOS item must be a dictionary")
                
                ios_data.append(IOSData(
                    sub_type=ios_item.get("subType", ""),
                    path=ios_item.get("path", None)
                ))
            
            # FaceScan 데이터 파싱
            facescan_list = json_data.get("facescan", [])
            if not isinstance(facescan_list, list):
                raise ValueError("'facescan' must be a list")
            
            facescan_data = []
            for face_item in facescan_list:
                if not isinstance(face_item, dict):
                    raise ValueError("Each FaceScan item must be a dictionary")
                
                facescan_data.append(FaceScanData(
                    sub_type=face_item.get("subType", ""),
                    path=face_item.get("path", None)
                ))
            
            # CBCT 데이터 파싱
            cbct_dict = json_data.get("cbct", {})
            if not isinstance(cbct_dict, dict):
                raise ValueError("'cbct' must be a dictionary")
            
            cbct_data = CBCTData(path=cbct_dict.get("path", None))
            
            # SmileArch Bow 데이터 파싱 (선택사항)
            smilearch_bow_data = None
            if "smilearch_bow" in json_data:
                bow_dict = json_data["smilearch_bow"]
                if not isinstance(bow_dict, dict):
                    raise ValueError("'smilearch_bow' must be a dictionary")
                
                smilearch_bow_data = SmileArchBowData(path=bow_dict.get("path", None))
            
            return RegistrationConfig(
                ios_data=ios_data,
                facescan_data=facescan_data,
                cbct_data=cbct_data,
                smilearch_bow_data=smilearch_bow_data
            )
            
        except Exception as e:
            raise ValueError(f"Failed to create configuration: {e}")
    
    @staticmethod
    def validate_file_paths(config: RegistrationConfig, check_existence: bool = True) -> None:
        """파일 경로들의 유효성을 검증합니다"""
        if not check_existence:
            return
        
        # IOS 파일들 확인
        for ios in config.ios_data:
            if not Path(ios.path).exists():
                raise FileNotFoundError(f"IOS file not found: {ios.path}")
        
        # FaceScan 파일들 확인
        for face in config.facescan_data:
            if not Path(face.path).exists():
                raise FileNotFoundError(f"FaceScan file not found: {face.path}")
        
        # CBCT 경로 확인 (디렉토리일 수 있음)
        cbct_path = Path(config.cbct_data.path)
        if not (cbct_path.exists() and (cbct_path.is_file() or cbct_path.is_dir())):
            raise FileNotFoundError(f"CBCT path not found: {config.cbct_data.path}")
        
        # SmileArch Bow 파일 확인 (선택사항)
        if config.smilearch_bow_data and not Path(config.smilearch_bow_data.path).exists():
            raise FileNotFoundError(f"SmileArch Bow file not found: {config.smilearch_bow_data.path}")


class Neo3DRegistration:
    def __init__(self, json_string: str, websocket=None, validate_files: bool = False):
        self.version = "0.0.1"
        print(f"json_string: {json_string}")
        
        # JSON 파싱 및 설정 생성
        json_data = ConfigParser.parse_json(json_string)
        self.config = ConfigParser.create_config(json_data)
        
        # 파일 존재 검증 (선택사항)
        if validate_files:
            ConfigParser.validate_file_paths(self.config)
        
        # 기존 호환성을 위해 parsed_json도 유지
        self.parsed_json = json_data
        
        self.websocket = websocket
        self.progress_reporter = ProgressReporter(websocket)
    
    
    async def run_registration(self, visualize=False):       
        # 설정이 이미 검증되었으므로 별도 검증 불필요
        
        # 기본값 초기화 (에러 발생 시 사용)
        ios_laminate_result = np.array(RegistrationConstants.IDENTITY_MATRIX)
        ios_upper_result = np.array(RegistrationConstants.IDENTITY_MATRIX)
        ios_lower_result = np.array(RegistrationConstants.IDENTITY_MATRIX)
        facescan_laminate_result = np.array(RegistrationConstants.IDENTITY_MATRIX)
        transformed_face_smile_mesh = None
        facephoto_meshes = None
        type_of_facedata = None
        facescan_rest_result = np.array(RegistrationConstants.IDENTITY_MATRIX)
        facescan_retraction_result = np.array(RegistrationConstants.IDENTITY_MATRIX)
        cbct_result = np.array(RegistrationConstants.IDENTITY_MATRIX)
        ios_bow_result = np.array(RegistrationConstants.IDENTITY_MATRIX)
        condyle_result = None
        golden_proportion_result = None
        smilearch_outerline_result = None

        # IOS Laminate Registration
        try:
            await self.progress_reporter.report_progress("ios_laminate_registration")
            ios_laminate_result = self.__ios_laminate_registration(visualize=visualize)
            print(f'✅ ios_laminate_registration 성공')
        except Exception as e:
            print(f'❌ ios_laminate_registration 실패: {str(e)}')
            ios_laminate_result = np.array(RegistrationConstants.IDENTITY_MATRIX)

        # IOS Upper Registration
        try:
            await self.progress_reporter.report_progress("ios_upper_registration")
            ios_upper_result = ios_laminate_result #self.__ios_upper_registration()
            print(f'✅ ios_upper_registration 성공')
        except Exception as e:
            print(f'❌ ios_upper_registration 실패: {str(e)}')
            ios_upper_result = np.array(RegistrationConstants.IDENTITY_MATRIX)

        # IOS Lower Registration
        try:
            await self.progress_reporter.report_progress("ios_lower_registration")
            ios_lower_result = ios_laminate_result #self.__ios_lower_registration()
            print(f'✅ ios_lower_registration 성공')
        except Exception as e:
            print(f'❌ ios_lower_registration 실패: {str(e)}')
            ios_lower_result = np.array(RegistrationConstants.IDENTITY_MATRIX)

        # FaceScan Laminate Registration (중요: facephoto_meshes 보존)
        try:
            await self.progress_reporter.report_progress("facescan_laminate_registration")
            
            # FaceScan 인 경우 텍스처 파일을 찾아서 입술 지움, FacePhoto 인 경우 이미지에서 입술 지움
            try:
                self.__erase_mouth()
            except Exception as e:
                print(f'⚠️ __erase_mouth 실패 (계속 진행): {str(e)}')
            
            facescan_laminate_result, transformed_face_smile_mesh, type_of_facedata = self.__facescan_laminate_registration(visualize=visualize)
            print(f'✅ facescan_laminate_registration 성공 (type: {type_of_facedata})')
            
            # FacePhoto인 경우 transformed_face_smile_mesh를 반드시 보존
            if type_of_facedata == "FacePhoto":
                facephoto_meshes = transformed_face_smile_mesh
                print(f'✅ facephoto_meshes 보존 완료')
        except Exception as e:
            print(f'❌ facescan_laminate_registration 실패: {str(e)}')
            facescan_laminate_result = np.array(RegistrationConstants.IDENTITY_MATRIX)
            transformed_face_smile_mesh = None
            type_of_facedata = None

        # FaceScan Rest/Retraction Registration
        if type_of_facedata == "FaceScan":
            try:
                await self.progress_reporter.report_progress("facescan_rest_registration")
                facescan_rest_result, facescan_retraction_result = self.__facescan_rest_registration(transformed_face_smile_mesh, facescan_laminate_result, visualize=visualize)
                facephoto_meshes = None
                print(f'✅ facescan_rest_registration 성공')
            except Exception as e:
                print(f'❌ facescan_rest_registration 실패: {str(e)}')
                facescan_rest_result = np.array(RegistrationConstants.IDENTITY_MATRIX)
                facescan_retraction_result = np.array(RegistrationConstants.IDENTITY_MATRIX)
                facephoto_meshes = None
        elif type_of_facedata == "FacePhoto":
            # FacePhoto는 이미 위에서 facephoto_meshes에 저장됨
            facescan_rest_result = np.array(RegistrationConstants.IDENTITY_MATRIX)
            facescan_retraction_result = np.array(RegistrationConstants.IDENTITY_MATRIX)

        # CBCT Registration
        try:
            await self.progress_reporter.report_progress("cbct_registration")
            cbct_result = self.__cbct_registration()
            print(f'✅ cbct_registration 성공')
        except Exception as e:
            print(f'❌ cbct_registration 실패: {str(e)}')
            cbct_result = np.array(RegistrationConstants.IDENTITY_MATRIX)

        # IOS Bow Registration
        try:
            await self.progress_reporter.report_progress("ios_bow_registration")
            ios_bow_result = np.array(RegistrationConstants.IDENTITY_MATRIX) #self.__ios_bow_registration(ios_laminate_result, visualize=visualize)
            print(f'✅ ios_bow_registration 성공')
        except Exception as e:
            print(f'❌ ios_bow_registration 실패: {str(e)}')
            ios_bow_result = np.array(RegistrationConstants.IDENTITY_MATRIX)
                       
        # SmileArch Outerline Detection (템플릿 검색을 위해 꼭 필요)
        try:
            await self.progress_reporter.report_progress("smilearch_outerline_registration")
            smilearch_outerline_result = self.__smilearch_outerline_detect(visualize)
            print(f'✅ smilearch_outerline_registration 성공')
        except Exception as e:
            print(f'❌ smilearch_outerline_registration 실패: {str(e)}')
            smilearch_outerline_result = None

        # 결과 JSON 생성 (부분 결과라도 반환)
        try:
            result = self.__make_result_json(
                ios_laminate_result.tolist(), ios_upper_result.tolist(), ios_lower_result.tolist(), 
                facescan_laminate_result.tolist(), facephoto_meshes, 
                facescan_rest_result.tolist(), facescan_retraction_result.tolist(), 
                cbct_result.tolist(), ios_bow_result.tolist(), 
                # condyle_result, golden_proportion_result, 
                smilearch_outerline_result
            )
            
            await self.progress_reporter.report_completion(result)
            return result
        except Exception as e:
            print(f'❌ __make_result_json 실패: {str(e)}')
            # 결과 생성 실패 시에도 에러를 상위로 전달
            raise

    def __make_result_json(self, ios_laminate_result, 
                            ios_upper_result, 
                            ios_lower_result, 
                            facescan_laminate_result, 
                            facephoto_meshes,
                            facescan_rest_result, 
                            facescan_retraction_result, 
                            cbct_result, 
                            ios_bow_result,
                            # condyle_result,
                            # golden_proportion_result,
                            smilearch_outerline_result):
        
        print("=====================================")
        print(f'ios_laminate_result: {ios_laminate_result}')
        print(f'ios_upper_result: {ios_upper_result}')
        print(f'ios_lower_result: {ios_lower_result}')

        print(f'facescan_laminate_result: {facescan_laminate_result}')
        print(f'transformed_face_smile_mesh (only for photo): {facephoto_meshes}')
        print(f'facescan_rest_result: {facescan_rest_result}')
        print(f'facescan_retraction_result: {facescan_retraction_result}') 

        print(f'cbct_result: {cbct_result}')

        print(f'ios_bow_result: {ios_bow_result}')
        # print(f'condyle_result: {condyle_result}')
        # print(f'golden_proportion_result: {golden_proportion_result}')
        print(f'smilearch_outerline_result: {smilearch_outerline_result}')
        print("=====================================")

        for ios in self.parsed_json["ios"]:
            if ios["subType"] == "smileArch":
                ios["transform_matrix"] = ios_laminate_result
            elif ios["subType"] == "upper":
                ios["transform_matrix"] = ios_upper_result
            elif ios["subType"] == "lower":
                ios["transform_matrix"] = ios_lower_result
            
        for facescan in self.parsed_json["facescan"]:
            if facescan["subType"] == "faceSmile":
                facescan["transform_matrix"] = facescan_laminate_result
            elif facescan["subType"] == "faceRest":
                facescan["transform_matrix"] = facescan_rest_result
            elif facescan["subType"] == "faceRetraction":
                facescan["transform_matrix"] = facescan_retraction_result

        self.parsed_json["cbct"]["transform_matrix"] = cbct_result
        self.parsed_json["smilearch_bow"]["transform_matrix"] = ios_bow_result
                   
        if smilearch_outerline_result is not None:
            arch_depth, molar_width, landmark_points = smilearch_outerline_result
            smilearch_outerline_json = {
                "arch_depth": arch_depth,
                "molar_width": molar_width,
                "landmark_points": landmark_points
            }
            for ios in self.parsed_json["ios"]:
                if ios["subType"] == "smileArch":
                    ios["outerlineInfo"] = smilearch_outerline_json
                    break
            

        if facephoto_meshes is not None:            
            def mesh_to_json(mesh):
                if not mesh:
                    return None
                
                vertices = np.asarray(mesh.vertices)
                triangles = np.asarray(mesh.triangles)
                
                plane_json = {
                    "vertices": vertices.tolist(),
                    "triangles": triangles.tolist()
                }
                
                if hasattr(mesh, 'triangle_uvs') and len(mesh.triangle_uvs) > 0:
                    plane_json["triangle_uvs"] = np.asarray(mesh.triangle_uvs).tolist()
                
                if hasattr(mesh, 'textures') and len(mesh.textures) > 0:
                    texture = mesh.textures[0]
                    pil_img = Image.fromarray(np.asarray(texture))
                    buffer = io.BytesIO()
                    pil_img.save(buffer, format="PNG")
                    encoded_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    plane_json["texture"] = {
                        "format": "png",
                        "data": encoded_string
                    }
                return plane_json
            
            front_json = mesh_to_json(facephoto_meshes.front_plane)
            right_json = mesh_to_json(facephoto_meshes.right_plane)
            left_json = mesh_to_json(facephoto_meshes.left_plane)
            
            photo_json = {
                "front": front_json,
                "right": right_json,
                "left": left_json
            }
            self.parsed_json["photo"] = photo_json
        return self.parsed_json
        

    def __verify_file_info(self):
        # Check ios
        ios_data = self.parsed_json.get("ios")
        if ios_data is None:
            raise ValueError("ios is not defined")
        
        # Check ios internal data
        for ios in ios_data:
            sub_type = ios.get("subType")
            if sub_type == "smileArch":
                pass
            elif sub_type == "upper":
                pass
            elif sub_type == "lower":
                pass
            else:
                raise ValueError(f"Unknown subType: {sub_type}")

        # Check facescan
        if self.parsed_json.get("facescan") is None:
            raise ValueError("facescan is not defined")
        
        # Check cbct
        if self.parsed_json.get("cbct") is None:
            raise ValueError("cbct is not defined")

    def __ios_laminate_registration(self, visualize=False):
        print("ios_laminate_registration")
        
        # 새로운 설정 객체 사용
        smile_arch_ios = self.config.get_ios_by_subtype("smileArch")
        if not smile_arch_ios:
            raise ValueError("smileArch IOS data not found")
        
        # 실제 처리 시 경로 유효성 확인
        if not smile_arch_ios.is_valid_for_processing():
            print(f"Warning: smileArch IOS path is empty or invalid: '{smile_arch_ios.path}'")
            # 빈 경로인 경우 단위행렬 반환
            return np.array(RegistrationConstants.IDENTITY_MATRIX)
        
        print(f'ios path: {smile_arch_ios.path}')
        # Lazy import
        from pyNeo3DLib.iosRegistration.iosLaminateRegistration import IOSLaminateRegistration
        
        # Now register this file with the laminate model
        ios_laminate_registration = IOSLaminateRegistration(smile_arch_ios.path, RegistrationConstants.LAMINATE_PATH, visualize)
        result_matrix = ios_laminate_registration.run_registration()
        return result_matrix

    def __ios_upper_registration(self):
        print("ios_upper_registration")
        matrix = np.array(RegistrationConstants.IDENTITY_MATRIX)
        return matrix

    def __ios_lower_registration(self):
        print("ios_lower_registration")
        matrix = np.array(RegistrationConstants.IDENTITY_MATRIX)
        return matrix
    
    def __erase_mouth(self):
        """
        FaceScan인 경우 텍스처 파일에서 입술을 지우고, FacePhoto인 경우 이미지에서 입술을 지웁니다.
        """
        try:
            # Lazy import
            from pyNeo3DLib.mouthEraser.mouthEraser import MouthEraser
            
            mouth_eraser = MouthEraser()
            facescan_data = self.parsed_json.get("facescan", [])
            
            for facescan in facescan_data:
                if facescan.get("subType") == "faceSmile":
                    file_path = facescan.get("path", "")
                    
                    if not file_path or not os.path.exists(file_path):
                        print(f"파일을 찾을 수 없습니다: {file_path}")
                        continue
                    
                    # 파일 확장자에 따라 처리 방식 결정
                    if file_path.lower().endswith(('.obj', '.ply')):
                        # FaceScan: 3D 모델의 텍스처 파일 처리
                        self.__erase_mouth_from_texture(file_path, mouth_eraser)
                        
                    elif file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                        # FacePhoto: 이미지 파일 직접 처리
                        self.__erase_mouth_from_image(file_path, mouth_eraser)
                        
                    else:
                        print(f"지원하지 않는 파일 형식입니다: {file_path}")
                        
        except Exception as e:
            print(f"입술 지우기 중 오류 발생: {str(e)}")
    
    def __erase_mouth_from_texture(self, model_path, mouth_eraser):
        """
        3D 모델의 텍스처 파일에서 입술을 지웁니다.
        
        Args:
            model_path (str): 3D 모델 파일 경로 (.obj 또는 .ply)
            mouth_eraser (MouthEraser): MouthEraser 인스턴스
        """
        try:
            # 모델 파일과 같은 디렉토리에서 텍스처 파일 찾기
            model_dir = os.path.dirname(model_path)
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            
            # 일반적인 텍스처 파일 이름 패턴들
            texture_patterns = [
                f"{model_name}.jpg",
                f"{model_name}.png", 
                f"{model_name}_texture.jpg",
                f"{model_name}_texture.png",
                "texture.jpg",
                "texture.png"
            ]
            
            texture_file = None
            for pattern in texture_patterns:
                potential_path = os.path.join(model_dir, pattern)
                if os.path.exists(potential_path):
                    texture_file = potential_path
                    break
            
            if texture_file:
                print(f"텍스처 파일을 찾았습니다: {texture_file}")
                # 텍스처 파일에서 입술 지우기 (원본 파일 덮어쓰기)
                
                origin_path = texture_file.replace(os.path.splitext(texture_file)[1], "_origin" + os.path.splitext(texture_file)[1])
                if not os.path.exists(origin_path):
                    import shutil
                    shutil.copy2(texture_file, origin_path)
                    print(f"원본 파일 생성: {origin_path}")
                
                success = mouth_eraser.erase_mouth(texture_file, texture_file)
                if success:
                    print(f"텍스처 파일의 입술이 성공적으로 지워졌습니다: {texture_file}")
                else:
                    print(f"텍스처 파일의 입술 지우기에 실패했습니다: {texture_file}")
            else:
                print(f"텍스처 파일을 찾을 수 없습니다. 모델 경로: {model_path}")
                
        except Exception as e:
            print(f"텍스처 파일 처리 중 오류: {str(e)}")
    
    def __erase_mouth_from_image(self, image_path, mouth_eraser):
        """
        이미지 파일에서 직접 입술을 지웁니다.
        
        Args:
            image_path (str): 이미지 파일 경로
            mouth_eraser (MouthEraser): MouthEraser 인스턴스
        """
        try:
            print(f"이미지에서 입술을 지웁니다: {image_path}")
            
            # 백업 파일 생성 (선택사항)
            backup_path = image_path.replace(os.path.splitext(image_path)[1], "_backup" + os.path.splitext(image_path)[1])
            if not os.path.exists(backup_path):
                import shutil
                shutil.copy2(image_path, backup_path)
                print(f"백업 파일 생성: {backup_path}")
            
            # 입술 지우기 (원본 파일 덮어쓰기)
            success = mouth_eraser.erase_mouth(image_path, image_path)
            
            if success:
                print(f"이미지의 입술이 성공적으로 지워졌습니다: {image_path}")
            else:
                print(f"이미지의 입술 지우기에 실패했습니다: {image_path}")
                
        except Exception as e:
            print(f"이미지 파일 처리 중 오류: {str(e)}")

    def __facescan_laminate_registration(self, visualize=False):
        print("facescan_laminate_registration")
        facescan_data = self.parsed_json["facescan"]
        for facescan in facescan_data:
            if facescan["subType"] == "faceSmile":
                print(f'facescan["path"]: {facescan["path"]}')
                if facescan["path"].endswith(".obj") or facescan["path"].endswith(".ply"):
                    # Lazy import
                    from pyNeo3DLib.faceRegisration.faceLaminateRegistration import FaceLaminateRegistration
                    
                    # Now register this file with the laminate model
                    facescan_laminate_registration = FaceLaminateRegistration(facescan["path"], RegistrationConstants.LAMINATE_PATH, visualize)
                    final_transform, moved_smile_mesh = facescan_laminate_registration.run_registration()
                    return final_transform, moved_smile_mesh, "FaceScan"
                elif facescan["path"].endswith(".jpg") or facescan["path"].endswith(".png"):
                    # facephoto_registration = FacePhotoRegistration(facescan["path"], visualize)
                    # M_total_homogeneous, image_plane = facephoto_registration.run_registration()
                    
                    for face in facescan_data:
                        if face["subType"] == "faceRest":
                            print(f'facescan2["path"]: {face["path"]}')
                            rest_path = face["path"]
                        elif face["subType"] == "faceRetraction":
                            print(f'facescan2["path"]: {face["path"]}')
                            retraction_path = face["path"]
                    
                    # Lazy import
                    from pyNeo3DLib.faceRegisration.faceAlign import FaceAlignment3D
                    
                    face_aligner = FaceAlignment3D(front_image_path=facescan["path"], right_image_path=rest_path, left_image_path=retraction_path)
                    M_total_homogeneous, image_planes = face_aligner.run_registration(visualize=visualize)
                    return M_total_homogeneous, image_planes, "FacePhoto"
                else:
                    return None

    def __facescan_rest_registration(self, transformed_face_smile_mesh, facescan_laminate_result, visualize=False):
        print("facescan_rest_registration")
        facescan_data = self.parsed_json["facescan"]
        rest_path = None
        retraction_path = None
        for facescan in facescan_data:
            if facescan["subType"] == "faceRest":
                print(f'facescan["path"]: {facescan["path"]}')
                rest_path = facescan["path"]
            elif facescan["subType"] == "faceRetraction":
                print(f'facescan["path"]: {facescan["path"]}')
                retraction_path = facescan["path"]
        
        # Lazy import
        from pyNeo3DLib.faceRegisration.facesRegistration import FacesRegistration
        
        if rest_path.endswith(".obj") or rest_path.endswith(".ply"):
            facescan_rest_registration = FacesRegistration(transformed_face_smile_mesh, facescan_laminate_result, rest_path, retraction_path, visualize)
            result_for_rest, result_for_retraction = facescan_rest_registration.run_registration()
        else:
            result_for_rest = np.array(RegistrationConstants.IDENTITY_MATRIX)
            result_for_retraction = np.array(RegistrationConstants.IDENTITY_MATRIX)

        return result_for_rest, result_for_retraction

    
    def __cbct_registration(self):
        print("cbct_registration")
        matrix = np.array(RegistrationConstants.IDENTITY_MATRIX)
        return matrix
                
                
    def __smilearch_outerline_detect(self, visualize=False):
        print("smilearch_outerline_detect")
        
        # Lazy import
        from pyNeo3DLib.smileArchOuterline.core import analyze_upper_IOS_scandata
        
        ios_data = self.parsed_json["ios"]
        print(f'ios_data: {ios_data}')
        
        for data in ios_data:
            if data["subType"] == "upper":
                print(f'data["path"]: {data["path"]}')
                arch_depth, molar_width, landmarks = analyze_upper_IOS_scandata(
                    mesh_path=data["path"],
                    visualize_result=visualize
                )
                return arch_depth, molar_width, landmarks
        return None
        

    def __correct_reflection(self, matrix):
        # 3x3 회전 행렬의 행렬식 계산
        det = np.linalg.det(matrix[:3, :3])
        
        # 행렬식이 음수면 반사 변환이 있음
        if det < 0:
            print(f"반사 변환 감지됨 (행렬식: {det}). 보정 중...")
            # x축 반전 적용 (다른 축을 선택해도 됨)
            reflection_fix = np.eye(4)
            reflection_fix[0, 0] = -1
            return np.dot(reflection_fix, matrix)
        return matrix


