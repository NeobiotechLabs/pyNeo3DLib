import json
import numpy as np
import asyncio
import os
from PIL import Image
import io
import base64
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from pyNeo3DLib.fileLoader.mesh import Mesh

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
            ios_laminate_result = self.__ios_laminate_registration(visualize=False)
            print(f'✅ ios_laminate_registration 성공')
        except Exception as e:
            print(f'❌ ios_laminate_registration 실패: {str(e)}')
            ios_laminate_result = np.array(RegistrationConstants.IDENTITY_MATRIX)


        # IOS Upper/Lower Registration - 메시 객체 생성
        from pyNeo3DLib.fileLoader.mesh import Mesh

        smile_arch_path = self.config.get_ios_by_subtype("smileArch").path
        ios_upper_path = self.config.get_ios_by_subtype("upper").path
        ios_lower_path = self.config.get_ios_by_subtype("lower").path

        ios_upper_mesh = Mesh.from_file(ios_upper_path)
        ios_lower_mesh = Mesh.from_file(ios_lower_path)
        smile_arch_mesh = Mesh.from_file(smile_arch_path)

        # IOS Upper Registration
        ios_upper_result = await self._safe_compute_ios_transformation(
            progress_name="ios_upper_registration",
            ios_mesh=ios_upper_mesh,
            smile_arch_mesh=smile_arch_mesh,
            ios_laminate_result=ios_laminate_result,
            transformation_name="combined_transformation_matrix_upper",
            is_upper=True
        )
   
        # IOS Lower Registration
        ios_lower_result = await self._safe_compute_ios_transformation(
            progress_name="ios_lower_registration",
            ios_mesh=ios_lower_mesh,
            smile_arch_mesh=smile_arch_mesh,
            ios_laminate_result=ios_laminate_result,
            transformation_name="combined_transformation_matrix_lower",
            is_upper=False
        )

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

    async def _safe_compute_ios_transformation(
        self,
        progress_name: str,
        ios_mesh: "Mesh",
        smile_arch_mesh: "Mesh",
        ios_laminate_result: np.ndarray,
        transformation_name: str,
        is_upper: bool
    ) -> np.ndarray:
        """
        IOS transformation 계산을 안전하게 수행하는 헬퍼 메서드.
        실패 시 IDENTITY_MATRIX를 반환하여 프로그램이 죽지 않도록 합니다.
        
        Args:
            progress_name: 진행 상황 보고용 이름
            ios_mesh: IOS 메시 (upper 또는 lower)
            smile_arch_mesh: smile arch 메시
            ios_laminate_result: IOS laminate registration 결과
            transformation_name: transformation 행렬 이름 (로깅용)
            is_upper: True면 Upper, False면 Lower
            
        Returns:
            계산된 transformation 행렬 또는 IDENTITY_MATRIX (실패 시)
        """
        try:
            await self.progress_reporter.report_progress(progress_name)
            
            transformation_matrix = self._compute_ios_to_smilearch_transformation(
                ios_mesh=ios_mesh,
                smile_arch_mesh=smile_arch_mesh,
                ios_laminate_result=ios_laminate_result,
                is_upper=is_upper
            )
            
            if transformation_matrix is not None:
                print(f'✅ {transformation_name}: {transformation_matrix}')
                print(f'✅ {progress_name} 성공')
                return transformation_matrix
            else:
                print(f'❌ {transformation_name} 계산 실패')
                return np.array(RegistrationConstants.IDENTITY_MATRIX)
                
        except Exception as e:
            print(f'❌ {progress_name} 실패: {str(e)}')
            return np.array(RegistrationConstants.IDENTITY_MATRIX)
    
    def _compute_ios_to_smilearch_transformation(
        self,
        ios_mesh: "Mesh",
        smile_arch_mesh: "Mesh",
        ios_laminate_result: np.ndarray,
        is_upper: bool
    ) -> Optional[np.ndarray]:
        """
        IOS 메시를 Smile Arch로 정렬하는 변환 행렬을 계산합니다.
        
        처리 과정:
        1. PCA를 통한 주축 계산
        2. 레이캐스팅을 통한 방향 벡터 찾기
        3. 좌표계 구축 (x, y, z 축)
        4. 회전 + 이동 변환 행렬 계산
        
        Args:
            ios_mesh: IOS 메시 객체 (Upper 또는 Lower)
            smile_arch_mesh: Smile Arch 메시 객체
            ios_laminate_result: IOS Laminate 변환 행렬
            is_upper: True면 Upper, False면 Lower
            
        Returns:
            4x4 변환 행렬, 실패 시 None
        """
        try:
            mesh_type = "Upper" if is_upper else "Lower"
            
            # 1. 메시 데이터 준비
            ios_vertices, ios_faces, smile_arch_centroid = self._prepare_mesh_data(
                ios_mesh, smile_arch_mesh, ios_laminate_result, mesh_type
            )
            
            # 2. 주축 계산
            principal_axes, closest_axis, closest_axis_vector, centroid = self._compute_principal_axes(
                ios_vertices
            )
            
            # 3. Z축 벡터 계산(입천장 후보 주축 closest_axis_vector을 입력받아 양방향 후보군을 만든뒤 ios_mesh의 평균법선 벡터 방향과 가장 가까운 방향의 벡터를 반환)
            z_axis_vector = self._compute_z_axis_vector(
                ios_mesh, closest_axis_vector
            )
            
            if z_axis_vector is None:
                return None
            
            # 4. 단일 교차점 방향 찾기
            single_intersection_direction = self._find_single_intersection_direction(
                mesh_vertices=ios_vertices,
                mesh_faces=ios_faces,
                principal_axes=principal_axes,
                centroid=centroid,
                closest_axis_idx=closest_axis
            )
            
            if single_intersection_direction is None:
                print("⚠️ 단일 교차점 방향을 찾지 못했습니다.")
                return None
            
            
            # 6. 좌표계 구축 및 변환 행렬 계산
            combined_transformation_matrix = self._compute_final_transformation(
                single_intersection_direction=single_intersection_direction,
                z_axis_vector=z_axis_vector,
                centroid=centroid,
                ios_vertices=ios_vertices,
                smile_arch_centroid=smile_arch_centroid,
                is_upper=is_upper
            )

            if not is_upper:
                lower_translation_matrix = np.eye(4)
                lower_translation_matrix[:3, 3] = np.array([0, 0, -15])
                combined_transformation_matrix = np.matmul(lower_translation_matrix, combined_transformation_matrix)
                
            return combined_transformation_matrix
            
        except Exception as e:
            print(f"❌ IOS-SmileArch {mesh_type} 변환 행렬 계산 중 오류 발생: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _prepare_mesh_data(
        self,
        ios_mesh: "Mesh",
        smile_arch_mesh: "Mesh",
        ios_laminate_result: np.ndarray,
        mesh_type: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """메시 데이터를 준비합니다."""
        ios_vertices = ios_mesh.vertices
        ios_faces = ios_mesh.faces
        smile_arch_vertices = smile_arch_mesh.vertices
        
        # Smile Arch 변환 적용
        smile_arch_vertices = np.dot(
            smile_arch_vertices,
            ios_laminate_result[:3, :3].T
        ) + ios_laminate_result[:3, 3]
        
        smile_arch_centroid = np.mean(smile_arch_vertices, axis=0)
        
        print(f"✅ IOS {mesh_type} 메시: {ios_vertices.shape[0]} vertices")
        print(f"✅ Smile Arch 메시: {smile_arch_vertices.shape[0]} vertices")
        
        return ios_vertices, ios_faces, smile_arch_centroid
    
    def _compute_principal_axes(
        self, 
        vertices: np.ndarray
    ) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray]:
        """주축을 계산합니다."""
        # PCA를 통한 주축 계산
        principal_axes, _, centroid = compute_principal_axes_from_vertices(
            vertices, 
            verbose=True
        )
        
        # 분산이 가장 작은 주축 계산
        minimum_variance_axis, _, _ = compute_minimum_variance_axis_from_vertices(
            vertices,
            verbose=True
        )
        print(f"✅ 분산이 가장 작은 주축: {minimum_variance_axis}")
        
        # principal_axes에서 minimum_variance_axis와 가장 가까운 주축 찾기
        closest_axis = np.argmax(np.abs(np.dot(principal_axes, minimum_variance_axis)))
        closest_axis_vector = principal_axes[closest_axis]
        print(f"✅ 가장 가까운 주축 인덱스: {closest_axis}")
        print(f"✅ closest_axis_vector: {closest_axis_vector}")
        
        return principal_axes, closest_axis, closest_axis_vector, centroid
    
    def _compute_z_axis_vector(
        self,
        ios_mesh: "Mesh",
        closest_axis_vector: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Z축 벡터를 계산합니다 (Upper/Lower에 따라 다른 방식)."""

        if ios_mesh.normals is None:
            ios_mesh._compute_normals()
        
        ios_normals = np.asarray(ios_mesh.normals)
        ios_normals_mean = np.mean(ios_normals, axis=0)
        print(f"✅ 하악 메쉬 법선벡터 평균값: {ios_normals_mean}")
        
        # 내적으로 방향 확인
        inner_product = np.dot(closest_axis_vector, ios_normals_mean)
        if inner_product > 0:
            print("✅ 같은 방향")
            return closest_axis_vector
        else:
            print("✅ 반대 방향")
            return -closest_axis_vector


    
    def _compute_final_transformation(
        self,
        single_intersection_direction: np.ndarray,
        z_axis_vector: np.ndarray,
        centroid: np.ndarray,
        ios_vertices: np.ndarray,
        smile_arch_centroid: np.ndarray,
        is_upper: bool
    ) -> np.ndarray:
        """최종 변환 행렬을 계산합니다."""
        # 좌표계 구축
        x_axis_vector, y_axis_vector, z_axis_vector = self._build_coordinate_system(
            single_intersection_direction=single_intersection_direction,
            closest_axis_vector=z_axis_vector
        )
        
        # 회전 행렬 계산 (표준 좌표계로 변환)
        rotation_matrix = self._compute_rotation_matrix_to_standard_jaw(
            x_axis=x_axis_vector,
            y_axis=y_axis_vector,
            z_axis=z_axis_vector,
            is_upper=is_upper
        )
        
        # 회전 + 이동을 결합한 변환 행렬 계산
        combined_transformation_matrix = self._compute_combined_transformation(
            rotation_matrix=rotation_matrix,
            centroid=centroid,
            source_vertices=ios_vertices,
            target_centroid=smile_arch_centroid
        )
        
        return combined_transformation_matrix
    
    def _get_ray_casting_vector_to_centroid(
        self,
        mesh_vertices: np.ndarray,
        mesh_faces: np.ndarray,
        centroid: np.ndarray,
        axis_vector: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        도심점에서 axis_vector 방향으로 양방향 레이캐스팅하여 메쉬 표면 포인트를 검출하고,
        검출한 포인트에서 도심점 방향으로 가는 벡터를 반환합니다.
        
        Args:
            mesh_vertices: 메시의 정점 배열
            mesh_faces: 메시의 면 정보
            centroid: 메시의 무게중심
            axis_vector: 레이캐스팅 방향 벡터
            
        Returns:
            교차점에서 도심점으로 가는 벡터, 교차점이 없으면 None
        """
        import pyvista as pv
        from pyNeo3DLib.smileArchOuterline.utils.ray_caster import RayCaster
        
        # pyvista 메시 생성
        pv_mesh = self._create_pyvista_mesh(mesh_vertices, mesh_faces)
        ray_caster = RayCaster()
        
        # 양방향 레이캐스팅으로 표면 포인트 찾기
        surface_point = self._find_surface_point_by_raycasting(
            pv_mesh, ray_caster, centroid, axis_vector
        )
        
        if surface_point is None:
            return None
        
        # 교차점에서 도심점으로 가는 벡터
        return centroid - surface_point
    
    def _create_pyvista_mesh(
        self, 
        vertices: np.ndarray, 
        faces: np.ndarray
    ) -> Any:
        """PyVista 메시를 생성합니다."""
        import pyvista as pv
        faces_with_count = np.column_stack([np.full(len(faces), 3), faces])
        return pv.PolyData(vertices, faces_with_count)
    
    def _find_surface_point_by_raycasting(
        self,
        pv_mesh: Any,
        ray_caster: Any,
        origin: np.ndarray,
        direction: np.ndarray
    ) -> Optional[np.ndarray]:
        """레이캐스팅으로 표면 포인트를 찾습니다."""
        # plus 방향 레이캐스팅
        plus_intersections = ray_caster.ray_casting(
            pv_mesh, origin.reshape(1, 3), direction.reshape(1, 3)
        )
        
        if len(plus_intersections) > 0:
            return plus_intersections[0]
        
        # minus 방향 레이캐스팅
        minus_intersections = ray_caster.ray_casting(
            pv_mesh, origin.reshape(1, 3), (-direction).reshape(1, 3)
        )
        
        if len(minus_intersections) > 0:
            return minus_intersections[0]
        
        return None

    def _find_single_intersection_direction(
        self, 
        mesh_vertices: np.ndarray,
        mesh_faces: np.ndarray,
        principal_axes: np.ndarray, 
        centroid: np.ndarray, 
        closest_axis_idx: int
    ) -> Optional[np.ndarray]:
        """
        레이캐스팅을 통해 단일 교차점을 가진 축 방향을 찾습니다.
        
        Args:
            mesh_vertices: 메시의 정점 배열
            mesh_faces: 메시의 면 정보
            principal_axes: 주성분 분석으로 얻은 주축들 (3x3)
            centroid: 메시의 무게중심
            closest_axis_idx: 제외할 축의 인덱스
            
        Returns:
            단일 교차점을 가진 축 방향 벡터, 없으면 None
        """
        from pyNeo3DLib.smileArchOuterline.utils.ray_caster import RayCaster
        
        # PyVista 메시 및 RayCaster 준비
        pv_mesh = self._create_pyvista_mesh(mesh_vertices, mesh_faces)
        ray_caster = RayCaster()
        
        # 제외할 축을 제외한 나머지 축들
        remaining_axes_indices = [i for i in range(3) if i != closest_axis_idx]
        
        # 나머지 두 축에 대해 레이캐스팅 수행
        for axis_idx in remaining_axes_indices:
            axis_vector = principal_axes[axis_idx]
            
            # 단일 교차점 확인
            unit_vector = self._check_single_intersection(
                pv_mesh, ray_caster, centroid, axis_vector, axis_idx
            )
            
            if unit_vector is not None:
                return unit_vector
        
        print("⚠️ 최초 교차점이 1개인 축을 찾지 못했습니다.")
        return None
    
    def _check_single_intersection(
        self,
        pv_mesh: Any,
        ray_caster: Any,
        centroid: np.ndarray,
        axis_vector: np.ndarray,
        axis_idx: int
    ) -> Optional[np.ndarray]:
        """특정 축에 대해 단일 교차점 여부를 확인합니다."""
        # +방향 레이캐스팅
        plus_intersections = ray_caster.ray_casting(
            pv_mesh, centroid.reshape(1, 3), axis_vector.reshape(1, 3)
        )
        plus_has_intersection = len(plus_intersections) > 0
        
        # -방향 레이캐스팅
        minus_intersections = ray_caster.ray_casting(
            pv_mesh, centroid.reshape(1, 3), (-axis_vector).reshape(1, 3)
        )
        minus_has_intersection = len(minus_intersections) > 0
        
        # 교차점 개수 계산
        intersection_count = int(plus_has_intersection) + int(minus_has_intersection)
        print(f"  축 {axis_idx} 레이캐스팅 결과: 최초 교차점 {intersection_count}개")
        
        # 최초 교차점이 정확히 1개인 경우
        if intersection_count == 1:
            intersection_point = self._get_closest_intersection_point(
                plus_intersections if plus_has_intersection else minus_intersections,
                centroid
            )
            
            # 도심점에서 교차점 방향으로 나가는 단위벡터 계산
            direction_vector = intersection_point - centroid
            unit_vector = direction_vector / np.linalg.norm(direction_vector)
            
            print(f"✅ 최초 교차점이 1개인 축 발견: 축 {axis_idx}")
            print(f"   교차점: {intersection_point}")
            print(f"   도심점에서 교차점 방향 단위벡터: {unit_vector}")
            
            return unit_vector
        
        return None
    
    def _get_closest_intersection_point(
        self,
        intersections: np.ndarray,
        centroid: np.ndarray
    ) -> np.ndarray:
        """교차점들 중 도심점에 가장 가까운 점을 반환합니다."""
        distances = np.linalg.norm(intersections - centroid, axis=1)
        closest_idx = np.argmin(distances)
        return intersections[closest_idx]
    
    def _build_coordinate_system(
        self, 
        single_intersection_direction: np.ndarray, 
        closest_axis_vector: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        좌표계를 구축합니다 (x, y, z 축).
        
        그람-슈미트 직교 정규화를 적용하여 완벽한 직교 정규 기저를 생성합니다.
        
        Args:
            single_intersection_direction: Y축이 될 방향 벡터
            closest_axis_vector: Z축이 될 방향 벡터
            
        Returns:
            정규화된 x_axis, y_axis, z_axis 벡터 (단위 벡터, 서로 직교)
        """
        print(f"🔍 입력 벡터:")
        print(f"   y_axis (원본): {single_intersection_direction}, norm: {np.linalg.norm(single_intersection_direction):.6f}")
        print(f"   z_axis (원본): {closest_axis_vector}, norm: {np.linalg.norm(closest_axis_vector):.6f}")
        print(f"   y·z: {np.dot(single_intersection_direction, closest_axis_vector):.6f}")
        
        # 그람-슈미트 정규화로 직교 정규 기저 생성
        # y축을 우선으로 유지하고, z축을 조정한 후 x축을 재계산
        
        # 1. y축 정규화
        y_axis_vector = single_intersection_direction / np.linalg.norm(single_intersection_direction)
        
        # 2. z축을 y축에 직교하도록 조정 후 정규화
        z_orthogonal = closest_axis_vector - np.dot(closest_axis_vector, y_axis_vector) * y_axis_vector
        z_axis_vector = z_orthogonal / np.linalg.norm(z_orthogonal)
        
        # 3. x축을 y축과 z축에 직교하도록 외적으로 재계산
        x_axis_vector = np.cross(y_axis_vector, z_axis_vector)
        
        # 정규화된 축 벡터 검증
        print(f"✅ 정규화된 축 벡터:")
        print(f"   x_axis: {x_axis_vector}, norm: {np.linalg.norm(x_axis_vector):.10f}")
        print(f"   y_axis: {y_axis_vector}, norm: {np.linalg.norm(y_axis_vector):.10f}")
        print(f"   z_axis: {z_axis_vector}, norm: {np.linalg.norm(z_axis_vector):.10f}")
        print(f"   x·y: {np.dot(x_axis_vector, y_axis_vector):.10f} (0이어야 함)")
        print(f"   y·z: {np.dot(y_axis_vector, z_axis_vector):.10f} (0이어야 함)")
        print(f"   z·x: {np.dot(z_axis_vector, x_axis_vector):.10f} (0이어야 함)")
        
        return x_axis_vector, y_axis_vector, z_axis_vector

        
    
    def _compute_rotation_matrix_to_standard_jaw(
        self, 
        x_axis: np.ndarray, 
        y_axis: np.ndarray, 
        z_axis: np.ndarray,
        is_upper: bool
    ) -> np.ndarray:
        """
        상악/하악에 맞는 회전 행렬을 계산합니다.
        
        Args:
            x_axis: 정규화된 x축 벡터
            y_axis: 정규화된 y축 벡터
            z_axis: 정규화된 z축 벡터
            is_upper: True면 상악(upper), False면 하악(lower)
            
        Returns:
            4x4 동차변환 행렬
        """
        if is_upper:
            target_axes = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).T
        else:
            target_axes = np.eye(3)
        
        return self._compute_rotation_matrix_to_standard(x_axis, y_axis, z_axis, target_axes)

    def _compute_rotation_matrix_to_standard(
        self, 
        x_axis: np.ndarray, 
        y_axis: np.ndarray, 
        z_axis: np.ndarray,
        target_axes: np.ndarray
    ) -> np.ndarray:
        """
        현재 좌표계를 목표 좌표계로 변환하는 회전 행렬을 계산합니다.
        
        입력 축 벡터들은 이미 정규화되고 직교하는 것으로 가정합니다.
        
        Args:
            x_axis, y_axis, z_axis: 현재 좌표계의 정규화된 축 벡터들
            target_axes: 목표 좌표계 (3x3 행렬, 각 열이 목표 축)
            
        Returns:
            4x4 동차변환 행렬
        """
        # 현재 좌표계 행렬 (각 열이 축 벡터)
        current_coordinate_system = np.column_stack([x_axis, y_axis, z_axis])
        
        # 회전 행렬 계산: R = Target @ Current^T
        # 정규 직교 행렬이므로 역행렬 = 전치 행렬
        rotation_matrix_3x3 = target_axes @ current_coordinate_system.T
        
        # 회전 행렬 검증
        det = np.linalg.det(rotation_matrix_3x3)
        is_orthogonal = np.allclose(rotation_matrix_3x3 @ rotation_matrix_3x3.T, np.eye(3), atol=1e-6)
        
        print(f"✅ 회전 행렬 검증:")
        print(f"   행렬식: {det:.10f} (1이어야 함)")
        print(f"   직교 행렬 여부: {is_orthogonal}")
        
        if not is_orthogonal or abs(det - 1.0) > 1e-6:
            print(f"⚠️ 경고: 회전 행렬이 올바르지 않습니다!")
            print(f"   R @ R.T:\n{rotation_matrix_3x3 @ rotation_matrix_3x3.T}")
        
        # 4x4 동차변환 행렬로 확장
        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = rotation_matrix_3x3
        
        print(f"✅ rotation_matrix (4x4):\n{rotation_matrix}")
        
        return rotation_matrix

    
    def _compute_combined_transformation(
        self,
        rotation_matrix: np.ndarray,
        centroid: np.ndarray,
        source_vertices: np.ndarray,
        target_centroid: np.ndarray
    ) -> np.ndarray:
        """
        회전 + 이동을 결합한 단일 4x4 동차 변환 행렬을 계산합니다.
        
        변환 순서:
        1. 소스 도심점을 원점으로 이동 (T1)
        2. 회전 변환 적용 (R)
        3. target_centroid로 이동 (T2)
        
        최종 변환: T2 @ R @ T1
        
        Args:
            rotation_matrix: 4x4 동차 변환 행렬
            centroid: 회전 중심점 (메시의 무게중심)
            source_vertices: 변환할 메시의 정점 배열
            target_centroid: 목표 위치의 무게중심
            
        Returns:
            4x4 동차 변환 행렬
        """
        # 소스 도심점 계산
        source_centroid = np.mean(source_vertices, axis=0)
        
        # 1단계: 소스 도심점을 원점으로 이동하는 변환 행렬 (T1)
        T1 = np.eye(4)
        T1[:3, 3] = -source_centroid
        
        # 2단계: 회전 변환 행렬 (R)
        # rotation_matrix가 이미 4x4 형태이므로 그대로 사용
        R = rotation_matrix.copy()
        
        # 3단계: target_centroid로 이동하는 변환 행렬 (T2)
        T2 = np.eye(4)
        T2[:3, 3] = target_centroid
        
        # 4단계: 최종 변환 행렬 결합 (T2 @ R @ T1)
        combined_matrix = T2 @ R @ T1
        

        
        return combined_matrix


    
    def _apply_transformation_to_mesh(
        self,
        mesh,
        transformation_matrix: np.ndarray
    ):
        """
        메시에 4x4 동차 변환 행렬을 적용합니다 (in-place).
        
        Args:
            mesh: 변환할 Mesh 객체
            transformation_matrix: 4x4 동차 변환 행렬
            
        Returns:
            변환된 Mesh 객체
        """
        # 정점을 동차 좌표로 변환 (N x 3 -> N x 4)
        vertices_homogeneous = np.hstack([
            mesh.vertices,
            np.ones((len(mesh.vertices), 1))
        ])
        
        # 변환 적용
        transformed_vertices_homogeneous = vertices_homogeneous @ transformation_matrix.T
        
        # 3D 좌표로 변환 (N x 4 -> N x 3)
        mesh.vertices = transformed_vertices_homogeneous[:, :3]
        
        return mesh

def compute_principal_axes_from_vertices(vertices: np.ndarray, faces: Optional[np.ndarray] = None, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    메시 버텍스로부터 회전관성 주축을 계산합니다.
    trimesh 라이브러리를 사용하여 물리적으로 정확한 관성 텐서를 계산합니다.
    
    Args:
        vertices: 메시의 정점 좌표 배열 (N x 3)
        faces: 메시의 면 정보 (선택사항, 제공시 더 정확한 계산)
        verbose: 계산 과정을 출력할지 여부
    
    Returns:
        principal_axes: 회전관성 주축 (3x3 행렬, 각 열이 주축 벡터)
        eigenvalues: 각 주축에 대한 관성 모멘트 값 (작은 순서대로)
        centroid: 메시의 무게중심
    
    Example:
        >>> vertices = mesh.vertices
        >>> axes, moments, center = compute_principal_axes_from_vertices(vertices)
        >>> print(f"제1주축: {axes[:, 0]}")
        >>> print(f"제2주축: {axes[:, 1]}")
        >>> print(f"제3주축: {axes[:, 2]}")
    """
    import trimesh
    
    # trimesh 객체 생성
    if faces is not None:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    else:
        # faces가 없으면 convex hull 사용
        mesh = trimesh.convex.convex_hull(vertices)
    
    # trimesh의 내장 함수 사용
    principal_axes = mesh.principal_inertia_vectors
    eigenvalues = mesh.principal_inertia_components
    centroid = mesh.center_mass
    
    if verbose:
        print(f'📍 무게중심: {centroid}')
        print(f'📊 주관성 모멘트 (고유값): {eigenvalues}')
        print(f'🎯 회전관성 주축 (고유벡터):\n{principal_axes}')
        print(f'   - 제1주축 (최소 관성): {principal_axes[:, 0]}')
        print(f'   - 제2주축 (중간 관성): {principal_axes[:, 1]}')
        print(f'   - 제3주축 (최대 관성): {principal_axes[:, 2]}')
    
    return principal_axes, eigenvalues, centroid


def compute_minimum_variance_axis_from_vertices(vertices: np.ndarray, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA(주성분 분석)를 사용하여 메시 버텍스의 분산이 가장 작은 주축을 계산합니다.
    scipy를 사용하여 효율적으로 계산합니다.
    
    Args:
        vertices: 메시의 정점 좌표 배열 (N x 3)
        verbose: 계산 과정을 출력할지 여부
    
    Returns:
        minimum_variance_axis: 분산이 가장 작은 주축 벡터 (3,)
        all_axes: 모든 주성분 축 (3x3 행렬, 각 열이 주축 벡터, 분산 작은 순서)
        variances: 각 주축에 대한 분산 값 (작은 순서대로)
    
    Example:
        >>> vertices = mesh.vertices
        >>> min_axis, all_axes, variances = compute_minimum_variance_axis_from_vertices(vertices)
        >>> print(f"분산이 가장 작은 주축: {min_axis}")
        >>> print(f"분산 비율: {variances / np.sum(variances)}")
    
    Note:
        PCA는 공분산 행렬을 사용하여 데이터의 주성분을 찾습니다.
        분산이 가장 작은 축은 데이터가 가장 평평한 방향을 나타냅니다.
    """
    from scipy.linalg import eigh
    
    # 1. 데이터의 중심 계산
    centroid = np.mean(vertices, axis=0)
    if verbose:
        print(f'📍 PCA 중심점: {centroid}')
    
    # 2. 중심을 원점으로 이동
    centered_vertices = vertices - centroid
    
    # 3. 공분산 행렬 계산 (scipy 사용)
    covariance_matrix = np.cov(centered_vertices.T)
    if verbose:
        print(f'🔢 공분산 행렬:\n{covariance_matrix}')
    
    # 4. scipy의 eigh로 고유값/고유벡터 계산 (대칭 행렬에 최적화)
    eigenvalues, eigenvectors = eigh(covariance_matrix)
    
    # 5. 고유값(분산)이 작은 순서대로 정렬
    sorted_indices = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # 6. 결과 추출
    all_axes = eigenvectors  # 모든 주성분 축
    variances = eigenvalues  # 각 축의 분산
    minimum_variance_axis = eigenvectors[:, 0]  # 분산이 가장 작은 축
    
    if verbose:
        print(f'📊 주성분 분산 (고유값): {variances}')
        print(f'📊 분산 비율: {variances / np.sum(variances) * 100}%')
        print(f'🎯 주성분 축 (고유벡터):\n{all_axes}')
        print(f'   - 제1주축 (최소 분산): {all_axes[:, 0]} (분산: {variances[0]:.2f})')
        print(f'   - 제2주축 (중간 분산): {all_axes[:, 1]} (분산: {variances[1]:.2f})')
        print(f'   - 제3주축 (최대 분산): {all_axes[:, 2]} (분산: {variances[2]:.2f})')
        print(f'✨ 분산이 가장 작은 주축: {minimum_variance_axis}')
    
    return minimum_variance_axis, all_axes, variances














    
