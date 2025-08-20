import json
import numpy as np
import asyncio
import os
from PIL import Image
import io
import base64
from dataclasses import dataclass
from pyNeo3DLib.iosRegistration.iosLaminateRegistration import IOSLaminateRegistration
from pyNeo3DLib.faceRegisration.faceLaminateRegistration import FaceLaminateRegistration
from pyNeo3DLib.faceRegisration.facePhotoRegistration import FacePhotoRegistration
from pyNeo3DLib.faceRegisration.facesRegistration import FacesRegistration
from pyNeo3DLib.bowRegistration.iosBowRegistration import IOSBowRegistration
from pyNeo3DLib.condyleFinder.condyleFinder import CondyleFinder
from pyNeo3DLib.smileArchOuterline.core import analyze_upper_IOS_scandata
from pyNeo3DLib.faceRegisration.faceAlign import FaceAlignment3D
from pyNeo3DLib.goldenProportion.goldenProportionFinder import GoldenProportionFinder


LAMINATE_PATH = os.path.join(os.path.dirname(__file__), "smile_arch_half.stl")
CENTERPIN_PATH = os.path.join(os.path.dirname(__file__), "center_pin.stl")


class progress_event:
    def __init__(self, type, progress, message):
        self.type = type
        self.progress = progress
        self.message = message

    def __str__(self):
        return f"progress_event(type={self.type}, progress={self.progress}, message={self.message})"

    def __repr__(self):
        return self.__str__()
    
    def get_json(self):
        return {
            "type": self.type,
            "progress": self.progress,
            "message": self.message
        }
    

class Neo3DRegistration:
    def __init__(self, json_string, websocket):
        self.version = "0.0.1"
        print(f"json_string: {json_string}")
        self.parsed_json = self.__parse_json(json_string)        
        self.websocket = websocket

    def __parse_json(self, json_string):
        try:
            parsed_json = json.loads(json_string)
            print(f"parsed_json: {parsed_json}")
            return parsed_json
        except Exception as e:
            raise ValueError(f"Failed to parse JSON: {e}")
    
    
    async def run_registration(self, visualize=False):       
        self.__verify_file_info()
        
        total_progress = 10

        data = {
            "type": "progress",
            "progress": 100 / total_progress * 0,
            "message": "ios_laminate_registration",
            "random_text": "random_text",
            "timestamp": "sdafkljhsdf"
        }

        print(f"data: {data}")
        if(self.websocket is not None):
            print(type(self.websocket))
            await self.websocket.send_json(data)
            await asyncio.sleep(0.1)

        ios_laminate_result = self.__ios_laminate_registration(visualize=visualize)

        if(self.websocket is not None):
            await self.websocket.send_json(progress_event(type="progress", progress=100 / total_progress * 1, message="ios_upper_registration").get_json())            
            await asyncio.sleep(0.1)
        ios_upper_result = self.__ios_upper_registration()

        if(self.websocket is not None):
            await self.websocket.send_json(progress_event(type="progress", progress=100 / total_progress * 2, message="ios_lower_registration").get_json())
            await asyncio.sleep(0.1)
        ios_lower_result = self.__ios_lower_registration()

        if(self.websocket is not None):
            await self.websocket.send_json(progress_event(type="progress", progress=100 / total_progress * 3, message="facescan_laminate_registration").get_json())
            await asyncio.sleep(0.1)
        facescan_laminate_result, transformed_face_smile_mesh, type_of_facedata = self.__facescan_laminate_registration(visualize=visualize)

        if(type_of_facedata == "FaceScan"):
            if(self.websocket is not None):
                await self.websocket.send_json(progress_event(type="progress", progress=100 / total_progress * 4, message="facescan_rest_registration").get_json())
                await asyncio.sleep(0.1)
            facescan_rest_result, facescan_retraction_result = self.__facescan_rest_registration(transformed_face_smile_mesh, facescan_laminate_result, visualize=visualize)
            facephoto_meshes = None
        else:
            facephoto_meshes = transformed_face_smile_mesh
            facescan_rest_result = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0],
                                              [0, 0, 1, 0],
                                              [0, 0, 0, 1]])
            facescan_retraction_result = np.array([[1, 0, 0, 0],
                                                    [0, 1, 0, 0],
                                                    [0, 0, 1, 0],
                                                    [0, 0, 0, 1]])

        if(self.websocket is not None):
            await self.websocket.send_json(progress_event(type="progress", progress=100 / total_progress * 5, message="cbct_registration").get_json())
            await asyncio.sleep(0.1)
        cbct_result = self.__cbct_registration()

        if(self.websocket is not None):
            await self.websocket.send_json(progress_event(type="progress", progress=100 / total_progress * 6, message="ios_bow_registration").get_json())
            await asyncio.sleep(0.1)
        ios_bow_result = self.__ios_bow_registration(ios_laminate_result, visualize=visualize)
        
        if(self.websocket is not None):
            await self.websocket.send_json(progress_event(type="progress", progress=100 / total_progress * 7, message="condyle_registration").get_json())
            await asyncio.sleep(0.1)
        condyle_result = self.__condyle_detection(facescan_laminate_result, visualize)
        print(f'__condyle_detection result: {condyle_result}')
        
        if(self.websocket is not None):
            await self.websocket.send_json(progress_event(type="progress", progress=100 / total_progress * 8, message="golden_proportion_registration").get_json())
            await asyncio.sleep(0.1)
        golden_proportion_result = self.__golden_proportion_detection(facescan_laminate_result, visualize)
        print(f'__golden_proportion_detection result: {golden_proportion_result}')
        
        if(self.websocket is not None):
            await self.websocket.send_json(progress_event(type="progress", progress=100 / total_progress * 9, message="smilearch_outerline_registration").get_json())
            await asyncio.sleep(0.1)
        smilearch_outerline_result = self.__smilearch_outerline_detect(ios_laminate_result, visualize)

        result = self.__make_result_json(
            ios_laminate_result.tolist(), ios_upper_result.tolist(), ios_lower_result.tolist(), facescan_laminate_result.tolist(), facephoto_meshes, facescan_rest_result.tolist(), facescan_retraction_result.tolist(), cbct_result.tolist(), ios_bow_result.tolist(), condyle_result, golden_proportion_result, smilearch_outerline_result
        )
        
        if(self.websocket is not None):
            await self.websocket.send_json(progress_event(type="progress", progress=100, message="All registration completed").get_json())
            await self.websocket.send_json(progress_event(type="result", progress=100, message=result).get_json())
            await asyncio.sleep(0.1)
        return result

    def __make_result_json(self, ios_laminate_result, 
                            ios_upper_result, 
                            ios_lower_result, 
                            facescan_laminate_result, 
                            facephoto_meshes,
                            facescan_rest_result, 
                            facescan_retraction_result, 
                            cbct_result, 
                            ios_bow_result,
                            condyle_result,
                            golden_proportion_result,
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
        print(f'condyle_result: {condyle_result}')
        print(f'golden_proportion_result: {golden_proportion_result}')
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
        
        # condyle 정보 추가
        if condyle_result is not None:
            vertices, faces = self.__make_condyle_plane(condyle_result)
            condyle_json = {
                "vertices": vertices.tolist(),
                "faces": faces.tolist(),
                "points": condyle_result.tolist()
            }
            self.parsed_json["condyle"] = {"mesh": condyle_json}
            
        if golden_proportion_result is not None:
            golden_proportion_json = {
                "points": golden_proportion_result  # 이미 딕셔너리 형태이므로 그대로 사용
            }
            self.parsed_json["golden_proportion"] = golden_proportion_json
            
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
        ios_data = self.parsed_json["ios"]
        for ios in ios_data:
            if ios["subType"] == "smileArch":
                print(f'ios["path"]: {ios["path"]}')
                # Now register this file with the laminate model
                ios_laminate_registration = IOSLaminateRegistration(ios["path"], LAMINATE_PATH, visualize)
                result_matrix = ios_laminate_registration.run_registration()
                return result_matrix

    def __ios_upper_registration(self):
        print("ios_upper_registration")
        matrix = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        return matrix

    def __ios_lower_registration(self):
        print("ios_lower_registration")
        matrix = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        return matrix

    def __facescan_laminate_registration(self, visualize=False):
        print("facescan_laminate_registration")
        facescan_data = self.parsed_json["facescan"]
        for facescan in facescan_data:
            if facescan["subType"] == "faceSmile":
                print(f'facescan["path"]: {facescan["path"]}')
                if facescan["path"].endswith(".obj") or facescan["path"].endswith(".ply"):
                    # Now register this file with the laminate model
                    facescan_laminate_registration = FaceLaminateRegistration(facescan["path"], LAMINATE_PATH, visualize)
                    final_transform, moved_smile_mesh = facescan_laminate_registration.run_registration()
                    return final_transform, moved_smile_mesh, "FaceScan"
                elif facescan["path"].endswith(".jpg"):
                    # facephoto_registration = FacePhotoRegistration(facescan["path"], visualize)
                    # M_total_homogeneous, image_plane = facephoto_registration.run_registration()
                    
                    for face in facescan_data:
                        if face["subType"] == "faceRest":
                            print(f'facescan2["path"]: {face["path"]}')
                            rest_path = face["path"]
                        elif face["subType"] == "faceRetraction":
                            print(f'facescan2["path"]: {face["path"]}')
                            retraction_path = face["path"]
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
                
        if rest_path.endswith(".obj") or rest_path.endswith(".ply"):
            facescan_rest_registration = FacesRegistration(transformed_face_smile_mesh, facescan_laminate_result, rest_path, retraction_path, visualize)
            result_for_rest, result_for_retraction = facescan_rest_registration.run_registration()
        else:
            result_for_rest = np.array([[1, 0, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]])
            result_for_retraction = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0],
                                                [0, 0, 1, 0],
                                                [0, 0, 0, 1]])

        return result_for_rest, result_for_retraction

    
    def __cbct_registration(self):
        print("cbct_registration")
        matrix = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        return matrix
    
    def __ios_bow_registration(self, ios_laminate_result, visualize=False):
        print("ios_bow_registration")
        ios_data = self.parsed_json["ios"]
        for ios in ios_data:
            if ios["subType"] == "smileArch":
                ios_bow_registration = IOSBowRegistration(ios["path"], CENTERPIN_PATH, visualize)
                # result_matrix 는 ios 정렬된 상태에서의 bow 이동 매트릭스
                result_matrix = ios_bow_registration.run_registration()

                # ios_laminate_result 는 정렬 + 라미네이트 정합 이동의 합
                # 이를 빼고 라미네이트 정합 이동만 구하기 위해해 역행렬 적용
                ios_transform_matrix_inv = np.linalg.inv(ios_bow_registration.ios_transform_matrix)
                ios_moved = np.dot(ios_laminate_result, ios_transform_matrix_inv)
                
                # 라미네이트 정합 이동 + bow 이동 적용
                # final_result = np.dot(bow_moved, ios_laminate_result)
                final_result = np.dot(ios_moved, result_matrix)
                final_result = self.__correct_reflection(final_result)
                print(f'final_result: {final_result}')

                return final_result
            
    def __condyle_detection(self, face_registration_result, visualize=False):
        print("condyle detection")
        facescan_data = self.parsed_json["facescan"]
        for facescan in facescan_data:
            if facescan["subType"] == "faceSmile":
                print(f'facescan["path"]: {facescan["path"]}')
                if facescan["path"].endswith(".obj") or facescan["path"].endswith(".ply"):                
                    # Now register this file with the laminate model
                    condyle_finder = CondyleFinder(facescan["path"], visualize)
                    result = condyle_finder.run_analysis()
                    
                    # face_registration_result 변환 행렬을 콘딜 점들에 적용
                    if result is not None and len(result) > 0:
                        # result를 numpy 배열로 변환
                        condyle_points = np.array(result)
                        
                        # 동차 좌표로 변환 (4x4 행렬 적용을 위해)
                        ones = np.ones((condyle_points.shape[0], 1))
                        homogeneous_points = np.hstack([condyle_points, ones])
                        
                        # 변환 행렬 적용
                        transformed_points = np.dot(homogeneous_points, face_registration_result.T)
                        
                        # 3D 좌표만 추출 (동차 좌표에서 w=1로 나누기)
                        result = transformed_points[:, :3] / transformed_points[:, 3:4]
                    
                    return result
                else:
                    return None
                
    def __golden_proportion_detection(self, face_registration_result, visualize=False):
        print("golden_proportion_detection")
        facescan_data = self.parsed_json["facescan"]
        for facescan in facescan_data:
            if facescan["subType"] == "faceSmile":
                print(f'facescan["path"]: {facescan["path"]}')
                if facescan["path"].endswith(".obj") or facescan["path"].endswith(".ply"):
                    golden_proportion_finder = GoldenProportionFinder(facescan["path"], visualize)
                    result = golden_proportion_finder.run_analysis()
                    
                    print(f'golden_proportion_finder result: {result}')
                    
                    if result is not None and len(result) > 0:
                        # result를 numpy 배열로 변환
                        golden_proportion_points = np.array([result['a'], result['b'], result['c'], result['d']])
                        
                        # 동차 좌표로 변환 (4x4 행렬 적용을 위해)
                        ones = np.ones((golden_proportion_points.shape[0], 1))
                        homogeneous_points = np.hstack([golden_proportion_points, ones])
                        
                        # 변환 행렬 적용
                        transformed_points = np.dot(homogeneous_points, face_registration_result.T)
                        
                        # 3D 좌표만 추출 (동차 좌표에서 w=1로 나누기)
                        transformed_coords = transformed_points[:, :3] / transformed_points[:, 3:4]
                        
                        # a, b, c, d 키 값을 유지하여 딕셔너리로 반환
                        result = {
                            'a': transformed_coords[0].tolist(),
                            'b': transformed_coords[1].tolist(),
                            'c': transformed_coords[2].tolist(),
                            'd': transformed_coords[3].tolist()
                        }
                    
                    return result
                else:
                    return None
                
    def __smilearch_outerline_detect(self, ios_laminate_result, visualize=False):
        print("smilearch_outerline_detect")
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
    
            
    def __make_condyle_plane(self, condyle_points):
        print("make_condyle_plane")
        # 콘딜 점들을 포함하는 평면 만들기
        if condyle_points is not None and len(condyle_points) == 2:
            points = np.array(condyle_points)
            
            # 세 번째 점 생성 (첫 번째 점에서 y축으로 10만큼 이동)
            third_point = np.array([0, 0, 0])
            points = np.vstack([points, third_point])

            # 삼각형 메시 생성
            vertices = points
            faces = np.array([[0, 1, 2]])  # 세 점을 연결하는 하나의 삼각형

            return vertices, faces
        

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


