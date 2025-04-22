import json
import numpy as np
from pyNeo3DLib.iosRegistration.iosLaminateRegistration import IOSLaminateRegistration
from pyNeo3DLib.faceRegisration.faceLaminateRegistration import FaceLaminateRegistration
from pyNeo3DLib.faceRegisration.facesRegistration import FacesRegistration

LAMINATE_PATH = "D:/Projects/PyNeo3DLib/example/data/smile_arch_half.stl"

class Neo3DRegistration:
    def __init__(self, json_string):
        self.version = "0.0.1"
        self.parsed_json = self.__parse_json(json_string)


    def __parse_json(self, json_string):
        parsed_json = json.loads(json_string)
        return parsed_json
    
    
    def run_registration(self, visualize=False):       
        self.__verify_file_info()

        ios_laminate_result = self.__ios_laminate_registration(visualize=visualize)
        ios_upper_result = self.__ios_upper_registration()
        ios_lower_result = self.__ios_lower_registration()
        print(self.__facescan_laminate_registration(visualize=visualize))
        facescan_laminate_result, transformed_face_smile_mesh = self.__facescan_laminate_registration(visualize=visualize)
        if transformed_face_smile_mesh is None:
            raise ValueError("transformed_face_smile_mesh is None")
        facescan_rest_result, facescan_retraction_result = self.__facescan_rest_registration(transformed_face_smile_mesh, facescan_laminate_result, visualize=visualize)

        cbct_result = self.__cbct_registration()

        ios_bow_result = self.__ios_bow_registration()

        

        return self.__make_result_json(ios_laminate_result,
                           ios_upper_result, 
                            ios_lower_result, 
                            facescan_laminate_result, 
                            facescan_rest_result, 
                            facescan_retraction_result, 
                            cbct_result, 
                            ios_bow_result)

    def __make_result_json(self, ios_laminate_result, 
                            ios_upper_result, 
                            ios_lower_result, 
                            facescan_laminate_result, 
                            facescan_rest_result, 
                            facescan_retraction_result, 
                            cbct_result, 
                            ios_bow_result):
        
        print("=====================================")
        print(f'ios_laminate_result: {ios_laminate_result}')
        print(f'ios_upper_result: {ios_upper_result}')
        print(f'ios_lower_result: {ios_lower_result}')

        print(f'facescan_laminate_result: {facescan_laminate_result}')
        print(f'facescan_rest_result: {facescan_rest_result}')
        print(f'facescan_retraction_result: {facescan_retraction_result}') 

        print(f'cbct_result: {cbct_result}')

        print(f'ios_bow_result: {ios_bow_result}')
        print("=====================================")

        for ios in self.parsed_json["ios"]:
            if ios["subType"] == "smileArch":
                ios["transform_matrix"] = ios_laminate_result
            elif ios["subType"] == "upper":
                ios["transform_matrix"] = ios_upper_result
            elif ios["subType"] == "lower":
                ios["transform_matrix"] = ios_lower_result
            
        for facescan in self.parsed_json["facescan"]:
            if facescan["subType"] == "smile":
                facescan["transform_matrix"] = facescan_laminate_result
            elif facescan["subType"] == "rest":
                facescan["transform_matrix"] = facescan_rest_result
            elif facescan["subType"] == "retraction":
                facescan["transform_matrix"] = facescan_retraction_result

        self.parsed_json["cbct"]["transform_matrix"] = cbct_result
        self.parsed_json["smilearch_bow"]["transform_matrix"] = ios_bow_result

        return self.parsed_json
        

    def __verify_file_info(self):
            # ios 검사
        ios_data = self.parsed_json.get("ios")
        if ios_data is None:
            raise ValueError("ios is not defined")
        
        # ios 내부 데이터 검사
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

    # facescan 검사
        if self.parsed_json.get("facescan") is None:
            raise ValueError("facescan is not defined")
        
        # cbct 검사
        if self.parsed_json.get("cbct") is None:
            raise ValueError("cbct is not defined")

    def __ios_laminate_registration(self, visualize=False):
        print("ios_laminate_registration")
        ios_data = self.parsed_json["ios"]
        for ios in ios_data:
            if ios["subType"] == "smileArch":
                print(f'ios["path"]: {ios["path"]}')
                # 이제 이 파일과. 라미네이트 모델을 정합한다. 
                ios_laminate_registration = IOSLaminateRegistration(ios["path"], LAMINATE_PATH, visualize)
                return ios_laminate_registration.run_registration()

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
            if facescan["subType"] == "smile":
                print(f'facescan["path"]: {facescan["path"]}')
                # 이제 이 파일과. 라미네이트 모델을 정합한다. 
                facescan_laminate_registration = FaceLaminateRegistration(facescan["path"], LAMINATE_PATH, visualize)
                return facescan_laminate_registration.run_registration()

    def __facescan_rest_registration(self, transformed_face_smile_mesh, facescan_laminate_result, visualize=False):
        print("facescan_rest_registration")
        facescan_data = self.parsed_json["facescan"]
        rest_path = None
        retraction_path = None
        for facescan in facescan_data:
            if facescan["subType"] == "rest":
                print(f'facescan["path"]: {facescan["path"]}')
                rest_path = facescan["path"]
            elif facescan["subType"] == "retraction":
                print(f'facescan["path"]: {facescan["path"]}')
                retraction_path = facescan["path"]

        facescan_rest_registration = FacesRegistration(transformed_face_smile_mesh, facescan_laminate_result, rest_path, retraction_path, visualize)

        return facescan_rest_registration.run_registration()

    def __facescan_retraction_registration(self):
        print("facescan_retraction_registration")
        matrix = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        return matrix
    
    def __cbct_registration(self):
        print("cbct_registration")
        matrix = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        return matrix
    
    def __ios_bow_registration(self):
        print("ios_bow_registration")
        matrix = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        return matrix
