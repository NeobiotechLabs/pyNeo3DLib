import json
from pyNeo3DLib.iosRegistration.iosLaminateRegistration import IOSLaminateRegistration
from pyNeo3DLib.faceRegisration.faceLaminateRegistration import FaceLaminateRegistration
from pyNeo3DLib.faceRegisration.facesRegistration import FacesRegistration

LAMINATE_PATH = "D:/Projects/PyNeo3DLib/example/data/smile_arch_half.stl"

class Neo3DRegistration:
    def __init__(self, json_string):
        self.version = "0.0.1"
        self.file_info = self.__parse_json(json_string)


    def __parse_json(self, json_string):
        parsed_json = json.loads(json_string)
        return parsed_json
    
    
    def run_registration(self):       
        self.__verify_file_info()

        ios_laminate_result = self.__ios_laminate_registration()
        ios_upper_result = self.__ios_upper_registration()
        ios_lower_result = self.__ios_lower_registration()

        facescan_laminate_result = self.__facescan_laminate_registration()
        facescan_rest_result = self.__facescan_rest_registration()
        facescan_retraction_result = self.__facescan_retraction_registration()

        cbct_result = self.__cbct_registration()

        ios_bow_result = self.__ios_bow_registration()

        print(ios_laminate_result)
        print(facescan_laminate_result)


    def __make_result_json(self, ios_laminate_result, 
                            ios_upper_result, 
                            ios_lower_result, 
                            facescan_laminate_result, 
                            facescan_rest_result, 
                            facescan_retraction_result, 
                            cbct_result, 
                            ios_bow_result):
        pass
        

    def __verify_file_info(self):
            # ios 검사
        ios_data = self.file_info.get("ios")
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
        if self.file_info.get("facescan") is None:
            raise ValueError("facescan is not defined")
        
        # cbct 검사
        if self.file_info.get("cbct") is None:
            raise ValueError("cbct is not defined")

    def __ios_laminate_registration(self):
        print("ios_laminate_registration")
        ios_data = self.file_info["ios"]
        for ios in ios_data:
            if ios["subType"] == "smileArch":
                print(f'ios["path"]: {ios["path"]}')
                # 이제 이 파일과. 라미네이트 모델을 정합한다. 
                ios_laminate_registration = IOSLaminateRegistration(ios["path"], LAMINATE_PATH)
                return ios_laminate_registration.run_registration()
        pass

    def __ios_upper_registration(self):
        print("ios_upper_registration")
        pass

    def __ios_lower_registration(self):
        print("ios_lower_registration")
        pass

    def __facescan_laminate_registration(self):
        print("facescan_laminate_registration")
        facescan_data = self.file_info["facescan"]
        for facescan in facescan_data:
            if facescan["subType"] == "smile":
                print(f'facescan["path"]: {facescan["path"]}')
                # 이제 이 파일과. 라미네이트 모델을 정합한다. 
                facescan_laminate_registration = FaceLaminateRegistration(facescan["path"], LAMINATE_PATH)
                return facescan_laminate_registration.run_registration()
        pass

    def __facescan_rest_registration(self):
        print("facescan_rest_registration")
        facescan_data = self.file_info["facescan"]
        pass

    def __facescan_retraction_registration(self):
        print("facescan_retraction_registration")
        pass
    
    def __cbct_registration(self):
        print("cbct_registration")
        pass
    
    def __ios_bow_registration(self):
        print("ios_bow_registration")
        pass
