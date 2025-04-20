from pydantic import BaseModel

class dentModel:
    def __init__(self):
        pass


class RegistrationItem(BaseModel):
    origin_type: str
    origin_model: str
    target_type: str
    target_model: str



class RegistrationModels:
    def __init__(self):
        pass
    
    def request_registration(self, origin_type, origin_model, target_type, target_model):
        return {"transfer" : [0, 1, 2], "rotation" : [0, 1, 2]}
