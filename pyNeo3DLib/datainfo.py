class dataInfo:
    def __init__(self, path, type):
        self.path = path    
        self.type = type
        self.transformMatrix = None
        self.dicomInfo = None

class dicomInfo():
    def __init__(self, path):
        self.slice_count = 0
        self.slice_thickness = 0        


