from pyNeo3DLib import Neo3DRegistration

with open(f"{__file__}/../sample.json", "r") as f:
    json_string = f.read()
    reg = Neo3DRegistration(json_string)
    print(reg.version)
    print(reg.file_info)
    reg.run_registration()