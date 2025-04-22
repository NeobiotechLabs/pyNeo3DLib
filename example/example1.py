from pyNeo3DLib import Neo3DRegistration

with open(f"{__file__}/../sampleInput.json", "r") as f:
    json_string = f.read()
    reg = Neo3DRegistration(json_string)
    print(reg.version)
    print(reg.parsed_json)
    result = reg.run_registration(visualize=True)
    print(result)