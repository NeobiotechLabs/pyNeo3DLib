from pyNeo3DLib import fastserver

import requests
import time
import json
import threading

print("Server started, http://localhost:8000")


fastserver.start_server()



print("\nPress Enter to stop server...")
input()


fastserver.stop_server()
print("Server stopped")




