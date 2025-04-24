# main_controller.py
import sys
import subprocess

if len(sys.argv) < 2:
    print("Usage: python main_controller.py [client|train]")
    sys.exit(1)

mode = sys.argv[1]

if mode == "client":
    subprocess.run(["python", "client_main.py"])
elif mode == "train":
    subprocess.run(["python", "model_main.py"])
else:
    print("Unknown mode:", mode)
