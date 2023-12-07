import os
import sys
import json
import argparse

from krr import *
print(sys.path)
try:
	import numpy as np
except:
	print("Warning: NumPy not found. Install NumPy to python environment if any runtime error occur.")
import pykrr

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, default="common/configs/example_cbox.json", help="Path to scene file")
	args = parser.parse_args()

	config = json.load(open(args.config))
	# the application use pybind_json to forward json objects to c++ code
	pykrr.run(config=config)