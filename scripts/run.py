import os
import sys
import json
import argparse

from krr import *
import pykrr

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, default="common/configs/example.json", help="Path to scene file")
	args = parser.parse_args()

	config = json.load(open(args.config))
	pykrr.run(config=config)