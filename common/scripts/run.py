import os
import sys
import argparse

from krr import *
import pykrr

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--scene", type=str, default="common/assets/scenes/cbox/cbox.obj", help="Path to scene file")
	parser.add_argument("--env", type=str, default="common/assets/textures/snowwhite.jpg", help="Path to scene file")
	args = parser.parse_args()

	pykrr.run(scene=args.scene, env=args.env)