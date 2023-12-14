import os
import sys
import glob
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
os.chdir(ROOT_DIR)

sys.path += [os.path.dirname(pyd) for pyd in glob.iglob(os.path.join(ROOT_DIR, "build*", "**/*.pyd"), recursive=True)]
sys.path += [os.path.dirname(pyd) for pyd in glob.iglob(os.path.join(ROOT_DIR, "build*", "**/*.so"), recursive=True)]
sys.path += [os.path.dirname(pyd) for pyd in glob.iglob(os.path.join(ROOT_DIR, "out*", "**/*.pyd"), recursive=True)]
sys.path += [os.path.dirname(pyd) for pyd in glob.iglob(os.path.join(ROOT_DIR, "out*", "**/*.so"), recursive=True)]

import pykrr_common

os.add_dll_directory(os.path.join(pykrr_common.vulkan_root, "Bin"))
os.add_dll_directory(os.path.join(pykrr_common.pytorch_root, "lib"))

import pykrr

if __name__ == "__main__":
	print(sys.path)