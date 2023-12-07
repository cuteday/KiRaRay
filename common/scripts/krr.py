import os
import sys
import glob
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
os.chdir(ROOT_DIR)

print(ROOT_DIR)

sys.path += [os.path.dirname(pyd) for pyd in glob.iglob(os.path.join(ROOT_DIR, "build*", "**/*.pyd"), recursive=True)]
sys.path += [os.path.dirname(pyd) for pyd in glob.iglob(os.path.join(ROOT_DIR, "build*", "**/*.so"), recursive=True)]
sys.path += [os.path.dirname(pyd) for pyd in glob.iglob(os.path.join(ROOT_DIR, "out*", "**/*.pyd"), recursive=True)]
sys.path += [os.path.dirname(pyd) for pyd in glob.iglob(os.path.join(ROOT_DIR, "out*", "**/*.so"), recursive=True)]


TORCH_LIB_PATH = r"C:\Users\cuted\miniconda3\envs\mitsuba\Lib\site-packages\torch\lib"
CONDA_LIB_PATH = r"C:\Users\cuted\miniconda3\envs\mitsuba\bin"
CUDA_LIB_PATH = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\lib"
os.environ['PATH'] = os.environ['PATH'] + ";" + TORCH_LIB_PATH + ";" + CONDA_LIB_PATH

if __name__ == "__main__":
	print(sys.path)