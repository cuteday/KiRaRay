import os
import sys
import argparse

try: import cv2
except: raise Exception("Install opencv-python first for image IO.")

try: import numpy as np
except: raise Exception("Install numpy for manipulating arrays.")

from krr import *
import pykrr

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--rgb", type=str, default="common/scripts/images/noisy.exr", help="Path to scene file")
	parser.add_argument("--result", type=str, default="common/scripts/images/denoised.exr", help="Path to scene file")
	args = parser.parse_args()

	if not (args.rgb.endswith(".exr") or args.rgb.endswith(".hdr")):
		raise Exception("Denoiser is applied to HDR image only.")

	img_rgb = cv2.imread(args.rgb, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED).astype(np.float32)
	print(img_rgb.shape)

	img_result = pykrr.denoise(img_rgb, None, None)
	print(img_result.shape)
	print(img_result)
	img_result = img_result.reshape(img_rgb.shape)
	print(img_result.shape)
	print(np.sum(img_result))
	cv2.imwrite(args.result, img_result)