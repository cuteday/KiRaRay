import os
import sys
import argparse

try: import cv2
except: raise Exception("Install opencv-python first for image IO.")

try: import numpy as np
except: raise Exception("Install numpy for manipulating arrays.")

from krr import *
import pykrr

def denoise(rgb, normals:np.array=None, albedo:np.array=None)->np.array:
	if normals is not None: 
		assert (normals.shape == rgb.shape)
	if albedo is not None: 
		assert (albedo.shape == rgb.shape)
	return pykrr.denoise(rgb.astype(np.float32), normals, albedo)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--rgb", type=str, default="common/scripts/images/noisy.exr", help="path of the noisy hdr image")
	parser.add_argument("--albedo", type=str, default=None, help="[optional] path of the albedo")
	parser.add_argument("--normals", type=str, default=None, help="[optional] path of the normals")
	parser.add_argument("--result", type=str, default="common/scripts/images/denoised.exr", help="path of the output file")
	args = parser.parse_args()

	if not (args.rgb.endswith(".exr") or args.rgb.endswith(".hdr")):
		raise Exception("Denoiser is applied to HDR image only.")

	img_rgb = cv2.imread(args.rgb, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED).astype(np.float32)
	print("RGB image shape: ", img_rgb.shape)
	img_normals = None
	img_albedo = None

	if args.albedo:
		img_albedo = cv2.imread(args.albedo, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED).astype(np.float32)
	if args.normals:
		img_normals = cv2.imread(args.normals, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED).astype(np.float32)	

	img_result = denoise(img_rgb, img_normals, img_albedo)
	cv2.imwrite(args.result, img_result)