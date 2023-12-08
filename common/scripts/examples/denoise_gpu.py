import os
import sys
import torch
import argparse

try: import numpy as np
except: raise Exception("Install numpy for manipulating arrays.")

from krr import *
import pykrr

def denoise_tensor(rgb:torch.Tensor, normals:torch.Tensor=None, albedo:torch.Tensor=None)->np.array:
	if normals is not None: 
		assert (rgb.shape == normals.shape)
	if albedo is not None: 
		assert (rgb.shape == albedo.shape)
	return pykrr.denoise_torch_tensor(rgb, normals, albedo)

if __name__ == "__main__":
	a = torch.rand(size=[720, 1280, 3], device="cuda", dtype=torch.float32)
	print(a)
	b = denoise_tensor(a)
	print(b)