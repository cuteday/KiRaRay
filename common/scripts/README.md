# KiRaRay Python Binding

> After Python 3.8, the search paths of DLL dependencies has been reset. Only the system paths, the directory containing the DLL or PYD file are searched for load-time dependencies. Instead, a new function os.add_dll_directory() was added to supply additional search paths. 
Necessary DLL paths are exported from `pykrr_common`, see [krr.py](krr.py) for details.

### Start from script

You can start *KiRaRay* from python script with specified configuration. The configuration should be a python dict object. 

~~~Python
pykrr.run(config = cfg)
~~~

### Denoising Images

Kiraray implements a python wrapper for denoising images with optix's built-in ai denoiser. See [denoise.py](./examples/denoise.py) for an example. To denoise an image, the hdr noisy image is provided as arguments, with optionally the normals and albedo (in linear space). All arguments are numpy arrays with the same shape.

~~~Python
img_denoised = pykrr.denoise(img_noisy, img_normals, img_albedo)
~~~

This makes it easy to denoise many image files with python scripts. On my RTX3070 Laptop, denoising an image with 1920x1080 takes approximately 1s, while most of the overhead is the memory copy between host and device. It takes about 25ms when acting as a render pass (see [denoise.cpp](../../src/render/passes/denoise/denoise.cpp)).

#### Denoising PyTorch Tensor
To enable support for PyTorch, you should define the `TORCH_INSTALL_DIR` environment variable to point to the PyTorch installation directory (see [here](../build/FindPyTorch.cmake) for details). The tensor should be on GPU for no CPU-GPU memory copy.  

~~~Python
img_denoised = pykrr.denoise_pytorch_tensor(img_noisy, img_normals, img_albedo)
~~~