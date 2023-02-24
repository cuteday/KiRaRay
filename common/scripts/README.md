# KiRaRay Python Binding

### Start from script

You can start *KiRaRay* from python script with specified configuration. The configuration should be a python dict object. 

~~~Python
pykrr.run(config = cfg)
~~~

### Denoising Images

Kiraray implements a python wrapper for denoising images with optix's built-in ai denoiser. See [denoise.py](denoise.py) for an example. To denoise an image, the hdr noisy image is provided as arguments, with optionally the normals and albedo (in linear space). All arguments are numpy arrays with the same shape.

~~~Python
img_denoised = pykrr.denoise(img_noisy, img_normals, img_albedo)
~~~

This makes it easy to denoise many image files with python scripts. On my RTX3070 Laptop, denoising an image with 1920x1080 takes approximately 1s, while most of the overhead is the memory copy between host and device. It takes about 25ms when acting as a render pass (see [denoise.cpp](../src/render/passes/denoise.cpp)).

<table>
  <tr>
    <td>Noisy</td>
    <td>Denoised</td>
  </tr>
  <tr>
    <td><img src="images/noisy.png" width=390></td>
    <td><img src="images/denoised.png" width=390></td>
  </tr>
 </table>