## Samples

This directory contains some basic examples that demonstrate how to implement a new render pass in *KiRaRay*. To compile these examples, turn on `KRR_BUILD_STARLIGHT` in CMake options or via commandline argument `-DKRR_BUILD_STARLIGHT=ON`.

Note that the main focus of this project is still ray-tracing, while it provides basic support and extensibility for rasterztion-based rendering. The vulkan part of *KiRaRay* is wrapped by [nv-rhi](https://github.com/NVIDIAGameWorks/nvrhi), and part of the rendering pipeline implementation is adapted from [donut](https://github.com/NVIDIAGameWorks/donut).

### Example render passes

#### Triangle

<p align=center>
<img src=images/triangle.png width="300">

The adorable triangle... See [triangle.cpp](passes/triangle.cpp).

#### Sine wave simulation

<p align=center>
<img src=images/sinewave.png width="300">

Fill a Vulkan vertex buffer (exported from vulkan to cuda, as an `cudaExternalMemory`) with particles simluating the sine wave using a CUDA kernel, then draw it using Vulkan, see [sinewave.cpp](passes/sinewave.cpp). The kernel is adapted from the official cuda samples [here](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/simpleVulkan).

#### CUDA-VK triangle

<p align=center>
<img src=images/vk-cuda.png width="300">

Draw a triangle using Vulkan first, then dynamically colorize it using a CUDA kernel. To draw on a single framebuffer using two APIs, a semaphore (exported from Vulkan) is used to synchronize them. See [framebuffer.cpp](passes/framebuffer.cpp).