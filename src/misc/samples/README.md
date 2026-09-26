## Samples

This directory contains some basic examples that demonstrate how to implement a new render pass in *KiRaRay*. To compile these examples, turn on `KRR_BUILD_EXAMPLES` in CMake options or via commandline argument `-DKRR_BUILD_EXAMPLES=ON`.

KiRaRay focuses on ray tracing and also supports rasterization through [NVRHI](https://github.com/NVIDIA-RTX/NVRHI). The shared graphics helpers in `src/core/graphics` work with Vulkan and D3D12; parts were adapted from [Donut](https://github.com/NVIDIA-RTX/Donut). The sample executables use Vulkan by default.

### Example render passes

#### Triangle

<p align=center>
<img src=images/triangle.png width="300">

The adorable triangle... See [triangle.cpp](passes/triangle.cpp).

#### Sine wave simulation

<p align=center>
<img src=images/sinewave.png width="300">

Fill an NVRHI vertex buffer through `CudaBufferMapping` using a CUDA sine-wave kernel, then draw it using graphics commands. `RenderContext::CudaScope` handles the graphics/CUDA handoff; see [sinewave.cpp](passes/sinewave.cpp). The kernel is adapted from the official cuda samples [here](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/simpleVulkan).

#### CUDA/graphics triangle

<p align=center>
<img src=images/vk-cuda.png width="300">

Draw a triangle using NVRHI, then colorize the shared framebuffer using a CUDA kernel. The render context handles synchronization and resource ownership for the selected graphics API; see [framebuffer.cpp](passes/framebuffer.cpp).