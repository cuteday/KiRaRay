# KiRaRay

*KiRaRay* is a simple interactive ray-tracing renderer using optix.

<p align=center>
<img src=common/demo/attack-on-usagi.jpeg width="800">


### Features

> __Working in progress (?) project__  
> This toy renderer is purposed for learning and only with limited features.

- [x] GPU path tracing (megakernel/wavefront).
- [x] GPU volumetric rendering (wavefront).
- [x] Post processing passes (e.g. denoising).
- [x] Single/multi-level scene graph with animation support.
- [x] Interactive editing scene components with simple UI.

### Build and run

| *Windows (MSVC, C++17)* | [![Build](https://github.com/cuteday/KiRaRay/actions/workflows/main.yml/badge.svg)](https://github.com/cuteday/KiRaRay/actions/workflows/main.yml) |
| --------- | ------------------------------------------------------------ |

#### Requirements

- [OptiX](https://developer.nvidia.com/rtx/ray-tracing/optix) **7.5+** and [CUDA](https://developer.nvidia.com/cuda-toolkit) **12.5+**.
- CUDA Compute Capability 6+ (*Pascal*+, *Turing*+ is better).
- [Vulkan SDK](https://vulkan.lunarg.com/) **1.3+**.

This project is developed with on Windows (MSVC). It cannot compile on Linux. 

> *KiRaRay* has bumped the required CUDA version to 12.5 for newer libcu++ features. The most recent legacy version on CUDA 11.4 is at the [cu114](https://github.com/cuteday/KiRaRay/tree/cu114) branch.

> *KiRaRay* now uses Vulkan for interoperability with CUDA. If Vulkan is not desired, check the [legacy-GL](https://github.com/cuteday/KiRaRay/tree/legacy-GL) branch that instead depends on OpenGL.

#### Cloning the repository

*KiRaRay* uses thirdparty dependencies as submodules so fetch them recursively when cloning:

~~~bash
git clone --recursive https://github.com/cuteday/KiRaRay.git
~~~

#### Building

This project uses cmake to build, make sure CUDA is installed and added to PATH. Several optional variables are available for building configuration.

| Variable                | Default     | Description                                                  |
| ----------------------- | ----------- | ------------------------------------------------------------ |
| `OptiX_INSTALL_DIR`     | auto-detect | When auto-detection failed to find the correct OptiX path, this variable needs to be manually specified to point to the OptiX installation. |
| `KRR_RENDER_SPECTRAL`   | ON          | Whether to build spectral render. If turned OFF, the RGB renderer is build. |
| `KRR_PYTHON_PATH` | auto-detect | Manually specify this to enable Python binding for a specific version of Python. |

#### Running

Specify the json configuration file as command line argument to start the renderer, as the example below. Check the [example configurations](common/configs) for some test scenes, showcasing core features like volumetric rendering, animated scenes and motion blur.

~~~bash
build/src/kiraray.exe common/configs/example_cbox.json
~~~

> The two necessary entries in the configuration are `model` (specifying the relative path to the scene file) and `passes` (describing the render pipeline). Once compiled, directly run `kiraray` without specifying configuration (this [example configuration](common/configs/example_cbox.json) will be used) to get a feel for this toy renderer.

#### Usage

**Camera controlling.** Dragging `LeftMouse` for orbiting, dragging `Scroll` or `Shift+LeftMouse` for panning. `Scroll` for zooming in/out.

**Python binding.** Several simple interfaces are exposed to python scripting via [pybind11](https://github.com/pybind/pybind11), including a OptiX denoiser wrapper for denoising NumPy or PyTorch tensors, see [scripts](common/scripts) for details.

#### Known Build Issues
- In CUDA 12.5 there exist some CUDA-only expressions in thrust headers. If you use CUDA 12.5, you may consider disable the thrust routines in host code (as done in [this commit](https://github.com/cuteday/KiRaRay/commit/c25c2fab44f0ba18cd99b60a4bc757ec0e1ab2a6)) or update to 12.6.
- In CUDA 12.6 there is a compile error in NVTX related code referenced by thrust (`MemoryBarrier` undefined). While I do not know why, I temporarily disabled NVTX as a workaround by defining the `NVTX_DISABLE` macro.

### Galleries

<p align=center>
<img src=common/demo/gallery.png width="800">

### Additional Information

#### Writing new render passes.

It is possible to write your own render pass, see the examples [here](src/misc/samples/). Check [bindless render pass](src/render/rasterize/) (rasterization) or the [post-processing passes](src/render/passes/) for more working examples.

#### Scene loading

*Kiraray* provided limited support for importing scenes like OBJ, glTF2 using [Assimp](https://github.com/assimp/assimp.git). Animations in glTF2 models could be correctly imported, but skeleton animation is not yet supported. [pbrt-parser](https://github.com/ingowald/pbrt-parser) is used to import [pbrt-v3](https://github.com/mmp/pbrt-v3/) scenes (get some [here](https://benedikt-bitterli.me/resources/), modify the file url to download the pbrt-v3 format models).

<details>
  <summary>Epilogue</summary>

<p align="center">
  <a href="https://github.com/cuteday/KiRaRay">
    <img src="https://github.com/cuteday/KiRaRay/assets/31754324/cd762df1-daae-48ca-bae1-0c5ac5c4ae91">
  </a>

  <p align="center">Be happy today!
  </p>
</p>

Although the main purpose of this project is to let me (a beginner) learn c++ and optix, I really wish to add more features and make it a fully-functional renderer with support for both ray-tracing and rasterization based techniques, combined via vulkan-cuda interopration. However, it may be a long process and I don't know if I will continue to do it.  Since in reality i am so lazy, trying to sleep as more as possible (\*/ω＼\*).

</details>

### Credits
- The great optix tutorial for beginners: [optix7course](https://github.com/ingowald/optix7course).
- Some of the code are adapted from [pbrt](https://github.com/mmp/pbrt-v4) and [donut](https://github.com/NVIDIAGameWorks/donut). 
- *KiRaRay* has a [tiny math wrapper](https://github.com/cuteday/KiRaRay/tree/main/src/core/math) built upon [eigen](http://eigen.tuxfamily.org/).
- [ImGui](https://github.com/ocornut/imgui) is used to build simple user interfaces for this project. 
