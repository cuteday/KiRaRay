# KiRaRay

*KiRaRay* is a simple interactive ray-tracing renderer using optix.

<p align=center>
<img src=common/demo/kirara.jpg width="800">

### Features

> __Working in progress (?) project__  
> This toy renderer is purposed for learning and only with limited features.

- [x] GPU path tracing (megakernel/wavefront).
- [x] Post processing passes (e.g. denoising).
- [x] Basic support for scene animation (glTF).

### Build and run

| *Windows (MSVC)* | [![Build](https://github.com/cuteday/KiRaRay/actions/workflows/main.yml/badge.svg)](https://github.com/cuteday/KiRaRay/actions/workflows/main.yml) |
| --------- | ------------------------------------------------------------ |

#### Requirements

- Nvidia RTX GPU (Turing or higher).
- [OptiX](https://developer.nvidia.com/rtx/ray-tracing/optix) **7.3+** and [CUDA](https://developer.nvidia.com/cuda-toolkit) **11.4+**.
- [Vulkan SDK](https://vulkan.lunarg.com/) **1.3+**.

This project is developed with on Windows (MSVC). It cannot compile on Linux. 

> *KiRaRay* now uses Vulkan for better interoperability with CUDA, and extensibility to rasterization-based render passes. If Vulkan is not desired, check the [legacy-GL](https://github.com/cuteday/KiRaRay/tree/legacy-GL) branch that instead depends on OpenGL.

#### Cloning the repository

*KiRaRay* uses thirdparty dependencies as submodules so fetch them recursively when cloning:

~~~bash
git clone --recursive https://github.com/cuteday/KiRaRay.git
~~~

#### Building

This project uses cmake to build, Make sure cuda is installed and added to PATH, and `OptiX_INSTALL_DIR` environment variable points to the OptiX installation directory.

#### Running

Specify the json configuration file as command line argument to start the renderer, as the example below. Check the [example configuration](common/configs/example.json) for how to configure the renderer.

~~~bash
build/src/kiraray.exe common/configs/example.json
~~~

> The two necessary entries in the configuration are `model` (specifying the relative path to the scene file) and `passes` (describing the render pipeline). Once compiled, directly run `kiraray` without specifying configuration (the example configuration will be used) to get a feel for this toy renderer.

</details>

#### Usage

**Camera controlling.** Dragging `LeftMouse` for orbiting, dragging `Scroll` or `Shift+LeftMouse` for panning. `Scroll` for zooming in/out.

**Python binding.** Several simple interfaces are exposed to python scripting via [pybind11](https://github.com/pybind/pybind11), see [scripts](common/scripts) for details.

### Galleries

<p align=center>
<img src=common/demo/gallery.png width="800">

### Additional Information

#### Performance

Currently, the renderer runs extremely slow on *Debug* build for unknown reasons. Please switch to *Release* build for normal performance.

#### Writing new render passes.

It is possible to write your own render pass, see the examples [here](src/misc/samples/). Check [bindless render pass](src/render/rasterize/) (rasterization) or the [post-processing passes](src/render/passes/) for more working examples.

#### Scene loading

*Kiraray* provided limited support for importing scenes like OBJ, glTF2 using [Assimp](https://github.com/assimp/assimp.git). Animations in glTF2 models could be correctly imported, but skeleton animation is not yet supported. [pbrt-parser](https://github.com/ingowald/pbrt-parser) is used to import [pbrt-v3](https://github.com/mmp/pbrt-v3/) scenes (get some [here](https://benedikt-bitterli.me/resources/), modify the file url to download the pbrt-v3 format models).

### Epilogue

Although the main purpose of this project is to let me (a beginner) learn c++ and optix, I really wish to add more features and make it a fully-functional renderer with support for both ray-tracing and rasterization based techniques, combined via vulkan-cuda interopration. However, it may be a long process and I don't know if I will continue to do it.  Since in reality i am so lazy, trying to sleep as more as possible (\*/ω＼\*).

### Credits
- The great optix tutorial for beginners: [optix7course](https://github.com/ingowald/optix7course).
- Some of the code are adapted from [pbrt](https://github.com/mmp/pbrt-v4) and [donut](https://github.com/NVIDIAGameWorks/donut). 
- *KiRaRay* has a [tiny math wrapper](https://github.com/cuteday/KiRaRay/tree/main/src/core/math) built upon [eigen](http://eigen.tuxfamily.org/).
- [ImGui](https://github.com/ocornut/imgui) is used to build simple user interfaces for this project. 
