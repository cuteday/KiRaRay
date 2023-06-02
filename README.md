# KiRaRay

*KiRaRay* is a simple interactive ray-tracing renderer using optix. It is mainly for personal learning purpose with limited features.

<p align=center>
<img src=common/demo/kirara.jpg width="800">

### Features

> __Working in progress (?) project__  
> This project is purposed for learning only with limited features.

- [x] GPU path tracing (megakernel/wavefront).
- [x] Animated scenes (rigging only).
- [x] Post processing passes (e.g. denoising).
- [x] Basic support for importing multiple scenes formats.
- [x] Basic support for Vulkan and CUDA/OptiX interoperation.

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

</details>

#### Usage

**Camera controlling.** Dragging `LeftMouse` for orbiting, dragging `Scroll` or `Shift+LeftMouse` for panning. `Scroll` for zooming in/out.

**Python binding.** Several simple interfaces are exposed to python scripting via [pybind11](https://github.com/pybind/pybind11), see [scripts](common/scripts) for details.

### Galleries

<p align=center>
<img src=common/demo/gallery.png width="800">

### Algorithms

I tried to implement some algorithms designed for path tracing (see [misc](src/misc) for details). 
<details>
<summary>Click for details (・ω< )★ </summary>

Turn the CMake option `KRR_BUILD_STARLIGHT` on if one wants to build these additional algorithm implementations. Note that these code may not be maintained as the main repository.

#### Path Guiding

This implements [Practical Path Guiding (PPG)](https://github.com/Tom94/practical-path-guiding), which is a path guiding algorithm targeted for CPU offline rendering. What I did is largely to simply move the original implementation from CPU to GPU. The performance is not quite satisfying for real-time purposes on GPUs. 

~~~json
	"params": {
		"spp_per_pass": 4,
		"max_memory": 16,
		"bsdf_fraction": 0.5,
		"distribution": "full",
		"stree_thres": 2000,
		"dtree_thres": 0.005,
		"auto_build": true,
		"mode": "offline",
		"sample_combination": "atomatic",
		"budget": {
			"type": "spp",
			"value": 1000
		}
	}
~~~

I also implemented a later [Variance-aware](https://github.com/iRath96/variance-aware-path-guiding) enhancement, which improves PPG on the theoretical side. Use the `distribution` parameter to select from the two methods (`radiance` for standard PPG, and `full` for the variance-aware version).

</details>

### Additional Information

#### Performance

Currently, the renderer runs extremely slow on *Debug* build for unknown reasons. Please switch to *Release* build for normal performance.

#### Scene loading

*Kiraray* provided limited support for importing scenes like OBJ, glTF2 using [Assimp](https://github.com/assimp/assimp.git). Most of the rigging animations in glTF2 models could be correctly imported.. [pbrt-parser](https://github.com/cuteday/pbrt-parser.git) is used to import [pbrt-v3](https://github.com/mmp/pbrt-v3/) scenes (get some [here](https://benedikt-bitterli.me/resources/), change the file url to download the pbrt-v3 format models).

#### Writing new render passes.

It is possible to write your own render pass, see the examples [here](src/misc/samples/). Check these [post-processing passes](src/render/passes/) for more working examples.

### Epilogue

Although the main purpose of this project is to let me (a beginner) learn c++ and optix, I really wish to add more features and make it a fully-functional renderer with support for both ray-tracing and rasterization based techniques, combined via vulkan-cuda interopration. However, it may be a long process and I don't know if I will continue to do it.  Since in reality i am so lazy, trying to sleep as more as possible (\*/ω＼\*).

For anyone that (accidentally) found this project: any questions and suggestions are appreciated. Bug reports might not be necessary since any part of this project could possibly produce unexpected errors ;  ;

### Credits
- The great optix tutorial for beginners: [optix7course](https://github.com/ingowald/optix7course).
- Some of the code are adapted from [pbrt](https://github.com/mmp/pbrt-v4), [donut](https://github.com/NVIDIAGameWorks/donut) and [falcor](https://github.com/NVIDIAGameWorks/Falcor). 
- *KiRaRay* implements a [tiny math wrapper](https://github.com/cuteday/KiRaRay/tree/main/src/core/math) upon [eigen](http://eigen.tuxfamily.org/) for efficient vector/matrix arithmetic.
- [ImGui](https://github.com/ocornut/imgui) is used to build simple user interfaces for this project. 
