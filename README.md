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

- [CMake](https://cmake.org/) **3.24+** and Visual Studio with the C++ desktop development tools.
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

For a Release build with Ninja, run these commands from an **x64 Native Tools Command Prompt for Visual Studio** (or a Developer PowerShell configured for x64):

~~~powershell
cmake -S . -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/release --parallel 8
build/release/bin/kiraray.exe common/configs/example_cbox.json
~~~

The default CUDA architecture is `native`, targeting the installed GPU. Set `-DCMAKE_CUDA_ARCHITECTURES=120` to select an architecture explicitly, or an appropriate architecture for another machine. An explicit architecture is required when configuring without a GPU. CUDA 13 requires a Turing or newer target (compute capability 7.5+).

To choose between installed OptiX SDKs, pass an explicit CMake path, for example `-DOptiX_INSTALL_DIR="C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.1.0"`. This takes precedence over the environment and updates the include directory when reconfiguring an existing build. Without an explicit selection or environment variable, the newest SDK under `%PROGRAMDATA%/NVIDIA Corporation` is selected.

The following combinations have passed a Release build and short wavefront and megakernel Cornell box renders on an RTX 5080 (driver **616.52**, MSVC **19.44**):

| CUDA | OptiX |
| ---- | ----- |
| 12.9.41 | 8.0.0 |
| 13.2.51 | 8.0.0 |
| 13.2.51 | 9.1.0 |

CUDA 13.2-specific compiler options are version-gated; CUDA 12 builds do not receive them. CUDA 12.5 remains the minimum supported version, but it has not been retested locally with these changes.

| Variable                | Default     | Description                                                  |
| ----------------------- | ----------- | ------------------------------------------------------------ |
| `OptiX_INSTALL_DIR`     | auto-detect | When auto-detection failed to find the correct OptiX path, this variable needs to be manually specified to point to the OptiX installation. |
| `KRR_RENDER_SPECTRAL`   | ON          | Whether to build spectral render. If turned OFF, the RGB renderer is build. |
| `KRR_PYTHON_PATH` | auto-detect | Manually specify this to enable Python binding for a specific version of Python. |

#### VS Code and clangd

VS Code and clangd configuration files are local to each workspace and ignored by Git. Use the clangd and CMake Tools extensions with Ninja's compilation database in `build/release`. Clangd **22.1.6** was tested with CUDA 13.2; older versions bundled with Visual Studio cannot parse its headers. Configure clangd to remove NVCC-only flags, preserve CUDA parsing for `.cpp` sources compiled by NVCC, and use the selected toolkit's `--cuda-path`.

For Windows host-code debugging, select the **Debug** CMake variant. In Cursor, use CodeLLDB (`lldb`); Cursor does not support the MSVC debugger type (`cppvsdbg`). VS Code can use either CodeLLDB or the C/C++ extension's MSVC debugger. Some renderer CUDA files still produce Clang/NVCC host-device declaration or macro diagnostics; the NVCC build remains the authoritative compiler check.

#### Running

Specify the json configuration file as command line argument to start the renderer, as the example below. Check the [example configurations](common/configs) for some test scenes, showcasing core features like volumetric rendering, animated scenes and motion blur.

~~~bash
build/release/bin/kiraray.exe common/configs/example_cbox.json
~~~

> The two necessary entries in the configuration are `model` (specifying the relative path to the scene file) and `passes` (describing the render pipeline). Once compiled, directly run `kiraray` without specifying configuration (this [example configuration](common/configs/example_cbox.json) will be used) to get a feel for this toy renderer.

#### Usage

**Camera controlling.** Dragging `LeftMouse` for orbiting, dragging `Scroll` or `Shift+LeftMouse` for panning. `Scroll` for zooming in/out.

**Python binding.** Several simple interfaces are exposed to python scripting via [pybind11](https://github.com/pybind/pybind11), including a OptiX denoiser wrapper for denoising NumPy or PyTorch tensors, see [scripts](common/scripts) for details.

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
