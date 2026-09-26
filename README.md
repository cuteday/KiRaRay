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

| *Windows (MSVC)* | [![Build](https://github.com/cuteday/KiRaRay/actions/workflows/main.yml/badge.svg)](https://github.com/cuteday/KiRaRay/actions/workflows/main.yml) |
| --------- | ------------------------------------------------------------ |

#### Requirements

- [CMake](https://cmake.org/) **3.24+** and Visual Studio with the C++ desktop development tools.
- [OptiX](https://developer.nvidia.com/rtx/ray-tracing/optix) **7.5+** and [CUDA](https://developer.nvidia.com/cuda-toolkit) **12.5+**. Latest tested: CUDA 13.2 and OptiX 9.1.0.
- CUDA Compute Capability: Turing+ (7.5+)
- [Vulkan SDK](https://vulkan.lunarg.com/) **1.3+**.

#### Cloning the repository

*KiRaRay* uses thirdparty dependencies as submodules so fetch them recursively when cloning:

~~~bash
git clone --recursive https://github.com/cuteday/KiRaRay.git
~~~

When updating an existing checkout, synchronize the NVRHI submodule URL:

~~~bash
git submodule sync --recursive
git submodule update --init --recursive
~~~

#### Building

This project uses cmake to build, make sure CUDA is installed and added to PATH. Several optional variables are available for building configuration.

For a Release build with Ninja, run these commands from an **x64 Native Tools Command Prompt for Visual Studio** (or a Developer PowerShell configured for x64):

~~~powershell
cmake -S . -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/release --parallel 8
build/release/bin/kiraray.exe common/configs/example_cbox.json
~~~

The default CUDA architecture is `native`, targeting the installed GPU. Set `-DCMAKE_CUDA_ARCHITECTURES=120` to select an architecture explicitly, or an appropriate architecture for another machine. An explicit architecture is required when configuring without a GPU.

| Variable                | Default     | Description                                                  |
| ----------------------- | ----------- | ------------------------------------------------------------ |
| `OptiX_INSTALL_DIR`     | auto-detect | When auto-detection failed to find the correct OptiX path, this variable needs to be manually specified to point to the OptiX installation. |
| `KRR_RENDER_SPECTRAL`   | ON          | Whether to build spectral render. If turned OFF, the RGB renderer is build. |
| `KRR_PYTHON_PATH` | auto-detect | Manually specify this to enable Python binding for a specific version of Python. |
| `KRR_ENABLE_D3D12` | ON on Windows | Build the D3D12 graphics backend alongside Vulkan. |

#### Running

Specify the json configuration file as command line argument to start the renderer, as the example below. Check the [example configurations](common/configs) for some test scenes, showcasing core features like volumetric rendering, animated scenes and motion blur.

~~~bash
build/release/bin/kiraray.exe common/configs/example_cbox.json
~~~

> The two necessary entries in the configuration are `model` (specifying the relative path to the scene file) and `passes` (describing the render pipeline). Once compiled, directly run `kiraray` without specifying configuration (this [example configuration](common/configs/example_cbox.json) will be used) to get a feel for this toy renderer.

#### Usage

**Graphics API.** Vulkan is the default. Add `"graphics_api": "d3d12"` to a config
to use D3D12, including headless rendering and benchmarks. Both graphics backends
use the existing CUDA/OptiX integrators. See [graphics backends](src/core/graphics/README.md)
for implementation boundaries and interop ownership.

**Headless rendering and tests.** The Python API can render a config without a window and return
an owned RGB NumPy array. CTest runs a small CPU suite; local GPU smoke and image regression tests
are enabled with `-DKRR_ENABLE_GPU_TESTS=ON`. See [testing](tests/README.md) for build selection,
commands, artifacts, and reference updates, and [Python usage](common/scripts/README.md) for the API.

**Benchmarks and profiling.** The [benchmark runner](benchmarks/README.md) measures warmed headless
batches and captures Nsight Compute or Nsight Systems reports, with offline analysis and an optional
Nsight Python adapter. Use an optimized build for performance comparisons.

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
