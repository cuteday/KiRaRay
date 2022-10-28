# KiRaRay

*KiRaRay* is a simple interactive ray-tracing renderer using optix. It is mainly for personal learning purpose, with currently very limited features and some bugs.

<p align=center>
<img src=common/demo/kirara.jpg width="800">

### Features

> __Working in progress (?) project__  
> This project is only for learning purpose with very limited features, and not sure if it will continue developing.

- [x] Interactive orbit camera controlling.
- [x] GPU path tracing (a megakernel version and a wavefront version).
- [x] Post processing passes (e.g., tonemapping, accumulating and denoising).
- [x] Basic support for multiple scenes formats (e.g., OBJ, glTF2 and pbrt-v3).
- [x] Basic CPU/GPU performance profiling.

### Build and run

| *MSVC@Windows* | [![Building](https://github.com/cuteday/KiRaRay/actions/workflows/main.yml/badge.svg)](https://github.com/cuteday/KiRaRay/actions/workflows/main.yml) |
| -------------- | ------------------------------------------------- |

#### Requirements

- Nvidia gpu (Turing or higher if possible).
- OptiX 7.0+ and CUDA 11.0+ installed.

This project is only tested with optix 7.3/4 and cuda 11.4/5/6 on Windows (MSVC). It may not compile on Linux. 

#### Cloning the repository

*KiRaRay* uses external dependencies as submodules, so fetch them recursively with `--recursive` when cloning:

~~~bash
git clone --recursive --depth=1 https://github.com/cuteday/KiRaRay.git
~~~

#### Building

This project uses cmake to build, no additional setting is needed. Make sure cuda is installed and added to PATH. While it tries to guess the optix installation path, you may specify the `OptiX_INSTALL_DIR` environment variable manually in case it failed.

#### Running

Specify the json configuration file as command line argument to start the renderer. The sample configuration will used if no argument is provided, example:

~~~bash
build/src/kiraray.exe common/configs/example.json
~~~

One can also save the current parameters (including camera parameters, render passes and scene file path, etc.) to a configuration file via the option in main menu bar.

#### Usage

**Camera controlling.** Dragging `LeftMouse` for orbiting, dragging `Scroll` or `Shift+LeftMouse` for panning. `Scroll` for zooming in/out.

**Hot keys.** `F1` for showing/hiding UI,  `F2` for showing/hiding the profiler, and `F3` for screen shots.

**Python binding.** Several simple interfaces are exposed to python scripting via [pybind11](https://github.com/pybind/pybind11), including a wrapper of OptiX's built-in AI denoiser. See [scripts](scripts) for details.

### Galleries

<p align=center>
<img src=common/demo/kitchen.png width="800">

<p align=center>
<img src=common/demo/living-room.png width="800">

More visuals [here](https://cutesail.com/?p=493)!

### Issues

#### Performance

Switch to *Release* build for normal performance! The megakernel pathtracer should run at about 30 spp per second at 1920*1080 on an RTX 3070, if the average path length is less than 5. The [wavefront pathtracer](https://research.nvidia.com/publication/2013-07_megakernels-considered-harmful-wavefront-path-tracing-gpus) however, expected to be faster than the megakernel version, is currently slightly slower due to my poor implementation (it does run significantly faster when next event estimation is enabled though). 

#### Scene loading

*Kiraray* provided limited support for importing scenes like OBJ, glTF2 with [Assimp](https://github.com/assimp/assimp.git) as the default scene importer. Some commonly used material properties (e.g., roughness, metallic) and textures (normal, emission, opacity, etc.) are supported. [pbrt-parser](https://github.com/cuteday/pbrt-parser.git) is used to import [pbrt-v3](https://github.com/mmp/pbrt-v3/) scenes, and all pbrt materials are roughly approximated with the Disney Principled BSDF. Most of the scenes [here](https://benedikt-bitterli.me/resources/) could be loaded, while some of the materials might be visually biased.

### Credits
- The great optix tutorial for beginners: [optix7course](https://github.com/ingowald/optix7course).
- Some of the code (e.g., bsdf evaluation, wavefront path) are adapted from [pbrt](https://github.com/mmp/pbrt-v4) and [Falcor](https://github.com/NVIDIAGameWorks/Falcor). 
- *KiRaRay* implements a [tiny math library](https://github.com/cuteday/KiRaRay/tree/main/src/core/math) wrapper built upon [Eigen](http://eigen.tuxfamily.org/) for efficient vector/matrix arithmetic.
- [ImGui](https://github.com/ocornut/imgui) is used to build simple user interfaces for this project. 

### Epilogue

Although the main purpose of this project is to let me (a beginner) learn c++ and optix, I really wish to add more features and make it a fully-functional path-tracing renderer. However, it may be a long process and I don't know if I will continue to do it.  Since in reality i'm acting like a lazy old uncle, trying to sleep as more as possible (\*/ω＼\*).

For anyone that (accidentally) found this project: any questions, suggestions and bug reports are greatly appreciated!

</details>