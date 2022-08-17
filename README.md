# KiRaRay

*KiRaRay* is a simple interactive ray-tracing renderer using optix. It is mainly for personal learning purpose, with currently very limited features and some bugs.

<p align=center>
<img src=common/demo/kirara.jpg width="800">

### Features

> __Working in progress (?) project__  
> This project is only for learning purpose with very limited features, and not sure if it will continue developing.

- [x] Orbit camera controlling & thin lens camera.
- [x] Assimp as scene importer (OBJ and glTF2 scenes✅).
- [x] Diffuse, microfacet, disney and fresnel-blended bsdfs.
- [x] GPU path tracing (a megakernel version and a [wavefront](https://research.nvidia.com/publication/2013-07_megakernels-considered-harmful-wavefront-path-tracing-gpus) version).
- [x] Next event estimation and multiple importance sampling.
- [x] Post processing passes (e.g., tonemapping, accumulating and denoising).
- [x] Simple CPU/GPU performance profiling.

If possible, more features will be added in the future, for example:

- [ ] Spectral rendering.
- [ ] More BSDFs and light sources.
- [ ] Mesh instancing and scene animations.
- [ ] Importer for pbrt or mitsuba scenes.

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

#### Usage

**Camera controlling.** Dragging `LeftMouse` for orbiting, dragging `Scroll` or `Shift+LeftMouse` for panning. `Scroll` for zooming in/out.

**Hot keys.** `F1` for showing/hiding UI,  `F2` for showing/hiding the profiler, and `F3` for screen shots.

**Python binding.** Currently the python scripting is somewhat useless and can only start the renderer, for example:

~~~bash
python common/scripts/run.py --scene "common/assets/scenes/cbox/cbox.obj"
~~~

### Common Issues

#### Performance

Switch to *Release* build for normal performance! The megakernel pathtracer should run at about 30 spp per second at 1920*1080 on an RTX 3070, if the average path length is less than 5. The wavefront pathtracer however, expected to be faster than the megakernel version, is currently slightly slower due to my poor implementation (it does run significantly faster when next event estimation is enabled though). 

### Galleries

#### Bathroom

<p align=center>
<img src=common/demo/salle_de_bain.jpg width="700">

#### Higokumaru by [MooKorea](https://skfb.ly/ourA9)

<p align=center>
<img src=common/demo/higokumaru.jpg width="700">

### Credits
- The great optix tutorial for beginners: [optix7course](https://github.com/ingowald/optix7course).
- Some of the code (e.g., bsdf evaluation, wavefront path) are adapted from [pbrt](https://github.com/mmp/pbrt-v4) and [Falcor](https://github.com/NVIDIAGameWorks/Falcor). 
- *KiRaRay* implements a [tiny math library](https://github.com/cuteday/KiRaRay/tree/main/src/core/math) wrapper built upon [Eigen](http://eigen.tuxfamily.org/) for efficient vector/matrix arithmetic.
- [ImGui](https://github.com/ocornut/imgui) is used to build simple user interfaces for this project. 
- HDR environment images are from [sIBL Archive](http://www.hdrlabs.com/sibl/archive.html), and OBJ scenes are from [McGuire's Archive](https://casual-effects.com/data/) for demo images.

### Epilogue

Although the main purpose of this project is to let me (a beginner) learn c++ and optix, I really wish to add more features and make it a fully-functional path-tracing renderer. However, it may be a long process and I don't know if I will continue to do it.  Since in reality i'm acting like a lazy old uncle, trying to sleep as more as possible (\*/ω＼\*).

<details>
<summary>blue archive meme</summary>

I wish i was like: 

<p align=center>
<img src=https://cutesail.com/wp-content/uploads/2022/02/aris-meme.jpg width="500">

But actually i'm like **sleeps all the time**:

<p align=center>
<img src=https://cutesail.com/wp-content/uploads/2022/02/hoshino-meme.png width="320">

</details>