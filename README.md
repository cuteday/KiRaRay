# KiRaRay

*KiRaRay* is a simple interactive ray-tracing renderer using optix. The initial expectation of this project was to extend [optix7course](https://github.com/ingowald/optix7course) to a functional simple path tracer, and many parts of the coding structures are inspired by [pbrt-v4](https://github.com/mmp/pbrt-v4). Currently it has very limited features and potentially some bugs.

<p align=center>
<img src=common/demo/kirara.jpg width="750">

### Features

> __Working in progress (?) project__  
> This project is only for learning purpose with very limited features, and not sure if it will continue developing.

- [x] Assimp as scene importer.
- [x] Orbit camera controlling & thin lens camera.
- [x] Diffuse, microfacet, disney and fresnel-blended bsdfs.
- [x] GPU path tracing (a megakernel version and a [wavefront](https://research.nvidia.com/publication/2013-07_megakernels-considered-harmful-wavefront-path-tracing-gpus) version).
- [x] Next event estimation and multiple importance sampling (naive version).
- [x] Post processing passes (tone mapping and frame accumulating).
- [x] Simple CPU/GPU profiling.

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

This project is tested with optix 7.3/4 and cuda 11.4/5/6. It may not compile successfully on Linux, since it is only tested on newer versions of Windows (MSVC). 

#### Build

This project uses cmake to build, no additional setting is needed. Make sure cuda is installed and added to PATH. While it tries to guess the optix installation path, you may specify the `OptiX_INSTALL_DIR` environment variable manually in case it failed.

#### Usage

**Camera controlling.** Dragging `LeftMouse` for orbiting, dragging `Scroll` or `Shift+LeftMouse` for panning. `Scroll` for zooming in/out.

**Hot keys.** `F1` for showing/hiding UI,  `F2` for showing/hiding the profiler, and `F3` for screen shots.

### Common Issues

#### Choosing generator for Visual Studio

Ninja is recommended as generator when build with Visual Studio, otherwise some strange CUDA errors may be observed at runtime.

#### Test scenes and assets

For testing, HDR environment images can be downloaded at [sIBL Archive](http://www.hdrlabs.com/sibl/archive.html), OBJ scenes can be obtained at [McGuire's Archive](https://casual-effects.com/data/). Also glTF2 models can usually be loaded and rendered. 

#### Performance

The megakernel pathtracer should run at 30+ fps for rendering an 1 spp 1920*1080 image on an RTX 3060 or similar GPU, if the average path length is less than 5 and NEE disabled. The wavefront pathtracer however, expected to be faster than the megakernel version, is currently slightly slower due to my poor implementation (it does runs significantly faster when next event estimation is enabled though). 

### Epilogue

Although the initial purpose of this project is to let me (a beginner) learn c++ and optix in a more interesting way, I really wish to add more features and make it a fully-functional path-tracing renderer. However, it may be a long process and I don't know if I will continue to do it.  Since in reality i'm acting like a lazy old uncle, trying to sleep as more as possible (\*/ω＼\*).

<details>
<summary>blue archive meme</summary>

I wish i was like: 

<p align=center>
<img src=https://cutesail.com/wp-content/uploads/2022/02/aris-meme.jpg width="500">

But actually i'm like **sleeps all the time**:

<p align=center>
<img src=https://cutesail.com/wp-content/uploads/2022/02/hoshino-meme.png width="320">

</details>

### Galleries

#### Salle de bain

<p align=center>
<img src=common/demo/salle_de_bain.jpg width="700">

#### Higokumaru by [MooKorea](https://skfb.ly/ourA9)

<p align=center>
<img src=common/demo/higokumaru.jpg width="700">

[More](https://cutesail.com/?p=493) renderings...