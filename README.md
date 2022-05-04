# KiRaRay

*KiRaRay* is a simple interactive ray-tracing renderer using optix. The purpose of this project is to let me (a beginner) learn c++ and optix in a more interesting way. The initial expectation of this project was to extend [optix7course](https://github.com/ingowald/optix7course) to a functional simple path tracer, and some parts of the coding structures are inspired by [pbrt-v4](https://github.com/mmp/pbrt-v4). Currently it has very limited features and potentially some bugs.

<p align=center>
<img src=common/demo/kirara.jpg width="750">

### Features

> __Working in progress (?) project__  
> This project is only for learning purpose with very limited features, and not sure if it will continue developing.

- [x] Orbit camera controlling.
- [x] Assimp as scene importer.
- [x] Image-based environment lighting.
- [x] Diffuse, microfacet, disney and fresnel-blended bsdfs. 
- [x] Next event estimation and multiple importance sampling (naive ver.)
- [x] Tone mapping and frame accumulating.

If possible, more features will be added in the future, for example:

- [ ] Spectral rendering.
- [ ] Wavefront path tracer.
- [ ] More BSDFs and light sources.
- [ ] Mesh instancing and scene animations.
- [ ] Importer for pbrt or mitsuba scenes.

### Build and run

#### Requirements

- Nvidia gpu (Turing or higher if possible).
- OptiX 7.0+ and CUDA 11.0+ installed.

This project is tested with optix 7.3/4 and cuda 11.4/5/6. It may not compile successfully on Linux, since it is only tested on Windows (MSVC) currently. 

#### Build

This project uses cmake to build, no additional setting is needed. Make sure cuda is installed and added to PATH. While it tries to guess the optix installation path, you may specify the `OptiX_INSTALL_DIR` environment variable manually in case it failed.

### Common Issues

#### Choosing generator for Visual Studio

Ninja is recommended as generator when build with Visual Studio, otherwise some strange CUDA errors may be observed at runtime.

#### Test scenes and assets

For testing, HDR environment images can be downloaded at [sIBL Archive](http://www.hdrlabs.com/sibl/archive.html), OBJ scenes can be obtained at [McGuire's Archive](https://casual-effects.com/data/).

#### Performance

Currently the renderer should run at 30+ fps for rendering an 1 spp 1920*1080 image on an RTX 3060 or similar GPU, if the average path length is less than 5 and NEE disabled.

### Epilogue

I really wish to add more features and make it a fully-functional path-tracing renderer. However, it may be a long process and I don't know if I will continue to do it.  Since in reality i'm acting like a lazy old uncle, trying to sleep as more as possible (\*/ω＼\*).

<details>
<summary>blue archive meme</summary>

I wish i was like: 

<p align=center>
<img src=https://cutesail.com/wp-content/uploads/2022/02/aris-meme.jpg width="500">

But actually i'm like \***sleeps all the time**\*:

<p align=center>
<img src=https://cutesail.com/wp-content/uploads/2022/02/hoshino-meme.png width="320">

</details>

### Galleries

#### Cornell box

<p align=center>
<img src=common/demo/cbox.jpg width="700">

#### Salle de bain

<p align=center>
<img src=common/demo/salle_de_bain.jpg width="700">

#### Breakfast room

<p align=center>
<img src=common/demo/breakfast_room.jpg width="700">

#### Rungholt

<p align=center>
<img src=common/demo/rungholt.jpg width="700">
