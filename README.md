# KiRaRay

*KiRaRay* is an ray-tracing renderer using optix. The purpose of this project is to let me (a beginner) learn c++ and optix in a more interesting way. The initial expectation of this project was to extend [optix7course](https://github.com/ingowald/optix7course) to a functional simple path tracer. Currently it has very limited features and many bugs.

<p align=center>
<img src=common/demo/kirara.png width="650">

### Features

> __Working in progress (?) project__  
> This project is only for learning purpose with very limited features, and not sure if it will continue developing.

- Orbit camera controlling.
- Assimp as scene importer.
- Diffuse (only) materials and sampling. 

If possible, more features will be added in the future, for example:

- More BSDFs.
- More light sources, sampling via NEE.
- Mesh instancing and scene animations.

### Build and run

#### Requirements

This project is only tested with OptiX 7.3, CUDA 11.3, compiled on Windows using MSVC, on a device with an ampere gpu. But it is expected to run on platforms with OptiX 7.0+ and CUDA 11.0. 

#### Build

This project uses cmake to build, no additional settings is needed. 


### Common Issues

#### Choosing generator for Visual Studio

Please do note that one should choose Ninja as generator when build with Visual Studio, otherwise some strange CUDA errors will occur at runtime. This problem has totally gone beyond my knowledge and takes me a day to find it out o(*≧▽≦)ツ┏━┓.

### Epilogue

I realy wish to add more features and make it a fully-functional path-tracing renderer. However, it may be a long process and I don't know if I will continue to do it.  Since in reality i'm like a lazy old uncle, tring to sleep as more as possible (\*/ω＼\*).

I wish i was like: 

<details>
<summary>aris meme</summary>

<p align=center>
<img src=https://cutesail.com/wp-content/uploads/2022/02/aris-meme.jpg width="500">

</details>

But actually i'm like \***sleeps all the time**\*:

<details>
<summary>hoshino meme</summary>

<p align=center>
<img src=https://cutesail.com/wp-content/uploads/2022/02/hoshino-meme.png width="350">


</details>
