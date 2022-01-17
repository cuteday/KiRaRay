# KiRaRay

### Building

#### Common Issues

[If reporting “assimp/config.h”: No such file or directory] The building process of Assimp generates the necessasy `include/assimp/config.h`. So run CMakeLists.txt at the Assimp root directory or copy that file in you build directory to `include/assimp`.

[Visual Studio intellisense for CUDA] `CUDA_INCLUDE_DIRECTORIES` invalid, use `CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES` for a workaround. Refer [this](https://gitlab.kitware.com/cmake/cmake/-/issues/19229)