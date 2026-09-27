# Windows integration builds

Run these commands from the repository root in ordinary PowerShell:

```powershell
.\common\build\windows.ps1 -Preset blender -Action configure
.\common\build\windows.ps1 -Preset blender -Action build
.\common\build\windows.ps1 -Preset blender -Action test
```

`build` also configures, then creates the Blender extension ZIP. `test` builds all
targets, refreshes the ZIP, and runs CPU tests. The wrapper discovers Visual
Studio 2022 and initializes its x64 compiler environment on every invocation.
An existing cache keeps its exact compiler; a new build defaults to MSVC 14.44.
Use `-Jobs 8` to change parallelism or `-Toolset 14.44` to select a toolset for a
new build. An override that conflicts with an existing cache fails clearly.

| Preset | Build directory | Outputs |
| --- | --- | --- |
| `blender` | `build/blender-interop/host` | Native Hydra delegate and Blender extension; USD enabled |
| `usd` | `build/blender-interop/standalone` | Standalone executable and Python module with USD import |

The package is `build/blender-interop/host/blender/kiraray-blender-5.2-windows-x64.zip`.
Close Blender processes using the build's DLL before rebuilding it. An installed
extension is a separate copy and must be updated from the new ZIP; native changes
require a full Blender restart. See the [Blender guide](../../integrations/blender/README.md).

## SDK discovery and overrides

The SDKs are optional for ordinary KiRaRay builds. These presets reuse installed
SDKs; they do not download or build them during CMake configuration.

All builds fetch hash-pinned OpenEXR 3.4.13 and Imath 3.2.2 sources on first
configuration for EXR decoding, including Blender's DWAA/DWAB studio lights.
They build into private static libraries; no additional OpenEXR DLLs need to be
installed with the Blender extension. Downloads and generated files stay in the
selected build directory's `_deps` folder. Offline builds can supply source trees
with `FETCHCONTENT_SOURCE_DIR_KRR_OPENEXR` and `FETCHCONTENT_SOURCE_DIR_KRR_IMATH`.

| Setting | Managed location, relative to the repository |
| --- | --- |
| `KRR_BLENDER_SDK` | `build/blender-5.2/sdk` |
| `KRR_BLENDER_EXECUTABLE` | `build/blender-5.2/blender-5.2.2-windows-x64/blender.exe` |
| `KRR_USD_ROOT` | `build/usd-sdk/standalone/install` |

Explicit CMake selections take precedence, followed by environment variables of
the same name, then the existing managed locations. Cached selections remain in
effect until explicitly changed. Blender's executable is needed to run host tests;
the SDK is enough to build the package. Acquisition instructions are in the
[Blender SDK guide](../../integrations/hydra/probe/README.md) and
[standalone USD guide](../../integrations/usd/README.md).

CUDA, Vulkan, and OptiX retain their normal discovery rules. Select versions and
external SDK locations once when configuring:

```powershell
.\common\build\windows.ps1 -Preset blender -Action configure -CMakeArgs @(
    '-DKRR_BLENDER_SDK=D:/SDK/blender-5.2.2',
    '-DKRR_BLENDER_EXECUTABLE=D:/Apps/blender-5.2.2/blender.exe',
    '-DOptiX_INSTALL_DIR=C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0'
)
```

Machine-specific presets may also inherit the tracked presets in an ignored
`CMakeUserPresets.json`; use those with CMake directly from a developer terminal.
The wrapper accepts the two supplied integration presets.

SDK and GPU tests remain opt-in. Enable them without repeating the other options:

```powershell
.\common\build\windows.ps1 -Preset blender -Action configure -CMakeArgs @(
    '-DKRR_ENABLE_SDK_TESTS=ON', '-DKRR_ENABLE_GPU_TESTS=ON'
)
.\common\build\windows.ps1 -Preset blender -Action test
ctest --test-dir build/blender-interop/host -C RelWithDebInfo -L blender --output-on-failure
```

The wrapper's `test` action runs only the CPU label. The last command explicitly
selects the Blender integration tests, including GPU renders when enabled.

## Direct CMake commands

In an initialized x64 developer terminal, the equivalent short commands are:

```powershell
cmake --preset blender
cmake --build --preset blender-package --parallel 4
```

For a fully explicit configure using the managed SDK locations and the tested
CUDA 13.2 / OptiX 8 setup, run from the repository root after initializing the
developer terminal:

```powershell
$env:CUDA_PATH = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2'
$env:VULKAN_SDK = 'C:/VulkanSDK/1.4.313.1'
cmake -S . -B build/blender-interop/host -G Ninja `
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CUDA_ARCHITECTURES=native `
  "-DCMAKE_CUDA_COMPILER=$env:CUDA_PATH/bin/nvcc.exe" `
  "-DCUDAToolkit_ROOT=$env:CUDA_PATH" `
  '-DOptiX_INSTALL_DIR=C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0' `
  -DKRR_ENABLE_HYDRA=ON -DKRR_ENABLE_USD=ON `
  -DKRR_ENABLE_PYTHON=OFF -DKRR_ENABLE_OPENVDB_IO=OFF `
  -DKRR_RENDER_SPECTRAL=ON -DKRR_ENABLE_D3D12=ON -DKRR_BUILD_STARLIGHT=OFF `
  "-DKRR_BLENDER_SDK=$PWD/build/blender-5.2/sdk" `
  "-DKRR_BLENDER_EXECUTABLE=$PWD/build/blender-5.2/blender-5.2.2-windows-x64/blender.exe" `
  -DBUILD_TESTING=ON -DKRR_ENABLE_SDK_TESTS=ON -DKRR_ENABLE_GPU_TESTS=ON
cmake --build build/blender-interop/host --target krr_blender_package --parallel 4
```

The last two test options are for local integration validation, not requirements
for packaging. Set `Python_EXECUTABLE` if the packaging/test interpreter needs an
explicit selection. It need not be Blender's Python.

## Why the profiles differ

`KRR_ENABLE_PYTHON=OFF` disables KiRaRay's standalone `pykrr` extension. Blender
already supplies Python and its patched USD Python runtime; the delegate links
KiRaRay's C++ library directly. It does not import `pykrr`. A regular Python
interpreter is still used for packaging and tests, but its development headers
and libraries are no longer required by this profile. The standalone USD preset
keeps `pykrr` enabled and uses a separate Python-free USD SDK.

`KRR_ENABLE_OPENVDB_IO=OFF` excludes the legacy bundled OpenVDB file loader and
its old TBB/Half dependencies, which must not conflict with the selected USD or
Blender runtime. NanoVDB and the CUDA volume-rendering runtime remain available.
Imported Blender/USD volumes remain outside the supported integration subset.
This is a dependency-isolation choice in the current implementation, not a
fundamental prohibition on combining Blender with OpenVDB.

Fresh Hydra builds select these defaults automatically and enable USD in the
CMake cache. Fresh standalone USD builds disable legacy OpenVDB I/O and retain
Python bindings. Ordinary builds keep their previous defaults. Explicit or
previously cached incompatible options still report an error; keep the two SDK
profiles in separate build directories.

## Missing standard headers with Ninja

Errors about `<array>` or `<cstdio>` usually mean `cl.exe` was found but the
Visual Studio environment was not initialized. Ninja needs `INCLUDE`, `LIB`, and
the Windows SDK settings at build time, even if CMake remembers an absolute
compiler path. Use the wrapper above or the x64 developer terminal for both
configuration and building. Adding individual system include directories to the
project is not a complete repair.
