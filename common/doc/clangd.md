# clangd in VS Code and Cursor

The repository ignores `.clangd` and `.vscode/`. Create these files locally using
the examples below; adapt paths to your machine and keep them untracked. These
settings affect editor parsing only. MSVC and NVCC still compile the renderer.

## Install and select clangd

Install the **clangd** (`llvm-vs-code-extensions.vscode-clangd`) and **CMake Tools**
(`ms-vscode.cmake-tools`) extensions. If the Microsoft C/C++ extension is also
installed, disable its IntelliSense engine for this workspace to avoid duplicate
diagnostics; its debugger can still be used in VS Code.

Clangd **22.1.6** was tested with CUDA 13.2. Download
`clangd-windows-22.1.6.zip` from the official
[22.1.6 release](https://github.com/clangd/clangd/releases/tag/22.1.6), and extract it
under `build/tools`, giving `build/tools/clangd_22.1.6/bin/clangd.exe`. Keep its
`lib` directory alongside `bin`. An existing installation can also be used by
adjusting `clangd.path` below. Older clangd versions bundled with Visual Studio
may fail to parse CUDA 13 headers.

From the repository root, verify the selected executable:

```powershell
& ./build/tools/clangd_22.1.6/bin/clangd.exe --version
```

## Generate and select the compilation database

Run CMake from the same x64 Visual Studio developer shell used to build the
project. These examples use `build/release`; substitute your own Ninja build
directory consistently. Select the desired CUDA/OptiX versions as described in
the [build instructions](../../README.md#building).

```powershell
cmake -S . -B build/release -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build/release --target kiraray --parallel 8
```

Ninja generates `build/release/compile_commands.json`; building also creates
headers used by the renderer. Reconfigure after changing CMake options or source
files so clangd sees the current include paths and build variant.

Create or merge the following into your local `.vscode/settings.json`:

```json
{
    "cmake.generator": "Ninja",
    "cmake.buildDirectory": "${workspaceFolder}/build/release",
    "cmake.configureSettings": {
        "CMAKE_EXPORT_COMPILE_COMMANDS": true
    },
    "clangd.path": "${workspaceFolder}/build/tools/clangd_22.1.6/bin/clangd.exe",
    "clangd.arguments": [
        "--compile-commands-dir=${workspaceFolder}/build/release",
        "--background-index",
        "--completion-style=detailed",
        "--header-insertion=never"
    ],
    "files.associations": {
        "*.cu": "cuda-cpp",
        "*.cuh": "cuda-cpp"
    }
}
```

The explicit compilation database directory matters because `build/release`
is outside clangd's normal parent-directory search. See the official
[editor and project setup guide](https://clangd.llvm.org/installation).

## Adapt NVCC commands for clangd

Create `.clangd` at the repository root with the following YAML. Replace the
`--cuda-path` value with the toolkit selected by CMake, including when switching
back to CUDA 12. Use a literal path with forward slashes here; VS Code's
`${workspaceFolder}` substitutions do not apply inside `.clangd`.

```yaml
If:
  PathExclude: src/ext/.*
CompileFlags:
  Add:
    - --cuda-path=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2
    - -std=c++17
    - -ferror-limit=0
  Remove:
    - -forward-unknown-to-host-compiler
    - --forward-unknown-to-host-compiler
    - -Xcompiler*
    - /wd*
    - /D_ENABLE_EXTENDED_ALIGNED_STORAGE
    - -Xcudafe*
    - --diag_suppress=*
    - -arch=*
    - --generate-code=*
    - -rdc=*
    - --use_fast_math
    - --expt-relaxed-constexpr
    - --extended-lambda
    - -lineinfo
    - -ptx
    - --ptx
    - --skip-ptx-semantics-check
---
If:
  PathMatch:
    - src/core/light\.cpp
    - src/render/(media|color|spectrum)\.cpp
    - src/render/wavefront/(integrator|medium)\.cpp
    - src/render/passes/(denoise/denoise|errormeasure/errormeasure)\.cpp
    - src/util/tables\.cpp
    - src/misc/render/ppg/(integrator|medium)\.cpp
CompileFlags:
  Add: [-xcuda]
```

The first fragment keeps CMake's includes and defines while removing NVCC-only
options and their forwarded MSVC warning flags. `-ferror-limit=0` lets parsing
continue after compatibility diagnostics. The second fragment restores CUDA
language mode for the `.cpp` files marked `LANGUAGE CUDA` in
[`common/build/source.cmake`](../build/source.cmake); clangd discards NVCC's
`-x cu`. Keep this list aligned when adding CUDA-compiled `.cpp` files. Ordinary
`.cu` files already use CUDA mode.

These fragments use clangd's documented
[flag removal and path conditions](https://clangd.llvm.org/config.html#compileflags).
The toolkit override follows the official
[CUDA guidance](https://clangd.llvm.org/faq#does-clangd-support-cuda).

## Check the setup

Run **clangd: Restart language server**, then open a renderer source file. In
**View → Output → Clang Language Server**, check that the reported clangd version
and compilation database match your selections. For a command-line check:

```powershell
& ./build/tools/clangd_22.1.6/bin/clangd.exe --check=src/core/logger.cpp --compile-commands-dir=build/release
```

Missing generated headers usually mean the selected build has not been built.
Unknown NVCC options suggest `.clangd` was not loaded or needs an additional
filter. Some CUDA files still produce Clang/NVCC host-device declaration or macro
diagnostics; a successful NVCC build remains the authoritative compiler check.
Changing `--cuda-path` only changes clangd's parsing, so keep it consistent with
the toolkit recorded in the selected build's CMake cache.
