# Blender 5.2 integration

KiRaRay runs in Blender through a native Hydra render delegate. Blender supplies
evaluated meshes, instances, transforms, cameras, lights, and MaterialX networks;
the delegate uses the same native scene/material conversion as the USD importer.
The extension does not import `pykrr`, and its Python version is independent of
the standalone bindings.

## Build and install

For the managed local SDK layout, configure and package from ordinary PowerShell:

```powershell
.\common\build\windows.ps1 -Preset blender -Action build
```

This initializes the matching Visual Studio environment and discovers the existing
SDK paths. It creates the ZIP beneath `build/blender-interop/host/blender/`.
See the [Windows build guide](../../common/build/README.md) for presets, explicit
configure commands, path overrides, and the Python/OpenVDB dependency choices.

The tested host is Windows x64 Blender **5.2.2 LTS**, revision `d13f752e3b9c`,
with Blender's USD 26.03. Use the matching development bundle described in
[the ABI probe instructions](../hydra/probe/README.md). A generic OpenUSD SDK
cannot replace that bundle: Blender uses its own USD namespace and Python ABI.

Configure a separate MSVC 14.44 build with:

```powershell
cmake -S . -B build/blender-host -G Ninja `
  -DCMAKE_BUILD_TYPE=RelWithDebInfo `
  -DKRR_ENABLE_HYDRA=ON -DKRR_ENABLE_PYTHON=OFF `
  -DKRR_ENABLE_OPENVDB_IO=OFF -DKRR_BUILD_STARLIGHT=OFF `
  -DKRR_RENDER_SPECTRAL=ON -DKRR_BLENDER_SDK=D:/SDK/blender-5.2.2
cmake --build build/blender-host --target krr_blender_package
```

Set the CUDA, OptiX, and Vulkan SDK paths as for an ordinary KiRaRay build.
Use Blender's release CRT even with debug symbols. Legacy OpenVDB file loading
is disabled in this host build; NanoVDB runtime types remain available.

Install `build/blender-host/blender/kiraray-blender-5.2-windows-x64.zip` using
Blender's **Install from Disk**, then select **KiRaRay** in Render Properties.
The package contains the delegate, its DXC dependency, and MaterialX XML
definitions. Do not add private USD, TBB, MaterialX, or Python DLLs beside the
plugin; Blender supplies those.

For development, add `build/blender-host/blender` to Blender's `sys.path` and
call `import kiraray; kiraray.register()`. Rebuild the package after Python edits.
Close Blender instances using the plugin before replacing its native DLL.
After reinstalling the ZIP, fully restart Blender so Hydra loads the new DLL.
`KRR_HYDRA_PLUGIN_DIR` can override the directory containing `plugInfo.json`.

## Rendering

F12 renders progressively to the requested sample count. Rendered viewport mode
keeps one native session and restarts accumulation after edits. Camera changes
reset accumulation; transform edits update instances; material edits update
material resources; geometry changes rebuild scene geometry. Final rendering
takes priority over a viewport. One rendered viewport is supported at a time.

Set **Render Samples**, **Viewport Samples**, **Seed**, and **Graphics API** in
Render Properties. Both Vulkan and D3D12 use an offscreen device without a
KiRaRay window. Scenes must use the blend file's **Linear Rec.709** working
space. Blender handles display transforms such as AgX. Perspective cameras are
the initial supported camera profile.

EXR textures use OpenEXR Core, including the DWAB-compressed studio environments
used by Material Preview. The decoder is included in the native plugin; no extra
OpenEXR installation or conversion of Blender's studio lights is required.

Unsupported connected shader features produce a magenta error material and
actionable warnings in the render log and KiRaRay panel. No source node graphs
are modified. **Export Validated USD** records the same producer diagnostics on
USD materials so offline imports cannot silently hide an unsupported graph.

The Combined pass contains linear RGB. The optional Depth pass contains positive
camera-space Z from primary geometry, with infinity for misses. Authored opacity
below 0.5 does not contribute to depth; legacy opacity follows its existing
coverage rule. The delegate also supplies normalized projected depth to Hydra.
Blender 5.2's CPU Hydra viewport presentation uploads only color, so correct
occlusion of Blender overlays by that depth is not available through this path.

## Validation

Enable host integration tests explicitly:

```powershell
cmake -S . -B build/blender-host -DBUILD_TESTING=ON `
  -DKRR_ENABLE_SDK_TESTS=ON -DKRR_ENABLE_GPU_TESTS=ON `
  -DKRR_BLENDER_EXECUTABLE=D:/Apps/blender-5.2.2/blender.exe
cmake --build build/blender-host --target krr_blender_package
ctest --test-dir build/blender-host -L blender --output-on-failure
```

The background tests validate repeated Cornell renders, camera edits, genuine
depth, producer checks, emission units against Cycles, and diagnostic-preserving
USD export. Rendering uses
CTest's shared `krr_gpu` lock. Reports, EXRs, configs, and logs remain under the
selected build's `tests/artifacts/<configuration>/blender_*` directories.

Run the windowed viewport acceptance harness separately:

```powershell
blender.exe --factory-startup --enable-event-simulate --window-geometry 0 0 480 480 `
  --python integrations/blender/probe/viewport.py -- `
  --addon-dir build/blender-host/blender/kiraray `
  --artifacts build/blender-host/tests/artifacts/RelWithDebInfo/blender_viewport
```

It checks incremental material/transform/geometry updates, timeline changes,
error-material feedback, final-render handoff, file reload, and cancellation.
It saves a screenshot and JSON counters, then closes its Blender window.
Inspect `result.json`; `failure.txt` indicates a failed viewport assertion.
The harness temporarily displays only Blender's running-job widget in its own
status bar and clicks its stop button. Blender's simulated Escape events bypass
the raw keyboard cancellation hook, so they cannot test render cancellation.

The synthetic ABI probe is independent of CUDA and verifies plugin loading,
color/depth channel orientation, render passes, and repeated host teardown before
the full renderer is involved.
