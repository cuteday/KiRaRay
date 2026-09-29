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

Render Properties groups KiRaRay controls into **Sampling**, **Light Paths**, and
**System**. Sampling contains final/viewport sample counts and the seed. Light
Paths exposes **Maximum Path Depth** (default 10); zero includes only directly
visible emission and the environment. Its **Advanced** subsection contains
**Next Event Estimation** (on by default) and **Russian Roulette Survival**
(default 0.8). Survival 1 disables Russian roulette. Lower survival probabilities
reduce path work but can increase noise; disabling light sampling can greatly
increase noise in scenes lit by small emitters. These settings apply to F12 and
the viewport. Changing them restarts accumulation and retains scene geometry.

System contains **Graphics API** and **Asset Root**. Both Vulkan and D3D12 use an offscreen device without a
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

The supported surface graph includes a directly connected **Principled BSDF**,
or **Diffuse**, **Glossy**, **Glass**, and **Emission** shaders combined through
**Mix Shader** and **Add Shader**. The latter use a bounded mixture of native
BSDFs rather than preparing OpenPBR for each simple material. The compiler
retains all active scattering components and evaluates their combined value
and sampling PDF. The limit is eight scattering components after graph pruning.
Principled inside Mix/Add remains unsupported because Blender expands it into
additional layer closures; keep Principled connected directly to Surface.

These are deliberate approximations of Blender shading, not Cycles parity:

| Blender shader | KiRaRay evaluation |
|---|---|
| Diffuse | Lambertian; nonzero diffuse roughness is ignored |
| Glossy | Isotropic GGX conductor; anisotropic roughness axes are averaged and Fresnel can differ from Cycles Glossy |
| Glass | GGX dielectric with reflection/refraction, including zero roughness |
| Multiscatter GGX | Existing GGX model without Cycles' multiscatter compensation |
| Emission | Independent emission, including weighted Mix/Add emission |

Image textures support Linear, Closest, and Cubic filtering. Shader parameters
use the same supported image, UV, normal-map, and numeric expression nodes as
Principled. Unsupported distributions, procedural shading, layer closures, and
other unverified nodes retain the diagnostic material rather than guessing.

A final Mix Shader may combine **constant white Transparent BSDF** with one
supported opaque subtree. KiRaRay represents this as stochastic surface
coverage, consistently applied to camera, continuation, shadow, and emitter
sampling paths. This does not randomly select one material in a general BSDF
mixture. Blender 5.2.2 misexports this graph as zero surface opacity and a white
diffuse branch; the extension supplies an explicit `opaqueMixBranch` correction
through Hydra settings and validated USD metadata. Source nodes are never
modified. Colored transparency, nested Transparent shaders, and mixing two
Transparent shaders remain unsupported. Ordinary unannotated USD cannot recover
the Blender shader information already lost by its exporter.

The Combined pass contains linear RGB. The optional Depth pass contains positive
camera-space Z from primary geometry, with infinity for misses. Authored opacity
below 0.5 does not contribute to depth; legacy opacity follows its existing
coverage rule. The delegate also supplies normalized projected depth to Hydra.
Blender 5.2's CPU Hydra viewport presentation uploads only color, so correct
occlusion of Blender overlays by that depth is not available through this path.

Native Hydra clients can set `krr:maxDepth` (nonnegative integer), `krr:nee`
(boolean), and `krr:rr` (finite survival probability in `(0, 1]`). Invalid values
produce an actionable error and retain the last accepted setting; reapplying a
valid setting clears the error and retries rendering. Completion status JSON includes their effective
values under `wavefront`, using the native config keys `max_depth`, `nee`, and
`rr`; the KiRaRay log prints the same values when accumulation resets.

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

Check the Render Properties panels in a windowed host without rendering:

```powershell
blender.exe --factory-startup --enable-event-simulate --window-geometry 0 0 1000 1100 `
  --python integrations/blender/probe/panels.py -- `
  --addon-dir build/blender-host/blender/kiraray `
  --artifacts build/blender-host/tests/artifacts/RelWithDebInfo/blender_panels
```

This checks actual panel draw callbacks after registration and re-registration,
expanding collapsed panels in the test window. It saves a screenshot and
`result.json`, or `failure.txt` on failure, then closes its Blender window.

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

Inspect and export an external acceptance scene without modifying its blend file:

```powershell
blender.exe --background --factory-startup --disable-autoexec --python-exit-code 1 `
  --python integrations/blender/probe/material_scene.py -- `
  --scene D:/Projects/scenes_blender-main/kitchen/kitchen.blend `
  --artifacts build/blender-host/tests/artifacts/RelWithDebInfo/kitchen
```

This writes `preflight.json`, an annotated `scene.usdc`, a standalone headless
`config.json`, and export results. Add `--preflight-only` to skip export. Add
`--render --addon-dir build/blender-host/blender/kiraray --samples 16
--resolution 320 180` to perform a GPU acceptance render through the scene's
perspective camera. Rendering checks finite, nonblack output, primary depth,
completion, and native material diagnostics; it retains EXR, PNG, and status
artifacts. Use the other external scenes as additional probes rather than
checking large blend assets into the test suite. The source file is never saved.

The synthetic ABI probe is independent of CUDA and verifies plugin loading,
color/depth channel orientation, render passes, and repeated host teardown before
the full renderer is involved.
