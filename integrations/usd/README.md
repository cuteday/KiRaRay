# Standalone USD support

USD is optional. Ordinary builds keep the current scene loaders, Python module,
and legacy materials. USD builds require a separate, Python-free OpenUSD 26.03
SDK; the Blender delegate uses Blender's patched SDK instead.

If the SDK already exists in the managed location, ordinary PowerShell can build
the integration with `.\common\build\windows.ps1 -Preset usd -Action build`.
The wrapper initializes Visual Studio and uses the `usd` CMake preset. SDK paths
are discovered automatically or can be overridden; see the
[Windows build guide](../../common/build/README.md).

From an x64 MSVC 14.44 developer shell:

```powershell
python integrations/usd/build_sdk.py --parallel 4
cmake -S . -B build/usd -G Ninja `
  -DCMAKE_BUILD_TYPE=RelWithDebInfo `
  -DKRR_ENABLE_USD=ON -DKRR_ENABLE_OPENVDB_IO=OFF `
  -DKRR_USD_ROOT="$PWD/build/usd-sdk/standalone/install" `
  -DPython_EXECUTABLE="C:/path/to/python.exe" `
  -DKRR_ENABLE_SDK_TESTS=ON
cmake --build build/usd --parallel 4
ctest --test-dir build/usd -L cpu --output-on-failure
```

The SDK builder pins source versions and checksums and writes a dependency
manifest to `install/krr-sdk.json`. Its configure/build logs stay in the SDK
build directory. It does not install into Blender or the selected Python.
Do not mix the two SDK profiles in one build directory.

Use a `.usd`, `.usda`, or `.usdc` path in the existing `model` field. Scene
import nodes also accept `params.time_code` and `params.camera_path`. Omitted
time selects the stage start time. An explicit JSON camera wins over the
selected USD camera; otherwise the first supported camera in path order is
used. Dependencies resolve through USD's asset resolver without changing the
working directory. Existing `krr.render()` and `HeadlessRenderer.render()`
remain fresh-batch operations.

For example, use this scene entry in an existing rendering config:

```json
"scene": {
  "model": [{
    "model": "scenes/example.usdc",
    "params": {"time_code": 12, "camera_path": "/Scene/Camera"}
  }]
}
```

Select the USD build with `KRR_BUILD_DIR`, then use `krr.render(config, frames=128,
seed=0)` or `common/scripts/render_headless.py` as before. Omit `params` to use
the stage defaults. Paths can be absolute or relative to the configured asset
root; the Python process keeps its original working directory.

MaterialX OpenPBR and the supported USD Preview Surface subset have distinct
native model identities. An authored MaterialX surface takes priority over
Preview Surface. Unsupported active graphs become magenta diagnostic
materials and emit warnings. Import does not silently fall back to a different
surface model. Only linear Rec.709 working colors are supported; supported
image inputs may explicitly use sRGB or linear/raw data.
Preview Surface supports opaque materials, `presence` opacity, and threshold
cutouts. Nonopaque `transparent` mode is diagnosed because it retains a lighting
response that coverage opacity cannot reproduce. Authored occlusion is also
diagnosed. USD texture defaults use constant zero coordinates and black wrapping
when no wrapping metadata is present; EXR `wrapmodes` metadata is honored. Missing
image files remain diagnostic rather than silently hiding asset-resolution errors.
Imported UV meshes use MikkTSpace tangents. OpenPBR emission is authored in
nits, with 1,000 nits mapped to one renderer radiance unit. Blender 5.2.2 exports
Principled strength directly; the Hydra adapter and validated export apply an
explicit factor of 1,000 at material import. Validated USD materials carry
`kiraray:emissionLuminanceScale` custom data, including for connected strength
graphs. Ordinary unannotated USD keeps its authored nits. Preview Surface and
legacy emission keep their existing scene-linear units. Imported analytic area lights are invisible to primary
camera rays; authored emissive meshes remain visible.

Use Blender's **KiRaRay Validated USD** export to catch information that the
upstream exporter would otherwise discard. The command writes diagnostics to
the affected material's `kiraray:diagnostics` custom data. An ordinary USD file
cannot reveal Blender nodes that another exporter already removed.

The initial importer reads selected-time polygon snapshots, face-varying UVs
and normals, material subsets, native instances, point instances, affine
transforms, perspective cameras, and common lights. It converts stage units
to meters and Z-up stages to KiRaRay's Y-up convention. Subdivision uses the
authored polygon cage with a warning; camera depth of field/exposure and
finite-radius spot or distant-light angular sizes are currently diagnosed.
Imported volumes, curves/hair, skeletal evaluation, and motion blur are not
implemented. Volume-file loading is deliberately unavailable in this profile;
the NanoVDB runtime remains available to the renderer.

SDK-dependent CPU tests are selected with `KRR_ENABLE_SDK_TESTS`. They include
actual Blender 5.2.2 graph fixtures and stage diagnostics. GPU tests remain
opt-in through `KRR_ENABLE_GPU_TESTS`. Generated results go beneath the
selected build directory's `tests/artifacts/<configuration>/`.

The `cornell_usd` case exercises the existing Python API with an actual Blender
export, OpenPBR materials, a USD camera, and a normalized area light. Its spectral
reference follows the same deliberate generation and multi-seed calibration
policy as the legacy Cornell case:

```powershell
$env:KRR_BUILD_DIR = "$PWD/build/usd"
python tests/generate_reference.py --case tests/cases/cornell_usd `
  --artifacts build/usd/tests/artifacts/RelWithDebInfo/usd_reference_generation
```
