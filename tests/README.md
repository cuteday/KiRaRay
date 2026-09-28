# Tests

CTest runs small CPU tests in every build and optional local GPU tests. GPU tests use the
public Python API and a dedicated wavefront Cornell config. They create no window.
Config-selected integrators remain available independently of this initial test coverage.

## Build and run

Use the same Python interpreter for CMake, dependencies, and scripts. For an existing build:

```powershell
python -m pip install -r tests/requirements.txt
cmake -S . -B build/release -DBUILD_TESTING=ON -DKRR_ENABLE_GPU_TESTS=ON
cmake --build build/release --config Release --parallel 2
ctest --test-dir build/release -C Release -L cpu --output-on-failure
ctest --test-dir build/release -C Release -L smoke --output-on-failure
ctest --test-dir build/release -C Release -L regression --output-on-failure
```

Keep the existing generator, CUDA/OptiX paths, and other build settings when configuring.
For a fresh build, also supply the usual project build options and `Python_EXECUTABLE`.
`BUILD_TESTING` defaults to `ON`; `KRR_ENABLE_GPU_TESTS` defaults to `OFF`. Explicitly enabled
GPU tests fail if the required device is unavailable. The normal project build still requires
the CUDA, OptiX, and Vulkan development dependencies even when only CPU tests will run.

CPU executables do not initialize a GPU. Sampler and option tests do not link the renderer.
`krr_publication` checks immediate first/final images, the frequency cap, adaptive backoff
and recovery, pending-submission bookkeeping, delayed completions, resets, and bounded
pending GPU work without requiring Blender, USD, or CUDA.
`krr_exr` generates small EXR fixtures and checks ZIP/PIZ/DWAA/DWAB decoding,
half/float channels, tiled edges, data-window offsets, row orientation, alpha,
HDR values, and failed-load cleanup. It links only the CPU image decoder.
On Windows with CUDA 12, native CPU tests delay-load `nvcuda.dll` and reject any attempted
driver call, so they also run on CI hosts without an NVIDIA driver.
With `KRR_ENABLE_OPENVDB_IO=OFF`, `krr_volume_import` links the renderer and verifies that
both direct volume files and heterogeneous-medium configs report the disabled import feature.
Python metric tests import
NumPy and the independent `image_metrics.py` module. Benchmark unit tests cover argument
validation, timing summaries, process failures, and synthetic profiler reports without a GPU
or installed profiler. Run the Python tests directly with:

```powershell
$env:PYTHONPATH = "$PWD/common/scripts"
python -m unittest discover -s tests/unit -p 'test_*.py' -v
```

The `gpu` label selects all GPU cases. CTest runs each case in a fresh process, gives it a
240-second execution timeout (270 seconds including the wrapper), and serializes GPU use
with a resource lock. A small CUDA readback test verifies RGB order and top-to-bottom rows.
The 32×32 smoke test checks repeatable 4-frame batches, fresh accumulation, final pass output,
array lifetime, stable tracked GPU memory, and sequential renderer creation. It also verifies
that invalid configs, missing models, and invalid frame counts fail cleanly and permit a
subsequent render, config dictionaries are snapshotted, exit requests cannot shorten a batch,
and configured saves occur only after successful completion. It runs in spectral and RGB builds.
Only spectral builds register the 128×128 reference regression.

Each enabled graphics API runs the same GPU cases. Vulkan retains the original
test names; D3D12 tests have a `_d3d12` suffix and separate artifact directories.
Use `-L vulkan` or `-L d3d12` to select an API. Readback exercises 64 cumulative
CUDA updates on persistent resources, then resizes and repeats. It also checks
descriptor ownership, slot reuse, binding-cache identity, texture pixel copies,
color-space cache entries, and failed-load retries. Both backends
compare against the same reviewed spectral reference; tests never regenerate it.

The `benchmark` label selects benchmark unit tests and an optional 32×32 GPU smoke test.
The latter compares warmed benchmark batches with ordinary rendering, checks timing values,
repeatability, independent arrays, capture callbacks, and cleanup after callback errors. It
uses the same timeout and GPU lock as the other smoke tests and requires no profiler tools.

CTest supplies the exact native module directory through `KRR_MODULE_DIR`, avoiding imports
from another build. It also supplies `KRR_BUILD_DIR` and the source script directory.
GitHub Actions runs the CPU tests for each existing CUDA/OptiX and spectral/RGB matrix entry;
GPU tests remain local.

## Python rendering

```python
from pathlib import Path
import krr

project_root = Path(krr.get_build_info()["project_root"])
config_path = project_root / "tests/cases/cornell_wavefront/config.json"
image = krr.render(config_path, frames=128, seed=17)
with krr.HeadlessRenderer(config_path, asset_root=project_root) as renderer:
    image = renderer.render(frames=128, seed=17)
```

For standalone scripts, put `common/scripts` on `PYTHONPATH` and set `KRR_BUILD_DIR` to the
chosen CMake build directory, or set `KRR_MODULE_DIR` directly to the directory containing
`pykrr` and `pykrr_common`. The API accepts a config dictionary or JSON path and an optional
asset root. Relative assets retain their project-root interpretation by default. Imports and
rendering leave the process working directory unchanged.

The returned array owns its data and is contiguous float32 RGB with shape `(height, width, 3)`
and top-to-bottom rows. It contains the final configured pass output. Each render call starts
a fresh scene and accumulation at time zero and executes the requested positive frame count.
Frames only equal samples per pixel when the configured integrator produces one sample per
frame, as in this test. The current API permits one active renderer per process. Use `close()`
or a context manager to release it before constructing another.

In headless batches, `save_on_finish` and nonzero `save_every` request a single image after
successful completion. Periodic `save_every`/`save_intermediate` requests are combined into
that final save; interactive periodic saving is unchanged. Closing a renderer or failing a
batch does not save an image.

## References and artifacts

`cases/cornell_wavefront/` keeps its config, spectral reference array, preview, and metadata
together. Geometry is reused from `common/assets/scenes/cbox/`. The config uses wavefront and
accumulation without denoising or tone mapping. Comparisons use untouched linear RGB values
in float64: MSE is the mean squared channel error, and normalized RMSE is
`sqrt(MSE / mean(reference ** 2))`. NaN/Inf and invalid dimensions always fail.
For an all-zero reference, normalized RMSE is zero for identical output and infinity otherwise.

The preview applies Reinhard and sRGB for viewing only. It never affects comparison metrics.
The metadata records the config/hash, revision, dirty working-tree status, renderer/toolchain,
GPU information exposed by the renderer, Python environment, seeds, and calibration results.

Normal tests cannot update references. Generate one deliberately using the selected spectral
build and review the preview and metadata before committing:

```powershell
$env:KRR_BUILD_DIR = "$PWD/build/release"
python tests/generate_reference.py --artifacts build/release/tests/artifacts/Release/reference_generation
# To replace an existing reference, append --force after deciding the change is expected.
```

Generation uses 16,384 reference frames and a separate reference seed. Calibration begins at
128 frames across five seeds; the threshold is 1.5 times their worst normalized RMSE, with a
minimum of `1e-6`. It must reject black output and a 50% exposure reduction. If it cannot,
calibration doubles the frame budget through 2,048 frames and fails rather than accepting an
uninformative threshold. The committed metadata sets the regression frame count and threshold.
This same-renderer baseline catches changes; it does not prove that the integrator is physically
correct. Review expected rendering changes before regenerating it.

Runtime results go beneath `<build>/tests/artifacts/<configuration>/<case>/`, including native
logs, process exit status/time, images, difference preview, metrics, and configs. Results are
saved on successful runs too, making calibration and local failures easier to inspect. Each
case directory represents the latest run; copy it elsewhere if a result should be retained.

## Extending the suite

`krr_material_program` tests the CPU expression compiler; `krr_openpbr` checks EON energy and
reciprocity, GGX sampling, F82 tint and LTC fuzz normalization without initializing CUDA.
`krr_materials` runs local CUDA checks for real image sampling, sRGB/channels/UV orientation,
simple and interpreted materials, opacity and classification slices, repeated uploads,
normal maps, and native BSDF energy/PDF agreement. Combined value/PDF evaluation is
checked against the separate functions for sampled and independent directions,
both transport modes, and independent base/coat normals. Native OpenPBR uses finite glossy lobes
even at zero roughness
(minimum roughness 0.001), with the reference's small IOR adjustment around unity; the legacy
materials keep their existing delta behavior. See the [implementation notes](../src/render/materials/openpbr/NOTICE).
OpenPBR emission uses 1000 nits per scene-linear radiance unit. Blender 5.2.2 exports
Principled strength directly; the adapter explicitly normalizes its constant and connected
strength inputs. SDK tests cover ordinary nits, the producer scale, invalid metadata, and an
actual Blender emission export with and without validated unit metadata. Preview Surface
and legacy material emission keep their existing units.

`krr_blender_preflight` runs without Blender or a GPU. Regenerate the checked-in producer
fixtures deliberately with Blender 5.2.2, then review their USD/MaterialX graphs and SDK test
results before replacing `tests/fixtures/usd/blender52/`:

```powershell
blender --background --factory-startup --python-exit-code 1 --python tests/blender/material_fixtures.py -- --output build/producer-fixtures
```

These fixtures cover supported math, UV/mapping, scalar/vector/color mixing, remapping,
channel conversions, base/coat normals and anisotropy, plus rejected reachable features.
Implicit color/vector-to-scalar links are rejected before export because Blender 5.2.2 can
reduce them to the first component; use explicit Separate Color/XYZ nodes instead.

The native `krr_session` GPU cases compare split frame steps with a fresh batch and cover
camera/resize resets, owned color/depth snapshots, cancellation, explicit finalization,
material emission changes, instance transforms in both hierarchy modes, scene replacement,
and sequential worker-thread cleanup.
`krr_camera` checks external projection rays on the CPU. The C++ `RenderSession` API keeps
scene resources between `step(frames)` calls; `snapshot(true)` additionally returns linear
and projected depth through an immutable shared snapshot. Repeated snapshots reuse depth
until a scene, camera, or resolution update invalidates it. All operations run on its
constructing thread except `requestCancel()`. `step()` submits GPU work; use `wait()`, a
snapshot, or a reset before inspecting device-backed scene data.
`requestSnapshot()` returns a pending image or no request when its two readback slots are
occupied. Use `isSnapshotReady()` and `collectSnapshot()` to obtain an owned image after its
graphics event completes; request metadata retains the original sample count and generation.
External camera projections preserve vertical framing when the output aspect ratio changes;
an explicit camera update replaces that conformed projection.
`finish()` performs configured saves once; snapshots, edits, cancellation, and `close()` do not.

Hydra publishes the first sample promptly and the final requested sample unconditionally.
Intermediate publications use elapsed time with a 75 ms minimum interval (13.3 Hz maximum).
The interval increases to ten times the smoothed submission/collection CPU cost, capped at
500 ms, and recovers as readback becomes cheaper. First-image startup and blocking waits
are excluded from this estimate. This is a pacing heuristic toward a roughly 10% publication
budget, not an isolated GPU transfer measurement.

Intermediate readbacks use two reusable NVRHI staging textures and event queries. The
existing render worker submits copies and continues stepping, then collects the newest
completed image when the publication interval allows. No additional CPU thread is used.
First and final images may wait for completion; scene changes discard stale requests, and
slot reuse waits for any outstanding graphics copy. Existing CUDA/graphics queue ordering
is preserved, so CPU-nonblocking readback does not imply transfer/kernel overlap on the GPU.
At most four samples remain pending before a GPU wait. Depth is captured only for bound
depth AOVs and cached across image publications. Hydra render buffers also reuse unchanged
color and depth data. This is an in-process CPU image transfer, not shared-memory IPC.
`krr_blender_render_scene` compares depth-enabled and color-only F12 output, verifies final
sample counts and depth-cache counters, and checks asynchronous intermediate publication on
a small 256-sample render. The windowed viewport probe checks the same counters across scene edits,
F12 handoff, reload, and cancellation. Its status JSON records publication and depth-capture
counts, pending readback storage, the adaptive interval, and readback, submission, waiting,
and total-update timings. Readback totals include explicit first/final collection waits;
time spent queued while rendering continues is not counted as CPU readback work.

With Hydra, SDK tests, and GPU tests enabled, `krr_blender_environment` loads the pinned
Blender installation's actual DWAB-compressed `forest.exr`. It checks finite, nonblack
scene-linear illumination and a black result after removing the world. Source metadata,
float32 EXRs, and logs are retained under `blender_environment`. Run the corresponding
Material Preview check in a separate windowed Blender process with:

```powershell
blender.exe --factory-startup --enable-event-simulate --window-geometry 0 0 480 480 `
  --python integrations/blender/probe/environment.py -- --viewport `
  --addon-dir build/blender-interop/host/blender/kiraray `
  --artifacts build/blender-interop/host/tests/artifacts/RelWithDebInfo/blender_environment_viewport
```

This selects Blender's `forest.exr` studio light with the scene world disabled, waits for
KiRaRay to converge, saves its status and a screenshot, then closes its own window.
Inspect `result.json`; `failure.txt` records any failed viewport assertion.
The background test removes the world for its dark control: Blender 5.2.2's Hydra bridge
does not forward Background Strength when the world uses an environment image.

Put CPU tests in `unit/`, GPU test runners in `render/`, and scene-specific inputs/references
in `cases/<name>/`. Register them in `tests/CMakeLists.txt` with appropriate labels and timeouts.
Keep rendering thresholds and assertions here. Reusable image calculations belong in
`common/scripts/image_metrics.py` and must remain independent of native renderer imports.

Benchmark scripts and their own configs live in the top-level `benchmarks/` directory, with
results beneath `<build>/benchmarks/<configuration>/<run>/`. See the [benchmark guide](../benchmarks/README.md)
for timing runs, optional profiler captures, and offline analysis. Benchmarks import `krr`
and reusable helpers rather than importing test runners.
