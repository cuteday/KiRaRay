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

CPU executables do not initialize a GPU or link the renderer. Python metric tests import
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
CUDA updates on persistent resources, then resizes and repeats. Both backends
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

Put CPU tests in `unit/`, GPU test runners in `render/`, and scene-specific inputs/references
in `cases/<name>/`. Register them in `tests/CMakeLists.txt` with appropriate labels and timeouts.
Keep rendering thresholds and assertions here. Reusable image calculations belong in
`common/scripts/image_metrics.py` and must remain independent of native renderer imports.

Benchmark scripts and their own configs live in the top-level `benchmarks/` directory, with
results beneath `<build>/benchmarks/<configuration>/<run>/`. See the [benchmark guide](../benchmarks/README.md)
for timing runs, optional profiler captures, and offline analysis. Benchmarks import `krr`
and reusable helpers rather than importing test runners.
