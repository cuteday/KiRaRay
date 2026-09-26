# Headless benchmarks

Use `run.py` for ordinary timings, optional profiler captures, and offline report analysis.
The runner starts a separate Python worker selected from the chosen CMake build. Importing
the orchestration and analysis modules does not import KiRaRay or initialize a GPU.

Benchmark configs live in `cases/` and reuse project assets. The initial Cornell case uses
512×512 wavefront rendering followed by accumulation, without tone mapping or denoising.
Future cases can select other integrators and passes through their own configs. Benchmarks
do not import test runners or alter image regression references.

## Build and run

Build an optimized `Release` or `RelWithDebInfo` configuration using the usual CUDA, OptiX,
Vulkan, and Python options. Use the same Python interpreter for dependencies and the native
module. The build directory's name does not determine its configuration: a directory named
`build/release` can still contain a Debug build.

```powershell
python -m pip install -r benchmarks/requirements.txt
python benchmarks/run.py run --build-dir build/release --frames 64 --warmup 8 --repeats 3
```

For multi-configuration generators, also select `--configuration Release`. The runner reads
the interpreter from `CMakeCache.txt`; `--python` explicitly overrides it and must match the
native module's Python ABI. `--allow-debug` permits development checks, but these results
should not be used to compare optimized renderer performance. Vulkan/NVRHI validation is
disabled by default for benchmark workers; add `--validation` when investigating correctness.

```powershell
python benchmarks/run.py run --build-dir build/release --config benchmarks/cases/cornell_wavefront.json --resolution 1024 1024 --frames 128 --warmup 16 --repeats 5 --seed 17
```

Each repetition starts a fresh scene and pass pipeline. Within that batch, warm-up frames
run before the measured frames, without recreating the scene between them. Warm-up samples
remain in accumulation: the saved image contains `warmup + frames` frames. Scene time is
fixed at zero and each repetition uses the requested seed. The default case produces one
sample per pixel per frame; other pass configurations may behave differently.

`render_ms` measures elapsed wall time around the measured native frame loop and waits for
its CUDA/Vulkan work to complete. It includes frame submission and synchronization, making
it an end-to-end rendering throughput measurement. It is not the sum of individual kernel
durations. Setup, warm-up, readback, and finalization have separate timing fields. Capture
callbacks are excluded from `render_ms`; `total_ms` includes them. The old interactive
timing profiler is temporarily disabled during a benchmark batch.

The summary contains minimum, median, mean, maximum, and sample standard deviation for
batch duration, milliseconds per measured frame, and frames per second. A single repetition
has zero reported standard deviation. Keep config, resolution, seed, build variant, and GPU
comparable when comparing results. Use an otherwise idle GPU and repeat ordinary runs to
judge variability. Timings produced while a profiler is attached are marked as profiled and
are not turned into the ordinary performance summary.

## Capture and analyze

Install the desired NVIDIA profiler separately. Tools are discovered from `PATH` and common
installation directories; use `--tool-path` or `KRR_NCU` / `KRR_NSYS` to select an executable.
The selected executable and its version are printed before each CLI capture.
Captures use one batch and default to one measured frame after eight warm-up frames.

```powershell
python benchmarks/run.py profile --build-dir build/release --tool nsys --frames 4 --warmup 8
python benchmarks/run.py profile --build-dir build/release --tool ncu --frames 1 --warmup 8
python benchmarks/run.py profile --build-dir build/release --tool ncu --kernel-regex generateCameraRays --metrics gpu__time_duration.sum
```

NCU uses kernel replay, preserves `.ncu-rep`, and collects the basic metric set unless
`--metrics` selects explicit metrics. Start with a small frame count or kernel filter because
a wavefront frame launches many kernels. Range replay is unsuitable for this renderer's
CUDA/Vulkan external-resource interop. OptiX profiling exposes user kernels with some NVIDIA
internal implementation hidden. For OptiX source information in an optimized build, configure
with `-DKRR_PROFILE_OPTIX=ON` and rebuild. This preserves the default OptiX optimization level
and adds debug information; NVCC already includes line information in optimized builds.

Nsight Systems captures CUDA, Vulkan, and NVTX activity. Both CLI captures use native CUDA
profiler start/stop calls around the warmed measured interval, excluding setup and readback.
NVTX frame/pass ranges provide context for navigating the reports. Profilers can alter
execution timing through replay, serialization, and instrumentation; use ordinary `run`
results for performance comparisons.

On Windows, Nsight Systems may need administrator rights to register its Vulkan tracing
layer. If it reports a Vulkan JSON registration permission error, run the full capture
from an Administrator terminal and then check whether subsequent runs work without
elevation. For CUDA/OptiX and NVTX tracing without Vulkan API events, explicitly select:

```powershell
python benchmarks/run.py profile --build-dir build/release --tool nsys --nsys-trace cuda,nvtx --frames 4 --warmup 8
```

The chosen trace set is recorded in the artifacts. The default remains `cuda,vulkan,nvtx`;
the runner never silently drops Vulkan tracing. CPU sampling and context-switch tracing
are disabled for these GPU captures.

Captures automatically run offline analysis and print the readable summary to the terminal,
while preserving `analysis.json` and `summary.md`. You can also analyze an existing report
without rendering or requiring a GPU; this prints and saves the same summary:

```powershell
python benchmarks/run.py analyze path/to/profile.ncu-rep --output-dir build/release/benchmarks/ncu-analysis
python benchmarks/run.py analyze path/to/profile.nsys-rep --output-dir build/release/benchmarks/nsys-analysis
python benchmarks/run.py analyze path/to/profile.sqlite --output-dir build/release/benchmarks/sqlite-analysis
```

NCU analysis loads the official `ncu_report` module shipped with Nsight Compute and retains
per-kernel metrics, units, available rule findings, and NVTX context. Nsight Systems exports
SQLite for Python analysis and retains tool statistics. Its summary includes kernel/API
durations, transfer totals, and the union of CUDA activity intervals. Timeline gaps are
uncovered intervals in that exported activity, not proof that the entire GPU was idle.
Native reports remain the source for deeper investigation in NVIDIA's GUI tools.

An explicitly requested tool, capture, or analysis failure returns a nonzero exit status.
Partial reports and logs are retained for diagnosis and are not accepted as successful runs.
`--timeout` bounds execution (900 seconds by default).

## Optional Nsight Python backend

The optional backend uses [Nsight Python](https://docs.nvidia.com/nsight-python/) for in-process
NCU collection. Ordinary benchmarks and the CLI backends do not require this package.

Nsight Python requires Python 3.10 or newer and Nsight Compute 2026.2.1 or newer. Create a
matching Python environment, install the extra requirements, and configure a separate KiRaRay
build with that environment's `Python_EXECUTABLE`. An existing CPython 3.9 `.pyd` cannot be
loaded by a Python 3.10+ interpreter.

This adapter is experimental. Published Nsight Python 1.0.0 cannot load the Windows injection
library; the optional requirements pin NVIDIA's [Windows compatibility fix](https://github.com/NVIDIA/nsight-python/commit/e11f4db8f0dea0219c1f367a370a10fd37c851d9)
at commit `e11f4db8f0dea0219c1f367a370a10fd37c851d9` on all platforms. Installation requires
Git. Captures record both the installed package version and its source revision.
The selected NCU injection library must export `nvInjBeginProfiling` and `nvInjEndProfiling`.
Some internal NCU builds lack these symbols despite a recent version number. The adapter
checks this before importing KiRaRay; use the `ncu` CLI backend or select a compatible public
NCU build with `--tool-path` if initialization fails.

```powershell
python -m pip install -r benchmarks/requirements-nsight.txt
python benchmarks/run.py profile --build-dir build/nsight --tool nsight-python --frames 1 --warmup 8
```

The worker imports Nsight Python before KiRaRay so its CUDA injection is installed before
renderer initialization. Two native capture callbacks surround only the measured frame
window. This preserves the same fresh-batch and warm-up behavior used by CLI profiling.
The initial adapter collects duration metrics for the launches in the annotation, retains a
native NCU report, and reuses per-kernel offline analysis. Its aggregate Python results are
not treated as a complete frame timing model. Use the `ncu` backend for custom metrics or
kernel filters, and `nsys` for the combined Vulkan/CUDA timeline.

## Artifacts and Python API

Each invocation creates a unique directory at
`<build>/benchmarks/<configuration>/<timestamp>-<id>/`. `--output-dir` selects an empty output
directory explicitly. Artifacts include the effective `config.json`, `request.json`, worker
command/logs, `result.json`, the final owned RGB `render.npy`, and `summary.json` for ordinary
timing runs. Metadata records the revision and dirty flag, seed, Python interpreter, build
variant, CUDA/OptiX versions, GPU/driver, and validation setting. Captures also retain profiler
commands/version/logs, the native report, and analysis outputs. Failures retain their error
records and any partial artifacts.

The native benchmark API is also available directly:

```python
import krr

with krr.HeadlessRenderer("benchmarks/cases/cornell_wavefront.json", validation=False) as renderer:
    result = renderer.benchmark(frames=64, warmup=8, seed=0)
image = result["image"]
timings = result["timings"]
```

For direct use, select the build with `KRR_BUILD_DIR` or `KRR_MODULE_DIR` and put
`common/scripts` on `PYTHONPATH`. Images follow the ordinary `render()` API's format and pass
ordering. `frames` must be positive, `warmup` nonnegative, and their sum fit in an unsigned
32-bit frame index. Seeds are unsigned 64-bit integers. Optional `capture=True` enables native
CUDA profiler start/stop, while `on_capture_begin` and `on_capture_end` support Python capture
contexts. Once begin returns successfully, end runs once even if rendering fails. Failed
batches close the renderer and propagate the error; `close()` remains idempotent.

## Tests and extension

CPU unit tests cover option validation, timing summaries, process failures, and report parsing
with synthetic fixtures. They need no GPU, installed profiler, pandas, or Nsight Python.
The optional 32×32 GPU benchmark smoke checks warmed-frame output against ordinary rendering,
repeatability, independent arrays, capture callbacks, failure cleanup, and timing validity.

```powershell
ctest --test-dir build/release -C Release -L benchmark --output-on-failure
```

GPU tests require `KRR_ENABLE_GPU_TESTS=ON` and share the existing CTest GPU resource lock.
Keep reusable orchestration and report readers in `krr_bench/`, benchmark cases in `cases/`,
and correctness assertions in `tests/`. Add new backends behind the same capture/analysis
interface without making their Python packages mandatory for ordinary runs.
