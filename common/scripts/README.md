# KiRaRay Python Binding

Build the bindings with the Python interpreter that will run your scripts. Select it with
`-DPython_EXECUTABLE=<path-to-python>` when configuring CMake.

> After Python 3.8, the search paths of DLL dependencies has been reset. Only the system paths, the directory containing the DLL or PYD file are searched for load-time dependencies. Instead, a new function os.add_dll_directory() was added to supply additional search paths. 
Necessary DLL paths are exported from `pykrr_common`, see [krr.py](krr.py) for details.

### Start from script

Select the CMake build explicitly when using the Python interface:

```powershell
$env:PYTHONPATH = "$PWD/common/scripts"
$env:KRR_BUILD_DIR = "$PWD/build/release"
```

Alternatively, set `KRR_MODULE_DIR` to the directory containing the selected `pykrr*.pyd`
(normally the build's `lib` directory). This is required for ambiguous builds, including
multiple configurations. Importing `krr` does not change the working directory or initialize
the GPU.

### Render without a window

```python
from pathlib import Path
import krr

project_root = Path(krr.get_build_info()["project_root"])
config_path = project_root / "tests/cases/cornell_wavefront/config.json"
image = krr.render(config_path, frames=128, seed=0)

with krr.HeadlessRenderer(config_path, asset_root=project_root) as renderer:
    first = renderer.render(frames=128, seed=0)
    repeat = renderer.render(frames=128, seed=0)
```

Both entry points accept a config dictionary or JSON path, with optional `asset_root` for
relative assets. Assets default to the project root; config paths resolve from the caller's
working directory. The config chooses the integrator and pass order.

The result is an owned, contiguous `float32[height, width, 3]` RGB array, with top-to-bottom
rows. It contains the final pass output. For linear HDR, use wavefront and accumulation
without tone mapping or denoising. Frames count pipeline executions, not necessarily SPP.
Each call uses a fresh scene/pass batch at time zero; it does not resume accumulation.

One renderer can be active at a time. Use the context manager or `close()` to release it.
Configured saves run after successful rendering; closing does not save. In headless batches,
`save_on_finish` or periodic `save_every`/`save_intermediate` requests produce one final image,
without intermediate files. Interactive periodic saving is unchanged. Ordinary errors raise
exceptions and close the failed renderer. The shared CUDA/OptiX context and spectral
tables remain available for later rendering or denoising calls.

See [tests](../../tests/README.md) for CTest and reference-generation commands. Python calls
are synchronous; concurrent rendering and zero-copy output are not supported.

`HeadlessRenderer.benchmark(frames=64, warmup=8, seed=0)` returns an image and synchronized
batch timings. Warm-up samples remain in the image, while setup and readback are excluded
from measured rendering time. See [benchmarks](../../benchmarks/README.md) for repeated runs,
NVTX capture ranges, and the optional Nsight profiler backends.

The [headless CLI example](render_headless.py) reads a config and writes its final image to
the chosen directory. Run it from the project root with the build selected above:

```powershell
python -m pip install numpy Pillow
python common/scripts/render_headless.py --config common/configs/example_cbox.json --output-dir build/release/renders/cornell --frames 128 --seed 0
```

It saves `render.npy` with the unchanged float32 output and `render.png` clipped to `[0, 1]`
and converted to 8-bit RGB. It applies no additional tone mapping or gamma; use a config
with `ToneMappingPass`, such as `example_cbox.json`, for a display-ready PNG. The array retains
HDR values when the pipeline produces them. Reusing an output directory overwrites these files.
Any pass-configured saves also use this directory. `--asset-root` optionally overrides relative
asset resolution; `--help` lists all arguments without loading the native renderer.

### Interactive rendering

Start the interactive renderer using a config dictionary or JSON path:

~~~Python
from pathlib import Path
import krr

project_root = Path(krr.get_build_info()["project_root"])
krr.run(project_root / "common/configs/example_cbox.json")
~~~

### Denoising Images

Kiraray implements a python wrapper for denoising images with optix's built-in ai denoiser. See [denoise.py](./examples/denoise.py) for an example. To denoise an image, the hdr noisy image is provided as arguments, with optionally the normals and albedo (in linear space). All arguments are numpy arrays with the same shape.

~~~Python
img_denoised = krr.denoise(img_noisy, img_normals, img_albedo)
~~~

This makes it easy to denoise many image files with python scripts. On my RTX3070 Laptop, denoising an image with 1920x1080 takes approximately 1s, while most of the overhead is the memory copy between host and device. It takes about 25ms when acting as a render pass (see [denoise.cpp](../../src/render/passes/denoise/denoise.cpp)).

#### Denoising PyTorch Tensor
To enable support for PyTorch, you should define the `TORCH_INSTALL_DIR` environment variable to point to the PyTorch installation directory (see [here](../build/FindPyTorch.cmake) for details). The tensor should be on GPU for no CPU-GPU memory copy.  

~~~Python
img_denoised = krr.denoise_torch_tensor(img_noisy, img_normals, img_albedo)
~~~
