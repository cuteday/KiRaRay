"""KiRaRay's Python interface. Select a build with KRR_MODULE_DIR or KRR_BUILD_DIR."""

import copy
import importlib.machinery
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
_dll_handles = []


def _modules(directory, recursive=False):
    pattern = "**/pykrr*" if recursive else "pykrr*"
    return [path for path in directory.glob(pattern)
            if any(path.name == "pykrr" + suffix
                   for suffix in importlib.machinery.EXTENSION_SUFFIXES)]


def _module_directory():
    selected = os.environ.get("KRR_MODULE_DIR")
    if selected:
        directory = Path(selected).resolve()
        if not _modules(directory):
            raise ImportError(f"No pykrr module in KRR_MODULE_DIR={directory}")
        return directory
    build = os.environ.get("KRR_BUILD_DIR")
    if build:
        directory = Path(build).resolve()
        candidates = _modules(directory / "lib", recursive=True)
        candidates += _modules(directory / "bin", recursive=True)
    elif importlib.util.find_spec("pykrr") is not None:
        return Path(importlib.util.find_spec("pykrr").origin).parent
    else:
        candidates = _modules(ROOT_DIR / "build", recursive=True)
        candidates += _modules(ROOT_DIR / "out", recursive=True)
    if len(candidates) != 1:
        raise ImportError("Select a KiRaRay build with KRR_MODULE_DIR (the directory containing "
                          "pykrr.pyd) or KRR_BUILD_DIR (one CMake build directory).")
    return candidates[0].parent


_module_dir = _module_directory()
sys.path.insert(0, str(_module_dir))
import pykrr_common

if os.name == "nt":
    _directories = [_module_dir, _module_dir.parent / "bin",
                    _module_dir.parent.parent / "bin" / _module_dir.name,
                    Path(pykrr_common.vulkan_root) / "Bin",
                    ROOT_DIR / "src/ext/openvdb/openvdb/bin",
                    ROOT_DIR / "src/ext/openvdb/tbb/lib"]
    if pykrr_common.pytorch_root:
        _directories.append(Path(pykrr_common.pytorch_root) / "lib")
    _directories = [str(path) for path in _directories if path.is_dir()]
    _dll_handles.extend(os.add_dll_directory(path) for path in _directories)
    # Older Conda builds also use the legacy DLL search path.
    os.environ["PATH"] = os.pathsep.join(_directories + [os.environ.get("PATH", "")])

import pykrr


def _config(value):
    if isinstance(value, (str, os.PathLike)):
        with open(value, encoding="utf-8") as source:
            value = json.load(source)
    if not isinstance(value, dict):
        raise TypeError("config must be a dictionary or JSON path")
    return copy.deepcopy(value)


class HeadlessRenderer:
    """Render independent batches through the configured passes, without a window."""

    def __init__(self, config, *, asset_root=None, validation=True):
        if not isinstance(validation, bool):
            raise ValueError("validation must be a boolean")
        self._renderer = pykrr.HeadlessRenderer(
            _config(config), "" if asset_root is None else os.fspath(asset_root), validation)

    def render(self, *, frames, seed=0):
        """Return the final pipeline's RGB float32 image, with top-to-bottom rows."""
        try:
            if isinstance(frames, bool) or not isinstance(frames, int):
                raise ValueError("frames must be a positive integer")
            if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
                raise ValueError("seed must be an unsigned 64-bit integer")
            return self._renderer.render(frames, seed)
        except Exception:
            self.close()
            raise

    def benchmark(self, *, frames, warmup=8, seed=0, capture=False,
                  on_capture_begin=None, on_capture_end=None):
        """Time a fresh batch; warm-up samples remain in the returned image.

        render_ms covers the synchronized measured frame range, including CPU
        submissions and interop. Capture hooks run outside that timing range.
        """
        try:
            if isinstance(frames, bool) or not isinstance(frames, int) or not 0 < frames < 2**32:
                raise ValueError("frames must be a positive 32-bit integer")
            if (isinstance(warmup, bool) or not isinstance(warmup, int) or
                    warmup < 0 or warmup + frames >= 2**32):
                raise ValueError("warmup must be nonnegative and warmup + frames must fit in 32 bits")
            if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
                raise ValueError("seed must be an unsigned 64-bit integer")
            if not isinstance(capture, bool):
                raise ValueError("capture must be a boolean")
            if any(callback is not None and not callable(callback)
                   for callback in (on_capture_begin, on_capture_end)):
                raise ValueError("capture callbacks must be callable or None")
            return self._renderer.benchmark(
                frames=frames, warmup=warmup, seed=seed, capture=capture,
                on_capture_begin=on_capture_begin, on_capture_end=on_capture_end)
        except Exception:
            self.close()
            raise

    @property
    def closed(self):
        return self._renderer.closed

    def close(self):
        self._renderer.close()

    def __enter__(self):
        self._renderer.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def render(config, *, frames, seed=0, asset_root=None):
    """Render a fresh batch; config controls the integrator and output color representation."""
    with HeadlessRenderer(config, asset_root=asset_root) as renderer:
        return renderer.render(frames=frames, seed=seed)


def run(config):
    return pykrr.run(_config(config))


denoise = pykrr.denoise
get_build_info = pykrr.get_build_info
if hasattr(pykrr, "denoise_torch_tensor"):
    denoise_torch_tensor = pykrr.denoise_torch_tensor
