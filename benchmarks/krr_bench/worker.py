"""One isolated benchmark or profiler target process."""

import argparse
import json
from pathlib import Path
import platform
import sys
import traceback

from .runner import validate_options, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    request = json.loads(args.request.read_text(encoding="utf-8"))
    directory = args.output_dir
    try:
        validate_options(request["frames"], request["warmup"], request["repeats"], request["seed"])
        import numpy as np
        runs = []
        profile_info = None
        image = None

        def record(result):
            nonlocal image, profile_info
            image = result.pop("image")
            if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.float32 or not np.isfinite(image).all():
                raise RuntimeError("Benchmark produced an invalid image")
            if "profile" in result:
                profile_info = result.pop("profile")
            runs.append(result)

        if request["backend"] == "nsight-python":
            # Import the adapter before the renderer so profiler injection precedes CUDA.
            from .nsight_adapter import profile
            result = profile(request["config"], frames=request["frames"], warmup=request["warmup"],
                seed=request["seed"], asset_root=request["asset_root"],
                validation=request["validation"], output_dir=directory,
                tool_path=request.get("tool_path"), allow_debug=request["allow_debug"],
                configuration=request["configuration"])
            import krr
            record(result)
        else:
            import krr
            info = krr.get_build_info()
            if info["build_type"].lower() not in ("release", "relwithdebinfo") and not request["allow_debug"]:
                raise RuntimeError("Selected native module is not an optimized build; use --allow-debug only for development")
            if info["build_type"] != request["configuration"]:
                raise RuntimeError("Native module build type differs from the selected configuration")
            with krr.HeadlessRenderer(request["config"], asset_root=request["asset_root"],
                                      validation=request["validation"]) as renderer:
                for index in range(request["repeats"]):
                    print(f"Starting batch {index + 1}/{request['repeats']}", flush=True)
                    record(renderer.benchmark(frames=request["frames"], warmup=request["warmup"],
                        seed=request["seed"], capture=request["backend"] in ("ncu", "nsys")))
        np.save(directory / "render.npy", image, allow_pickle=False)
        output = {"schema_version": 1, "status": "passed", "backend": request["backend"],
                  "profiled": request["backend"] != "timing", "runs": runs,
                  "build": krr.get_build_info(), "python": {"executable": sys.executable,
                  "version": platform.python_version()}, "revision": request["revision"],
                  "validation": request["validation"], "seed": request["seed"],
                  "warmup_samples_in_image": True}
        if profile_info is not None:
            output["profile"] = profile_info
        write_json(directory / "result.json", output)
        return 0
    except Exception as error:
        write_json(directory / "worker_failure.json", {"status": "failed", "error": str(error)})
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
