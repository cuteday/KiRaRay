"""Check warmed headless batches without requiring an external profiler."""

import argparse
import math
from pathlib import Path

import numpy as np

from support import load_config, metadata, save_image, validate_image, write_json

import krr


def check_result(result, frames, warmup):
    if result["frames"] != frames or result["warmup_frames"] != warmup:
        raise AssertionError("Benchmark frame counts do not match the request")
    validate_image(result["image"], [32, 32])
    timings = result["timings"]
    for name in ("setup_ms", "warmup_ms", "render_ms", "readback_ms", "finalize_ms", "total_ms"):
        if not math.isfinite(timings[name]) or timings[name] < 0:
            raise AssertionError(f"Invalid benchmark timing: {name}={timings[name]}")
    if timings["render_ms"] <= 0 or timings["total_ms"] < timings["render_ms"]:
        raise AssertionError("Benchmark did not time a completed frame range")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    config["resolution"] = [32, 32]
    write_json(args.artifacts / "config.json", config)
    cwd = Path.cwd()
    frames, warmup, seed = 4, 2, 17
    expected = krr.render(config, frames=frames + warmup, seed=seed)
    baseline_bytes = krr.get_build_info()["tracked_bytes"]
    records = []

    def check_memory(stage):
        current = krr.get_build_info()["tracked_bytes"]
        if current != baseline_bytes:
            raise AssertionError(f"Tracked GPU memory changed after {stage}: "
                                 f"{baseline_bytes} -> {current}")

    def check_recovery(stage):
        image = krr.render(config, frames=frames + warmup, seed=seed)
        np.testing.assert_allclose(image, expected, rtol=1e-6, atol=1e-7)
        check_memory(stage)

    callbacks = []
    with krr.HeadlessRenderer(config, validation=False) as renderer:
        first = renderer.benchmark(frames=frames, warmup=warmup, seed=seed,
                                   on_capture_begin=lambda: callbacks.append("begin"),
                                   on_capture_end=lambda: callbacks.append("end"))
        check_result(first, frames, warmup)
        if callbacks != ["begin", "end"]:
            raise AssertionError(f"Unexpected capture callbacks: {callbacks}")
        np.testing.assert_allclose(first["image"], expected, rtol=1e-6, atol=1e-7)
        save_image(args.artifacts, "render", first["image"])
        records.append(first["timings"])
        check_memory("first batch")

        repeat = renderer.benchmark(frames=frames, warmup=warmup, seed=seed)
        check_result(repeat, frames, warmup)
        np.testing.assert_allclose(repeat["image"], expected, rtol=1e-6, atol=1e-7)
        if np.shares_memory(first["image"], repeat["image"]):
            raise AssertionError("Benchmark images share storage")
        repeat["image"].fill(0)
        np.testing.assert_allclose(first["image"], expected, rtol=1e-6, atol=1e-7)
        records.append(repeat["timings"])
        check_memory("repeated batch")

        no_warmup = renderer.benchmark(frames=frames + warmup, warmup=0, seed=seed)
        check_result(no_warmup, frames + warmup, 0)
        np.testing.assert_allclose(no_warmup["image"], expected, rtol=1e-6, atol=1e-7)
        check_memory("zero warmup")
    renderer.close()

    for arguments in ({"frames": 0}, {"frames": True}, {"frames": 1, "warmup": -1},
                      {"frames": 1, "warmup": True}, {"frames": 2**32 - 1, "warmup": 1},
                      {"frames": 1, "seed": -1}):
        renderer = krr.HeadlessRenderer(config, validation=False)
        try:
            renderer.benchmark(**arguments)
        except (ValueError, RuntimeError, OverflowError):
            if not renderer.closed:
                raise AssertionError("Invalid benchmark arguments left the renderer open")
        else:
            raise AssertionError(f"Invalid benchmark arguments were accepted: {arguments}")
        finally:
            renderer.close()
    check_recovery("invalid argument recovery")

    for hook in ("on_capture_begin", "on_capture_end"):
        callbacks = []

        def begin():
            callbacks.append("begin")
            if hook == "on_capture_begin":
                raise RuntimeError("benchmark callback failure")

        def end():
            callbacks.append("end")
            if hook == "on_capture_end":
                raise RuntimeError("benchmark callback failure")

        renderer = krr.HeadlessRenderer(config, validation=False)
        try:
            renderer.benchmark(frames=frames, warmup=warmup, seed=seed,
                               on_capture_begin=begin, on_capture_end=end)
        except RuntimeError as error:
            if str(error) != "benchmark callback failure":
                raise
            if not renderer.closed:
                raise AssertionError("Failed capture callback left the renderer open")
        else:
            raise AssertionError(f"A failed {hook} callback was ignored")
        finally:
            renderer.close()
        expected_callbacks = ["begin"] if hook == "on_capture_begin" else ["begin", "end"]
        if callbacks != expected_callbacks:
            raise AssertionError(f"Unexpected callbacks after {hook} failure: {callbacks}")
        check_recovery(f"{hook} recovery")

    if Path.cwd() != cwd:
        raise AssertionError("Benchmarking changed the process working directory")
    write_json(args.artifacts / "metadata.json", metadata(krr))
    write_json(args.artifacts / "result.json", {"passed": True, "frames": frames,
               "warmup_frames": warmup, "seed": seed, "timings": records})
    print("Benchmark smoke passed: warmed batches, timings, callbacks, and cleanup")


if __name__ == "__main__":
    main()
