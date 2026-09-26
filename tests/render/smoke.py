"""Exercise the public Python API with a small wavefront render."""

import argparse
import copy
import gc
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from support import ROOT, load_config, metadata, save_image, validate_image, write_json

IMPORT_CWD = Path.cwd()
import krr


def expect_error(operation):
    try:
        operation()
    except (ValueError, RuntimeError, OSError):
        return
    raise AssertionError("Invalid rendering request was accepted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--graphics-api", choices=("vulkan", "d3d12"), default="vulkan")
    args = parser.parse_args()
    if Path.cwd() != IMPORT_CWD:
        raise AssertionError("Importing krr changed the process working directory")
    args.artifacts.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    config["graphics_api"] = args.graphics_api
    config["resolution"] = [32, 32]
    write_json(args.artifacts / "config.json", config)
    cwd = Path.cwd()
    memory = []

    def check_memory(stage):
        current = krr.get_build_info()["tracked_bytes"]
        memory.append({"stage": stage, "tracked_bytes": current})
        write_json(args.artifacts / "memory.json", memory)
        if current != memory[0]["tracked_bytes"]:
            raise AssertionError(f"Tracked GPU memory changed after {stage}: "
                                 f"{memory[0]['tracked_bytes']} -> {current} bytes")

    def check_recovery(stage):
        image = krr.render(config, frames=4, seed=17)
        np.testing.assert_allclose(image, saved, rtol=1e-6, atol=1e-7)
        check_memory(stage)

    snapshot = copy.deepcopy(config)
    with krr.HeadlessRenderer(snapshot) as renderer:
        snapshot["resolution"] = [16, 16]
        snapshot["passes"][0]["params"]["max_depth"] = 1
        first = renderer.render(frames=4, seed=17)
        check_memory("first batch")
        validate_image(first, [32, 32])
        saved = first.copy()
        save_image(args.artifacts, "render", first)
        other = renderer.render(frames=4, seed=29)
        check_memory("different seed")
        repeat = renderer.render(frames=4, seed=17)
        check_memory("repeated seed")
        validate_image(other, [32, 32])
        np.testing.assert_array_equal(first, saved)
        np.testing.assert_allclose(repeat, saved, rtol=1e-6, atol=1e-7)
        if np.array_equal(other, first):
            raise AssertionError("Changing the render seed did not change the samples")
        if np.shares_memory(first, repeat):
            raise AssertionError("Returned images share storage")
        for _ in range(2):
            try:
                second = krr.HeadlessRenderer(config)
            except RuntimeError:
                pass
            else:
                second.close()
                raise AssertionError("A second active renderer was accepted")
        write_json(args.artifacts / "metadata.json", metadata(krr))
    renderer.close()
    try:
        renderer.render(frames=4, seed=17)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Rendering after close was accepted")
    del renderer
    gc.collect()
    np.testing.assert_array_equal(first, saved)

    # Exercise JSON loading and sequential device creation with the same batch.
    recreated = krr.render(str(args.artifacts / "config.json"), frames=4, seed=17,
                           asset_root=str(ROOT))
    np.testing.assert_allclose(recreated, saved, rtol=1e-6, atol=1e-7)
    check_memory("sequential recreation")

    budget_config = copy.deepcopy(config)
    budget_config["passes"][1]["params"].update({
        "task": {"type": "spp", "value": 1}, "exit_on_finish": True})
    budget_image = krr.render(budget_config, frames=4, seed=17)
    np.testing.assert_allclose(budget_image, saved, rtol=1e-6, atol=1e-7)
    check_memory("configured exit request")

    invalid = copy.deepcopy(config)
    invalid["passes"][0]["name"] = "UnknownTestPass"
    expect_error(lambda: krr.render(invalid, frames=4, seed=17))
    check_recovery("unknown pass recovery")

    expect_error(lambda: krr.HeadlessRenderer(config, asset_root=str(args.config.resolve())))
    check_recovery("invalid asset root recovery")

    missing_model = args.artifacts.resolve() / "missing-scene.obj"
    if missing_model.exists():
        raise AssertionError(f"Missing-file fixture unexpectedly exists: {missing_model}")
    invalid = copy.deepcopy(config)
    invalid.pop("scene")
    invalid["model"] = str(missing_model)
    expect_error(lambda: krr.render(invalid, frames=4, seed=17))
    check_recovery("missing model recovery")
    invalid = copy.deepcopy(config)
    invalid["scene"]["model"][0]["model"] = str(missing_model)
    expect_error(lambda: krr.render(invalid, frames=4, seed=17))
    check_recovery("missing scene model recovery")

    renderer = krr.HeadlessRenderer(config)
    expect_error(lambda: renderer.render(frames=0, seed=17))
    if not renderer.closed:
        raise AssertionError("Invalid render arguments did not close the renderer")
    renderer.close()
    check_recovery("invalid frame count recovery")

    for periodic in (False, True):
        with tempfile.TemporaryDirectory(prefix="save_", dir=args.artifacts) as directory:
            output_dir = Path(directory).resolve()
            save_config = copy.deepcopy(config)
            save_config["output"] = str(output_dir)
            save_config["passes"][1]["params"].update({
                "save_on_finish": not periodic, "save_every": 1 if periodic else 0})
            expected = output_dir / "result.exr"
            renderer = krr.HeadlessRenderer(save_config)
            try:
                image = renderer.render(frames=4, seed=17)
                np.testing.assert_allclose(image, saved, rtol=1e-6, atol=1e-7)
                if list(output_dir.glob("*.exr")) != [expected] or expected.stat().st_size == 0:
                    raise AssertionError("Configured save did not produce only the completed image")
                os.utime(expected, ns=(1_000_000_000, 1_000_000_000))
                saved_time = expected.stat().st_mtime_ns
                renderer.close()
                if expected.stat().st_mtime_ns != saved_time:
                    raise AssertionError("Closing the renderer rewrote the saved image")
                name = "saved_periodic.exr" if periodic else "saved_on_finish.exr"
                shutil.copyfile(expected, args.artifacts / name)
            finally:
                renderer.close()

            save_config["output"] = str(output_dir / "failed")
            renderer = krr.HeadlessRenderer(save_config)
            expect_error(lambda: renderer.render(frames=0, seed=17))
            renderer.close()
            if list((output_dir / "failed").glob("*.exr")):
                raise AssertionError("An unsuccessful batch saved an image")
            check_memory("periodic save" if periodic else "save on finish")

    final_config = copy.deepcopy(config)
    final_config["passes"].append({"name": "ToneMappingPass", "params": {
        "exposure": 0.5, "operator": "aces"}})
    processed = krr.render(final_config, frames=4, seed=17)
    validate_image(processed, [32, 32])
    if np.allclose(processed, saved):
        raise AssertionError("Returned image did not include the final configured pass")
    check_memory("final pass pipeline")
    if Path.cwd() != cwd:
        raise AssertionError("Rendering changed the process working directory")
    write_json(args.artifacts / "result.json", {"passed": True, "frames": 4, "seeds": [17, 29]})
    print("Wavefront smoke passed: repeated batches, cleanup, error recovery, and final output")


if __name__ == "__main__":
    main()
