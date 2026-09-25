"""Explicitly generate and calibrate the Cornell spectral reference."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np

from render.support import (config_hash, load_config, metadata, preview, save_image,
                            validate_image, write_json)
from image_metrics import compare_images
import krr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path,
                        default=Path(__file__).resolve().parent / "cases" / "cornell_wavefront")
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--force", action="store_true", help="Replace an existing reviewed reference")
    args = parser.parse_args()
    if args.frames < 16384:
        parser.error("references require at least 16384 frames")
    paths = [args.case / f"reference_spectral.{extension}" for extension in ("npy", "png", "json")]
    if any(path.exists() for path in paths) and not args.force:
        parser.error("reference already exists; use --force only for a deliberate reference update")
    if not krr.get_build_info()["spectral"]:
        parser.error("reference generation requires a spectral build")
    seeds = [17, 29, 43, 71, 113]
    if args.seed in seeds:
        parser.error("reference seed must differ from the calibration seeds")
    args.artifacts.mkdir(parents=True, exist_ok=True)
    config = load_config(args.case / "config.json")
    write_json(args.artifacts / "config.json", config)
    start = time.perf_counter()
    with krr.HeadlessRenderer(config) as renderer:
        print(f"Rendering reference: {args.frames} frames, seed {args.seed}", flush=True)
        reference = renderer.render(frames=args.frames, seed=args.seed)
        validate_image(reference, config["resolution"])
        save_image(args.artifacts, "reference", reference)
        faults = {
            "black": compare_images(np.zeros_like(reference), reference),
            "half_exposure": compare_images(reference * 0.5, reference),
        }
        frames = 128
        history = []
        while frames <= 2048:
            measurements = []
            for seed in seeds:
                print(f"Calibrating: {frames} frames, seed {seed}", flush=True)
                image = renderer.render(frames=frames, seed=seed)
                validate_image(image, config["resolution"])
                metrics = compare_images(image, reference)
                measurements.append({"seed": seed, **metrics})
                save_image(args.artifacts, f"calibration_{frames}_{seed}", image)
            threshold = max(1e-6, 1.5 * max(item["normalized_rmse"] for item in measurements))
            history.append({"frames": frames, "measurements": measurements, "threshold": threshold})
            if all(value["normalized_rmse"] > threshold for value in faults.values()):
                break
            frames *= 2
        else:
            write_json(args.artifacts / "calibration.json", {"history": history, "faults": faults})
            raise RuntimeError("Noise still masks the fault checks at 2048 frames; inspect this case")
        environment = metadata(krr)

    record = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "config_sha256": config_hash(config),
        "reference_frames": args.frames,
        "reference_seed": args.seed,
        "regression_frames": frames,
        "regression_seed": seeds[0],
        "normalized_rmse_threshold": threshold,
        "calibration_margin": 1.5,
        "calibration": history,
        "fault_checks": faults,
        "elapsed_seconds": time.perf_counter() - start,
        **environment,
    }
    np.save(paths[0], reference, allow_pickle=False)
    preview(paths[1], reference)
    write_json(paths[2], record)
    write_json(args.artifacts / "calibration.json", record)
    print(f"Reference generated; {frames}-frame regression threshold: {threshold:.6g}")
    print("Review the reference preview and metadata before committing these files.")


if __name__ == "__main__":
    main()
