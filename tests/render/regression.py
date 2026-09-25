"""Compare the linear Cornell render with its reviewed spectral reference."""

import argparse
from pathlib import Path

import numpy as np

from support import config_hash, load_config, metadata, save_image, validate_image, write_json
from image_metrics import compare_images
import krr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    config = load_config(args.case / "config.json")
    baseline = load_config(args.case / "reference_spectral.json")
    if not krr.get_build_info()["spectral"]:
        raise RuntimeError("This reference requires a spectral build")
    if config_hash(config) != baseline["config_sha256"]:
        raise RuntimeError("Test config differs from the reference; review and regenerate it explicitly")
    reference = np.load(args.case / "reference_spectral.npy", allow_pickle=False)
    validate_image(reference, config["resolution"])
    write_json(args.artifacts / "config.json", config)
    write_json(args.artifacts / "reference_metadata.json", baseline)
    image = krr.render(config, frames=baseline["regression_frames"], seed=baseline["regression_seed"])
    save_image(args.artifacts, "render", image)
    save_image(args.artifacts, "reference", reference)
    save_image(args.artifacts, "difference", np.abs(image - reference))
    write_json(args.artifacts / "metadata.json", metadata(krr))
    validate_image(image, config["resolution"])
    metrics = compare_images(image, reference)
    threshold = baseline["normalized_rmse_threshold"]
    passed = metrics["normalized_rmse"] <= threshold
    write_json(args.artifacts / "metrics.json", {**metrics, "threshold": threshold, "passed": passed})
    print(f"MSE: {metrics['mse']:.8g}; normalized RMSE: {metrics['normalized_rmse']:.6g}; "
          f"threshold: {threshold:.6g}")
    if not passed:
        raise AssertionError("Cornell rendering exceeded the reference error threshold")


if __name__ == "__main__":
    main()
