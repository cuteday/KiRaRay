"""Render a JSON config without a window and save the final pipeline image."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to the scene JSON config")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for rendered images")
    parser.add_argument("--frames", type=int, default=128, help="Number of frames to render (default: 128)")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed (default: 0)")
    parser.add_argument("--asset-root", type=Path, help="Root for relative assets (default: project root)")
    args = parser.parse_args()
    if not 0 < args.frames < 2**32:
        parser.error("--frames must be a positive 32-bit integer")
    if not 0 <= args.seed < 2**64:
        parser.error("--seed must be an unsigned 64-bit integer")

    import numpy as np
    from PIL import Image
    import krr

    with args.config.open(encoding="utf-8") as source:
        config = json.load(source)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Route any saves requested by the passes to the same output directory.
    config["output"] = str(output_dir)

    with krr.HeadlessRenderer(config, asset_root=args.asset_root) as renderer:
        image = renderer.render(frames=args.frames, seed=args.seed)

    array_path = output_dir / "render.npy"
    np.save(array_path, image, allow_pickle=False)
    print(f"Saved {array_path}")
    if not np.isfinite(image).all():
        raise ValueError("Render output contains NaN or infinity; the raw array was saved")

    # PNG expects display-ready output; the config controls tone mapping and gamma.
    pixels = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
    image_path = output_dir / "render.png"
    Image.fromarray(pixels).save(image_path)
    print(f"Saved {image_path}")


if __name__ == "__main__":
    main()
