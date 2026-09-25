import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common" / "scripts"))


def load_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def config_hash(config):
    content = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def preview(path, image):
    # Previews use Reinhard and sRGB; comparison always uses the untouched arrays.
    mapped = np.maximum(np.asarray(image, dtype=np.float64), 0)
    mapped = mapped / (1 + mapped)
    mapped = np.where(mapped <= 0.0031308, 12.92 * mapped,
                      1.055 * np.power(mapped, 1 / 2.4) - 0.055)
    Image.fromarray(np.uint8(np.rint(np.clip(mapped, 0, 1) * 255))).save(path)


def save_image(directory, name, image):
    np.save(directory / f"{name}.npy", image, allow_pickle=False)
    if np.isfinite(image).all():
        preview(directory / f"{name}.png", image)


def validate_image(image, resolution):
    expected = (resolution[1], resolution[0], 3)
    if image.shape != expected or image.dtype != np.float32:
        raise AssertionError(f"Expected float32 {expected}, got {image.dtype} {image.shape}")
    if not image.flags.c_contiguous or not image.flags.owndata:
        raise AssertionError("Render output must own contiguous storage")
    if not np.isfinite(image).all():
        raise AssertionError("Render output contains NaN or infinity")
    if float(np.max(image)) <= 0:
        raise AssertionError("Render output is black")


def metadata(krr):
    def git(*args):
        result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True,
                                text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else "unavailable"

    return {
        "renderer": krr.get_build_info(),
        "revision": git("rev-parse", "HEAD"),
        "working_tree": git("status", "--short"),
        "python": sys.version,
        "platform": platform.platform(),
        "module_dir": os.environ.get("KRR_MODULE_DIR"),
        "build_dir": os.environ.get("KRR_BUILD_DIR"),
    }
