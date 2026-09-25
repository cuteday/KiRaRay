"""Image comparisons in linear RGB; no renderer or GPU dependencies."""

import numpy as np


def _images(image, reference):
    image = np.asarray(image)
    reference = np.asarray(reference)
    if image.shape != reference.shape:
        raise ValueError("Image and reference dimensions must match")
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) == 0:
        raise ValueError("Images must have nonempty shape (height, width, 3)")
    for value in (image, reference):
        if value.dtype.kind not in "fiu":
            raise ValueError("Images must contain real numeric values")
        if not np.isfinite(value).all():
            raise ValueError("Images must not contain NaN or infinity")
    return image.astype(np.float64), reference.astype(np.float64)


def compare_images(image, reference):
    """Return MSE and RMSE normalized by the reference's RMS intensity."""
    image, reference = _images(image, reference)
    error = float(np.mean(np.square(image - reference)))
    energy = float(np.mean(np.square(reference)))
    normalized = np.sqrt(error / energy) if energy else (0.0 if error == 0 else np.inf)
    return {"mse": error, "normalized_rmse": float(normalized)}


def mse(image, reference):
    return compare_images(image, reference)["mse"]


def normalized_rmse(image, reference):
    return compare_images(image, reference)["normalized_rmse"]
