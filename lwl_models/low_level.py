"""Itti-inspired intensity/color/orientation center-surround saliency.

This transparent approximation is not the original reference implementation:
we use two signed opponent channels and quadrature Gabor energy, a Gaussian
pyramid, and local-maximum competition. No learned weights or gaze fitting.
"""

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, convolve, maximum_filter


def resize_map(array, shape):
    height, width = shape
    image = Image.fromarray(np.asarray(array, dtype=np.float32))
    return np.asarray(image.resize((width, height), Image.Resampling.BILINEAR), dtype=float)


def competition_normalize(array):
    """Normalize to [0,1], then favor maps with few strong local peaks."""
    array = np.asarray(array, dtype=float)
    spread = np.ptp(array)
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite saliency feature map.")
    if spread < 1e-10:
        return np.zeros_like(array)
    normalized = (array - array.min()) / spread
    peaks = normalized[(normalized == maximum_filter(normalized, size=3, mode="reflect"))
                       & (normalized > 0) & (normalized < 1 - 1e-10)]
    mean_other_peak = peaks.mean() if peaks.size else 0.0
    return normalized * (1 - mean_other_peak) ** 2


def pyramid(array, levels=9):
    result = [array]
    for _ in range(1, levels):
        result.append(gaussian_filter(result[-1], sigma=1.0, mode="reflect")[::2, ::2])
    return result


def gabor_energy(array, angle):
    y, x = np.mgrid[-4:5, -4:5]
    rotated_x = x * np.cos(angle) + y * np.sin(angle)
    envelope = np.exp(-(x*x + y*y) / 8.0)
    real = envelope * np.cos(2 * np.pi * rotated_x / 4.0)
    imag = envelope * np.sin(2 * np.pi * rotated_x / 4.0)
    real -= real.mean()
    imag -= imag.mean()
    real /= np.sqrt(np.sum(real**2))
    imag /= np.sqrt(np.sum(imag**2))
    return np.hypot(convolve(array, real, mode="reflect"),
                    convolve(array, imag, mode="reflect"))


def center_surround_sum(levels, common_shape):
    features = []
    for center in (2, 3, 4):
        for offset in (3, 4):
            surround = resize_map(levels[center + offset], levels[center].shape)
            feature = np.abs(levels[center] - surround)
            features.append(competition_normalize(resize_map(feature, common_shape)))
    return np.sum(features, axis=0)


def low_level_saliency(image):
    rgb = np.asarray(image.convert("RGB"), dtype=float) / 255.0
    intensity = rgb.mean(axis=2)
    normalized_rgb = np.divide(rgb, intensity[..., None],
                               out=np.zeros_like(rgb),
                               where=intensity[..., None] > 0.1 * max(intensity.max(), 1e-12))
    red, green, blue = np.moveaxis(normalized_rgb, -1, 0)
    intensity_levels = pyramid(intensity)
    common_shape = intensity_levels[4].shape
    intensity_map = competition_normalize(center_surround_sum(intensity_levels, common_shape))
    color_map = competition_normalize(
        center_surround_sum(pyramid(red - green), common_shape)
        + center_surround_sum(pyramid(blue - (red + green) / 2), common_shape)
    )
    orientation_maps = []
    for angle in np.deg2rad([0, 45, 90, 135]):
        levels = [gabor_energy(level, angle) for level in intensity_levels]
        orientation_maps.append(competition_normalize(center_surround_sum(levels, common_shape)))
    orientation_map = competition_normalize(np.sum(orientation_maps, axis=0))
    combined = competition_normalize((intensity_map + color_map + orientation_map) / 3)
    shape = intensity.shape
    return {"low_level": resize_map(combined, shape),
            "intensity": resize_map(intensity_map, shape),
            "color": resize_map(color_map, shape),
            "orientation": resize_map(orientation_map, shape)}
