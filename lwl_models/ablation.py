"""Shared ablations and signed attribution for recognition and contrast."""

import numpy as np
from PIL import Image, ImageFilter

GAUSSIAN_SIGMA_FRACTION = 0.45


def patch_boxes(size, patch_size=64, stride=24):
    """Return (left, top, right, bottom) rectangles with complete coverage."""
    width, height = size
    if not (0 < stride <= patch_size <= min(width, height)):
        raise ValueError("Require 0 < stride <= patch_size <= image dimensions.")

    def starts(length):
        return sorted(set(range(0, length - patch_size + 1, stride)) | {length - patch_size})

    return np.asarray([(x, y, x + patch_size, y + patch_size)
                       for y in starts(height) for x in starts(width)], dtype=np.int64)


def patch_mask(box, mask_shape="gaussian"):
    """Return the ablation strength inside one box, including its edge pixels."""
    left, top, right, bottom = map(int, box)
    width, height = right - left, bottom - top
    if width <= 0 or height <= 0:
        raise ValueError("Patch boxes must have positive area.")
    if mask_shape == "hard":
        return np.ones((height, width), dtype=np.float32)
    if mask_shape != "gaussian":
        raise ValueError("Mask shape must be 'hard' or 'gaussian'.")
    # Sigma is 0.45 of the half-width. Quantize to the same 8-bit alpha used
    # by PIL so map aggregation uses the mask that the model actually sees.
    x = (np.arange(width) + 0.5 - width / 2) / (width / 2)
    y = (np.arange(height) + 0.5 - height / 2) / (height / 2)
    mask = np.exp(-0.5 * ((x[None, :] / GAUSSIAN_SIGMA_FRACTION) ** 2 +
                          (y[:, None] / GAUSSIAN_SIGMA_FRACTION) ** 2))
    mask /= mask.max()
    return np.rint(mask * 255).astype(np.uint8).astype(np.float32) / 255


def ablated_images(image, boxes, mode="blur", blur_radius=20.0, mask_shape="gaussian"):
    """Replace each region with a fixed baseline, optionally with soft edges."""
    if mode == "blur":
        if blur_radius <= 0:
            raise ValueError("Blur radius must be positive.")
        replacement = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    elif mode == "mean":
        mean = tuple(np.asarray(image.convert("RGB")).mean(axis=(0, 1)).round().astype(int))
        replacement = Image.new("RGB", image.size, mean)
    else:
        raise ValueError("Ablation mode must be 'blur' or 'mean'.")
    for box in boxes:
        result = image.copy()
        region = tuple(map(int, box))
        if mask_shape == "hard":
            result.paste(replacement.crop(region), region)
        else:
            mask = Image.fromarray(np.rint(patch_mask(box, mask_shape) * 255).astype(np.uint8))
            blended = Image.composite(replacement.crop(region), image.crop(region), mask)
            result.paste(blended, region)
        yield result


def aggregate_patch_effects(boxes, effects, size, mask_shape="gaussian"):
    """Spread each patch effect according to its mask, averaging overlaps.

    This is regional attribution at patch resolution, not per-pixel causal
    identification. No extra smoothing or within-map rescaling is applied.
    """
    width, height = size
    effects = np.asarray(effects, dtype=float)
    if effects.shape != (len(boxes),) or not np.isfinite(effects).all():
        raise ValueError("Expected one finite score change for every patch.")
    total = np.zeros((height, width), dtype=float)
    coverage = np.zeros((height, width), dtype=float)
    for (left, top, right, bottom), effect in zip(boxes, effects):
        mask = patch_mask((left, top, right, bottom), mask_shape)
        total[top:bottom, left:right] += effect * mask
        coverage[top:bottom, left:right] += mask
    if not np.all(coverage > 0):
        raise ValueError("Patch grid leaves unmeasured image pixels.")
    return total / coverage, coverage


def score_effects(intact_logits, ablated_logits, target_index, competitor_indices):
    """Use pre-sigmoid logits; retain negative evidence as well as support."""
    intact_logits = np.asarray(intact_logits, dtype=float)
    ablated_logits = np.asarray(ablated_logits, dtype=float)
    if ablated_logits.ndim != 2 or intact_logits.shape != ablated_logits.shape[1:]:
        raise ValueError("Intact/ablated logits must share the same ordered prompts.")
    drops = intact_logits[None, :] - ablated_logits
    result = {"recognition": drops[:, target_index]}
    for condition, index in competitor_indices.items():
        # Competitor word is scored on the SAME masked target image.
        result[f"contrast_{condition.lower()}"] = drops[:, target_index] - drops[:, index]
    return result
