"""Generate low-level, ICNet, and SigLIP2 ablation maps for the study images."""

import argparse
import importlib.metadata
import json
import os
import time
from pathlib import Path
from datetime import datetime, timezone

from .study import ROOT, TARGETS, PAIRS, validate_pairs
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".model-cache" / "matplotlib"))

import numpy as np
from PIL import Image
import torch
from scipy.special import expit
from .assets import sha256
from .backends import ICNetPredictor, SiglipScorer, resolve_device
from .ablation import patch_boxes, ablated_images, score_effects, aggregate_patch_effects
from .low_level import low_level_saliency


def run(output, targets=TARGETS, patch_size=100, stride=50, mode="blur", blur_radius=20.0,
        batch_size=8, threads=4, device="auto", prompt_template="a photo of {label}.",
        siglip_dir=None, tokenizer_dir=None, overwrite=False):
    output = Path(output)
    if (output / "manifest.json").exists() and not overwrite:
        raise FileExistsError("Run already exists. Choose a new output path or explicitly use --overwrite.")
    if not targets or len(set(targets)) != len(targets) or not set(targets) <= set(TARGETS):
        raise ValueError("Choose distinct study targets.")
    if threads < 1:
        raise ValueError("Threads must be positive.")
    patch_boxes((500, 500), patch_size, stride)  # Fail on invalid settings before loading weights.
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    device = resolve_device(device)
    output.mkdir(parents=True, exist_ok=True)
    (output / "arrays").mkdir(exist_ok=True)
    local_csv = ROOT / "full_lwl_aois_Aug3.csv"
    pair_check = validate_pairs(local_csv) if local_csv.exists() else "CSV absent; using documented pairs"
    started = time.perf_counter()
    manifest = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "targets": list(targets), "pairs": PAIRS, "local_design_validation": pair_check,
        "settings": {"patch_size": patch_size, "stride": stride, "ablation_mode": mode,
                     "blur_radius": blur_radius, "batch_size": batch_size, "threads": threads,
                     "device": device, "prompt_template": prompt_template},
        "versions": {name: importlib.metadata.version(name) for name in
                     ("numpy", "scipy", "Pillow", "torch", "torchvision", "transformers")},
        "geometry": "Arrays are image-native top-down rows. Plot upright in the reference image box; do not fill AOI buffer.",
        "contrast_kind": "competitor category label, not distractor photograph",
        "low_level": {"kind": "Itti-inspired approximation", "pyramid_levels": 9,
                      "centers": [2, 3, 4], "surround_offsets": [3, 4],
                      "orientations_degrees": [0, 45, 90, 135], "common_pyramid_level": 4},
        "completed_targets": [], "results": {},
    }

    def write_manifest():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    write_manifest()
    print(f"Loading ICNet and cached SigLIP2 on {device}...", flush=True)
    icnet = ICNetPredictor(device=device)
    siglip = SiglipScorer(TARGETS, prompt_template, device, siglip_dir, tokenizer_dir)
    manifest["icnet"] = icnet.provenance
    manifest["siglip"] = siglip.provenance
    with Image.open(ROOT / "stim_images" / f"{targets[0]}.jpeg") as source:
        manifest["siglip"]["cached_text_forward_max_error"] = siglip.verify_forward(source.convert("RGB"))
    signed_limit = 0.0
    for index, target in enumerate(targets):
        target_started = time.perf_counter()
        image_path = ROOT / "stim_images" / f"{target}.jpeg"
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        if image.size != (500, 500):
            raise ValueError(f"Expected 500x500 stimulus: {image_path}")
        boxes = patch_boxes(image.size, patch_size, stride)
        print(f"[{index+1}/{len(targets)}] {target}: low-level, complexity, {len(boxes)} ablations", flush=True)
        maps = low_level_saliency(image)
        complexity_score, complexity, native_complexity = icnet.predict(image)
        intact = siglip.score([image], batch_size)[0]
        ablated = siglip.score(ablated_images(image, boxes, mode, blur_radius), batch_size)
        effects = score_effects(intact, ablated, TARGETS.index(target),
                                {condition: TARGETS.index(other) for condition, other in PAIRS[target].items()})
        for key, values in effects.items():
            maps[key], coverage = aggregate_patch_effects(boxes, values, image.size)
            maps[f"{key}_positive"] = np.maximum(maps[key], 0)
        maps["contrast_difference"] = maps["contrast_related"] - maps["contrast_unrelated"]
        signed_limit = max(signed_limit, *(float(np.max(np.abs(maps[key])))
                           for key in (*effects, "contrast_difference")))
        maps.update(icnet=complexity, icnet_native=native_complexity,
                    icnet_score=np.asarray(complexity_score), boxes=boxes, coverage=coverage,
                    intact_logits=intact, ablated_logits=ablated,
                    prompt_labels=np.asarray(TARGETS))
        if not all(np.isfinite(value).all() for key, value in maps.items() if key != "prompt_labels"):
            raise ValueError(f"Nonfinite model output for {target}")
        np.savez_compressed(output / "arrays" / f"{target}.npz", **maps)
        best_label = TARGETS[int(np.argmax(intact))]
        result = {
            "image_sha256": sha256(image_path), "complexity_score": complexity_score,
            "patch_count": len(boxes), "top_label": best_label, "target_is_top_label": best_label == target,
            "intact_logits": dict(zip(TARGETS, map(float, intact))),
            "intact_sigmoid_matches": dict(zip(TARGETS, map(float, expit(intact)))),
            "negative_recognition_pixel_fraction": float(np.mean(maps["recognition"] < 0)),
            "contrast_difference_max_abs": float(np.max(np.abs(maps["contrast_difference"]))),
            "elapsed_seconds": round(time.perf_counter() - target_started, 3),
        }
        manifest["results"][target] = result
        manifest["completed_targets"].append(target)
        write_manifest()
        print(f"  intact best label: {best_label}; complexity: {complexity_score:.3f}; "
              f"{result['elapsed_seconds']:.1f}s", flush=True)
    manifest["display"] = {"signed_logit_limit": signed_limit,
                           "signed_map_colors": "red = supports target/margin; blue = opposing evidence"}
    manifest["status"] = "complete"
    manifest["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    write_manifest()
    from .plotting import save_figures
    save_figures(output)
    manifest["figures_written"] = True
    write_manifest()
    print(f"Complete: {output}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "model_predictions" / "baseline_v1")
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--patch-size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--mode", choices=("blur", "mean"), default="blur")
    parser.add_argument("--blur-radius", type=float, default=20.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--prompt-template", default="a photo of {label}.")
    parser.add_argument("--siglip-dir", type=Path)
    parser.add_argument("--tokenizer-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = vars(parser.parse_args())
    run(**args)


if __name__ == "__main__":
    main()
