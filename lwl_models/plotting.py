"""Plot image-native maps inside the established stimulus/AOI geometry."""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
from PIL import Image
from .study import ROOT, TARGETS, PAIRS, STIM_BOX, AOI_BOX

MAP_LABELS = {
    "low_level": "Low-level saliency",
    "icnet": "ICNet complexity",
    "recognition": "SigLIP2 recognition",
    "contrast_related": "SigLIP2 related contrast",
    "contrast_unrelated": "SigLIP2 unrelated contrast",
    "contrast_difference": "Related − unrelated contrast",
}
MAIN_MAPS = tuple(MAP_LABELS)[:5]


def load_run(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError(f"Run is incomplete: {directory}")
    arrays = {}
    for target in manifest["targets"]:
        with np.load(directory / "arrays" / f"{target}.npz", allow_pickle=False) as archive:
            arrays[target] = {key: archive[key] for key in archive.files}
    return manifest, arrays


def color_style(key, manifest):
    if key in ("low_level", "icnet"):
        return "magma", Normalize(0, 1), "Relative saliency" if key == "low_level" else "Complexity"
    limit = max(manifest["display"]["signed_logit_limit"], 1e-8)
    return "RdBu_r", TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), "Score drop (logit units)"


def draw_map(ax, image, array, cmap, norm):
    if array.shape != (image.height, image.width):
        raise ValueError("Map resolution must match the original image.")
    ax.imshow(image, extent=STIM_BOX, origin="upper", zorder=1)
    ax.imshow(array, extent=STIM_BOX, origin="upper", cmap=cmap, norm=norm,
              alpha=0.75, interpolation="nearest", zorder=2)
    ax.add_patch(Rectangle((AOI_BOX[0], AOI_BOX[2]), AOI_BOX[1]-AOI_BOX[0],
                           AOI_BOX[3]-AOI_BOX[2], fill=False, edgecolor="0.3",
                           linestyle="--", linewidth=0.7, zorder=3))
    # Match the pooled R reference's viewport; maps only cover the image.
    ax.set_xlim(-49.94444444444444, 768.9444444444445)
    ax.set_ylim(153.36111111111111, 886.1388888888889)
    ax.set_aspect("equal")
    ax.set_xticks([0, 200, 400, 600])
    ax.set_yticks([300, 500, 700])
    ax.tick_params(labelsize=8)
    ax.set_axisbelow(True)
    ax.grid(color="0.9", linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_target_comparison(directory, target="comb"):
    manifest, arrays = load_run(directory)
    if target not in arrays:
        raise ValueError(f"Choose one of {list(arrays)}")
    fig, axes = plt.subplots(1, len(MAIN_MAPS), figsize=(19, 4.6), layout="constrained")
    with Image.open(ROOT / "stim_images" / f"{target}.jpeg") as source:
        image = source.convert("RGB")
    for ax, key in zip(axes, MAIN_MAPS):
        cmap, norm, unit = color_style(key, manifest)
        draw_map(ax, image, arrays[target][key], cmap, norm)
        title = MAP_LABELS[key]
        if key.startswith("contrast_"):
            condition = key.split("_")[1].capitalize()
            title += f"\n{target} vs {PAIRS[target][condition]}"
        ax.set_title(title, fontsize=10)
        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                     orientation="horizontal", shrink=0.8, pad=0.07, label=unit)
    fig.suptitle(f"{target}: candidate image-based predictors", fontsize=14)
    return fig


def plot_model_overview(directory, key="recognition"):
    manifest, arrays = load_run(directory)
    targets = [t for t in TARGETS if t in arrays]
    fig, axes = plt.subplots((len(targets)+3)//4, 4, figsize=(15, 8.8),
                             squeeze=False, layout="constrained")
    cmap, norm, unit = color_style(key, manifest)
    for ax, target in zip(axes.flat, targets):
        with Image.open(ROOT / "stim_images" / f"{target}.jpeg") as source:
            image = source.convert("RGB")
        draw_map(ax, image, arrays[target][key], cmap, norm)
        title = target
        if key in ("contrast_related", "contrast_unrelated"):
            title += f" vs {PAIRS[target][key.split('_')[1].capitalize()]}"
        ax.set_title(title, fontsize=10)
    for ax in list(axes.flat)[len(targets):]:
        ax.set_visible(False)
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=list(axes.flat),
                 orientation="horizontal", shrink=0.5, pad=0.05, label=unit)
    fig.suptitle(MAP_LABELS[key], fontsize=15)
    fig.supxlabel("Original reference coordinates; dashed box = left AOI; maps cover the image only")
    return fig


def save_figures(directory):
    directory = Path(directory)
    manifest, _ = load_run(directory)
    plot_dir = directory / "figures"
    plot_dir.mkdir(exist_ok=True)
    for key in MAP_LABELS:
        fig = plot_model_overview(directory, key)
        fig.savefig(plot_dir / f"{key}_overview.png", dpi=160)
        plt.close(fig)
    for target in manifest["targets"]:
        fig = plot_target_comparison(directory, target)
        fig.savefig(plot_dir / f"{target}_comparison.png", dpi=160)
        plt.close(fig)
