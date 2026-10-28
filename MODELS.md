# Computational predictors

The hypotheses and agreed scope are in [RESEARCH_PLAN.md](RESEARCH_PLAN.md).
This first implementation produces candidate spatial predictors; it does
not fit them to children's gaze or rank the hypotheses by behavioral fit.

## Environment and execution

Use the existing **davinci** Python 3.10 environment, also used for the
gaze notebook. All dependencies in `requirements-models.txt` are already
installed there. No shared environment changes are required.

From the repository root:

```bash
conda activate davinci
python -m lwl_models.setup_assets
python -m lwl_models.pipeline
```

The equivalent interpreter on this machine is
`/data/kushinm/miniconda3/envs/davinci/bin/python`.
The pipeline selects CUDA when available; this machine's first run uses
CPU, four threads, and batches of eight images. Use `--device`, `--threads`,
and `--batch-size` to change these settings.

Default output: `outputs/model_predictions/baseline_v1/`. Existing runs
are protected: choose another `--output` for a new configuration, or use
`--overwrite` explicitly. The local gaze CSV is optional; when present,
only its design columns are read to check all 16 target/condition pairs.

## Reusing weights

Inference always uses `local_files_only=True`. It never downloads SigLIP2
weights or sends images or gaze data to a remote inference service.

The cached model is `google/siglip2-base-patch16-224`, pinned at revision
`75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`. The loader searches the configured
Hugging Face caches, including both the legacy root and its `hub/` subfolder.
Use `--siglip-dir` and `--tokenizer-dir` to specify local directories.

The existing base cache lacks tokenizer files. The complete cached
`siglip2-so400m-patch14-384` tokenizer has **the same published file hashes**
as the pinned base model, so it is safely reused with the base weights.
All three tokenizer hashes and the model-weight checksum are verified.
This does not load or run the larger model.

For another machine without these assets, downloading SigLIP2 is explicit:

```bash
python -m lwl_models.setup_assets --download-siglip
```

The setup command retrieves ICNet's public checkpoint and pinned official
`ICNet.py` into `.model-cache/ic9600/`, checking SHA-256 hashes. Source comes
from revision `3891e25ad1605763995801c1ddbba218a03f8720` of the
[official repository](https://github.com/tinglyfeng/IC9600). The upstream
source is downloaded locally rather than vendored into this repository.
The released state dictionary is loaded strictly with `weights_only=True`;
no separate ResNet pretraining download is needed.

## Model definitions

**Low-level saliency:** an Itti-inspired approximation with a nine-level
Gaussian pyramid, center levels 2/3/4 and surround offsets 3/4. Intensity,
red–green/blue–yellow opponent contrasts, and quadrature Gabor energy at
0/45/90/135 degrees are normalized with local-maximum competition and
combined with equal modality weights. Computation uses common pyramid
level 4, then upsamples to the original image. It is not an exact port of
Itti et al.; the explicit implementation is in `lwl_models/low_level.py`.
The first run processes isolated target images, not full paired displays.

**ICNet complexity:** the released model's global score and 64×64 map,
bilinearly resized to the 500×500 image using the official preprocessing
(512×512 RGB, ImageNet channel normalization). Both native and resized
maps are saved. The map is weakly supervised by image complexity labels;
it is not a learned gaze distribution or a purely low-level predictor.

**SigLIP2 recognition:** use the pre-sigmoid match logit for the target's
prompt. Default prompts are `a photo of {label}.`, lowercase, with fixed
64-token padding and truncation. Use the pinned model's image processor.
Intact scores for all eight labels are recorded before ablation. Sigmoid
matches are supplied for inspection but do not sum to one over labels.

**SigLIP2 competitor-category contrast:** attribute the margin between the
target-word and distractor-word logits, both evaluated on the same target
image. Related and Unrelated competitors come from `lwl_models/study.py`.
These are semantic competitor maps; the distractor's actual pixels are not
inputs to this version of the contrast model.

## Ablations and outputs

The default uses 100×100px patches on a 50px stride: 81 ablations per 500px
image. Each patch is replaced by the corresponding region of a globally
Gaussian-blurred copy (20px radius). Pixels outside the selected patch
remain unchanged. SigLIP's processor then resizes the ablated image to
224×224. Thus these are coarse region importance estimates, not independent
pixel effects; the small original image features may be below model resolution.

For each patch, recognition is the intact-minus-ablated target logit.
Contrast is the intact-minus-ablated target/competitor margin, equivalently
target logit drop minus competitor logit drop. Scores are spread uniformly
over the affected patch and averaged where patches overlap. Full edge
coverage is required. Positive values support the target score/margin;
negative values mean masking increased it. Signed maps and separate
positive-only maps are saved. Neither is a calibrated fixation distribution.

Each run saves:

- `manifest.json`: settings, versions, pinned weights/checksums, image
  hashes, model paths, pairings, intact naming checks, and runtime.
- `arrays/<target>.npz`: raw saliency channels, ICNet maps/score, signed and
  positive ablation maps, both competitor maps, their difference, every
  ablated logit vector, patch boxes, and overlap coverage.
- `figures/`: model overviews and one five-model comparison per target.

All arrays use image-native rows increasing downward. Plotting uses
`origin="upper"` for both the image and its model map, preserving the
reference's upright image and increasing-Y axes. The image remains 500px
inside the offset 620px AOI. No model prediction is fabricated for the
buffer outside the image. Signed SigLIP2 panels share a symmetric color
scale across targets and conditions; ICNet maps use their native [0,1]
scale. Low-level maps have their algorithm's relative saliency scale.

Outputs, downloaded code, and checkpoints are Git-ignored. Only aggregate
model results should be displayed in notebooks.

## Sensitivity checks and tests

Change one setting at a time and use another output directory:

```bash
python -m lwl_models.pipeline --patch-size 150 --stride 75 --output outputs/model_predictions/patch150
python -m lwl_models.pipeline --mode mean --output outputs/model_predictions/mean_baseline
python -m unittest discover -s tests -v
```

Analytic tests cover signed margins, the competitor effect, full edge
coverage, overlap averaging, spatially local perturbations, low-level
responses to flat images, and map/image/AOI alignment. Each inference run
also checks cached-text SigLIP scoring against its full forward pass.

Masking can introduce artifacts and different baselines can change the
attributions. Do not choose prompts or masking settings to maximize fit to
the same gaze data used for final evaluation. Inspect naming failures and
map disagreements before expanding the model set.
