# Research context and current modeling plan

## Scientific question

Which image features predict where children look when hearing an object
name? Compare visual conspicuity, image complexity, word-level recognition
evidence, and evidence that distinguishes the target from its competitor.
Model agreement is predictive evidence for these computations, not by
itself a causal explanation of children's attention.

## Agreed first models

1. **Low-level saliency:** an explicitly labeled Itti-inspired multiscale
   center-surround implementation using intensity, opponent color, and
   orientation. It is a transparent baseline, not a claim to reproduce
   every detail of the original Itti implementation.
2. **IC9600 / ICNet:** use Feng et al.'s released pretrained model to obtain
   its global complexity score and weakly supervised local complexity map.
   This replaces the previously proposed DeepGaze baseline. Complexity
   is a separate hypothesis; ICNet is not trained to predict eye fixations.
3. **SigLIP2 recognition:** local image ablation, scoring the decrease in
   the target word's image-text match logit. The user's existing SigLIP2
   weights should be reused before any new model download.
4. **SigLIP2 competitor-category contrast:** the decrease in the margin
   between target-word and distractor-word logits on the same ablated
   target image. Use the actual Related and Unrelated distractor labels.

Use the same masks, prompts, and model for recognition and contrast.
Keep signed importance: a negative value means the ablation increased
the score. Positive-only maps may be supplied separately for inspection.
SigLIP sigmoid match scores are not a softmax probability over the eight
labels, and attribution maps are not calibrated fixation probabilities.

For target image T and an ablation A_r of region r:

```
recognition(r) = s(T, target_word) - s(A_r(T), target_word)
contrast(r) = [s(T, target_word) - s(T, distractor_word)]
              - [s(A_r(T), target_word) - s(A_r(T), distractor_word)]
```

Subtracting an unchanged distractor-image score would cancel and give
the recognition map again. The first contrast model deliberately uses
the competitor's **category label**, not visual features of the actual
distractor photograph. Actual-image contrast, alternative backbones,
and active information-gathering models are later extensions.

## Implementation status

The first run is implemented in `lwl_models/` and documented in `MODELS.md`.
Use `notebooks/model_hypotheses.ipynb` to inspect its local results inline.
The `baseline_v1` run uses 100px blur patches, 50px stride, and a 20px blur
radius: 81 ablations per image, using cached SigLIP2 base weights. ICNet's
official pretrained weights and source are kept in the ignored local cache.

The completed first run ranked the intended label first for all eight
intact images. Related and Unrelated contrast maps differ for every target.
ICNet shows some prominent image-border responses; these are preserved
for inspection, not removed to improve agreement with gaze. All three
model families remain unevaluated against held-out children's looking.

## Data and geometry

Eight targets each have one Related and one Unrelated competitor:

| Target | Related | Unrelated |
| --- | --- | --- |
| apple | banana | bathtub |
| banana | apple | toothbrush |
| bathtub | potty | apple |
| carrots | peas | potty |
| comb | toothbrush | peas |
| peas | carrots | comb |
| potty | bathtub | carrots |
| toothbrush | comb | banana |

The existing analysis selects tracked, non-interpolated samples in
3550–5050 ms, inclusive, and translates right-target X coordinates by
−1200px. The defaults retain 43,854 target samples from the supplied CSV.
The pooled gaze maps are descriptive; testing contrast requires preserving
target × distractor condition, and eventually time relative to word onset.

Preserve the root reference PNG's increasing-Y plot convention and upright
images. The 500×500 stimulus occupies X=110–610, Y=290–790; the left AOI is
X=0–620, Y=190–810. Model maps belong to the image rectangle, not the whole
AOI. Buffer space is not evidence of zero model saliency; models predicting
only the image do not predict gaze outside it.

## Interpretation and later validation

- Acquisition geometry and actual spoken-word timing remain unverified.
  Matching the reference rendering does not independently validate them.
- Within each target, condition and screen side are confounded. Account
  for spatial/entry biases and inspect pre-word looking where possible.
- Samples are temporally dependent and dwell-weighted; they are not
  independent fixation events. Use participant/trial-level uncertainty.
- There are only eight object exemplars. Participant and image
  generalization must be distinguished.
- Check intact-image naming scores before interpreting ablations. Retain
  failures rather than selecting prompts using the children's gaze.
- Before model comparison, check whether model maps and condition effects
  are distinguishable and reliable. Use held-out location likelihoods with
  shared spatial support, smoothing, and a spatial baseline; fit any
  score-to-gaze calibration on training data only.
- Keep feature claims separate from location claims. Color, texture,
  shape, mask-size, and replacement-baseline controls are later tests.
- CDI word knowledge may moderate model fit after participant matching
  and missingness checks. It is not part of this first implementation.

## Sources

- [Itti, Koch & Niebur (1998)](https://hasler.ece.gatech.edu/Courses/MachineLearning/FoundationalPapers/Itti_Koch_Neiber1998.pdf)
- [IC9600 paper](https://doi.org/10.1109/TPAMI.2022.3232328)
  (published online in 2022; TPAMI 45(7), 8577–8593, 2023)
- [Official IC9600 code and checkpoint](https://github.com/tinglyfeng/IC9600)
- [SigLIP2 base model](https://huggingface.co/google/siglip2-base-patch16-224)
- [SigLIP2 preprocessing documentation](https://huggingface.co/docs/transformers/model_doc/siglip2)
- [RISE: perturbation-based attribution](https://arxiv.org/abs/1806.07421)
