# Python gaze heatmaps

Open `gaze_heatmaps.ipynb` and run all cells. In the IDE notebook kernel
picker, choose **Python Environments → davinci**, using this interpreter:

```text
/data/kushinm/miniconda3/envs/davinci/bin/python
```

This existing Conda environment uses Python 3.10.16 and already contains
the required packages. No package installation or kernel registration is
needed. Its verified dependencies are listed in `requirements.txt`.

The notebook works with its working directory at the repository root or
inside `notebooks/`. Both CSV inputs remain local; only the gaze CSV is
used for these plots. The notebook displays aggregate counts, a heatmap
overview, one selectable target, and a gaze-scatter overview. It can also
display every target separately and show the original reference PNG.

Edit the settings cell to change the time window, included conditions,
or target shown in detail. Defaults reproduce `gaze_location_lwl.R`:
3550–5050 ms inclusive, tracked and non-interpolated samples, target-only
heatmaps, and both conditions pooled. The product-kernel density estimate
uses the same bandwidth rule and shared 100×100 grid as R's `MASS::kde2d`.
Fonts and color interpolation can differ between plotting libraries.

The increasing-Y display convention, upright images, 1200px horizontal
translation, and offset 500px image within the 620px AOI match the root
reference PNG. No extra AOI clipping or image resizing removes the buffer.
These choices reproduce the existing analysis; acquisition geometry and
word timing still need independent verification.

With `SAVE_FIGURES = True`, PNGs are also written to the Git-ignored
`outputs/python_gaze_inspection/` directory. Saved notebook outputs contain
only aggregate counts and plots. Do not display or save raw participant rows
or identifiers in notebook outputs.
