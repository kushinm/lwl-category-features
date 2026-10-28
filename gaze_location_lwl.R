## ============================================================
## Gaze location within the Target AOI
## ============================================================
## Data: full_lwl_aois_Aug3.csv (Looking-While-Listening eyetracking data)
##
## Goal: For every trial, figure out which AOI (screen side) held the
## target object -- "left" when TargetObjectPos == "centerLeft", "right"
## when TargetObjectPos == "centerRight". Keep the raw gaze samples that
## landed in the Target AOI (distractor-side looks and trials are
## excluded -- since a target's distractor image differs by Condition
## anyway, pooling distractor looks across conditions has no single
## stimulus image to show alongside them), collapsed across Condition
## (Related + Unrelated trials pooled together), and plot where within
## the Target AOI box those samples fall for each TargetImage -- with the
## actual AOI bounding box drawn on the plot AND the actual stimulus image
## rendered behind the gaze points/density so you can see exactly what the
## participant was looking at.
## ============================================================

## Optional project-local packages, separated by R major/minor version.
local_library <- file.path(".R-library", paste(R.version$major,
  strsplit(R.version$minor, ".", fixed = TRUE)[[1]][1], sep = "."))
if (dir.exists(local_library)) .libPaths(c(local_library, .libPaths()))

library(dplyr)
library(ggplot2)
library(readr)
library(jpeg)        # readJPEG() -- install.packages("jpeg") if needed

## ---- 0. Paths -------------------------------------------------
data_path   <- "full_lwl_aois_Aug3.csv"   # edit if the file lives elsewhere
args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args)) args[[1]] else file.path("outputs", "gaze_inspection")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

## Folder holding the stimulus images, one file per TargetImage/
## DistracterImage value, e.g. stim_images/apple.jpeg, stim_images/banana.jpeg
## Edit image_dir / image_ext to match where your files actually live.
image_dir <- "stim_images"
image_ext <- ".jpeg"
image_path <- function(name) file.path(image_dir, paste0(name, image_ext))

## ---- 1. Stimulus & AOI geometry (assumed from prior script comments) ----
## WordSearch_updated.py is not available in this repository, so these
## acquisition settings remain provisional until checked against source.
## Two different rectangles are in play here, and they are NOT the same box:
##
## (a) The IMAGE itself. self.imageMatrix loads curTargetPic/curDistracterPic
##     with no .size override, so they render at native size -- confirmed
##     500 x 500 px for these stimulus files. Their position comes from
##     self.pos['centerLeft'] = (-600, 0) / ['centerRight'] = (600, 0), which
##     are PsychoPy pix-unit coordinates: origin at screen center, x right+,
##     y up+. The gaze data (GazePointXMean/YMean, range 0-1920 x 0-1080)
##     and the AOI objects below use a top-left-origin, y-down pixel system
##     instead, so we convert: pixel_x = screen_width/2 + psychopy_x,
##     pixel_y = screen_height/2 - psychopy_y.
##
## (b) The AOI HIT BOX used to compute LookAOI, i.e. self.aoiLeft =
##     AOI('rectangle', pos=(0,190), size=(620,620)) and self.aoiRight =
##     AOI('rectangle', pos=(1300,190), size=(620,620)) -- already in the
##     same top-left-origin pixel system as the gaze data. These are a
##     620x620 region, offset from (not centered on) the image, i.e. the
##     gaze-detection region is intentionally larger/shifted relative to
##     the picture -- it is NOT just "the image's own bounding box".
##
## We keep both: aoi_boxes (the dashed hit-box outline, used to classify
## AOIType and matches LookAOI exactly) and stim_boxes (where the image
## itself is drawn).
screen_width  <- 1920
screen_height <- 1080

## -- (a) image placement --
stim_width  <- c(-250, 250)   # 500 px wide, centered
stim_height <- c(-250, 250)   # 500 px tall, centered
left_stim_coords  <- c(screen_width / 2 - 600, screen_height / 2)   # (360, 540)
right_stim_coords <- c(screen_width / 2 + 600, screen_height / 2)   # (1560, 540)

## -- (b) AOI hit-box placement --
aoi_size <- c(620, 620)
aoi_left_topleft  <- c(0, 190)
aoi_right_topleft <- c(1300, 190)

## Horizontal translation between the two stim centers -- used to slide a
## right-placed trial's raw gaze coordinates onto the left-stim frame.
## IMPORTANT: this must be a TRANSLATION (a fixed shift), not a mirror/
## reflection. WordSearch_updated.py never applies flipHoriz, a negative
## scale, or an .ori change to curTargetPic between the centerLeft and
## centerRight placements -- it's the literal same image file, just moved
## 1200 px over. Reflecting the coordinates instead (e.g. about the
## screen midpoint) would flip left-right *within* the image itself: a
## gaze sample that fell right-of-center on a right-placed trial would
## get plotted left-of-center after "normalizing," when it was actually
## still right-of-center on the (unflipped) picture. That the two AOI
## hit-boxes ([0,620] and [1300,1920]) happen to be exact mirror images of
## each other about the screen's midline is a coincidence of how they were
## independently defined (both flush to their own side's screen edge) --
## not evidence that the image content should be reflected too.
x_shift <- right_stim_coords[1] - left_stim_coords[1]   # 1200

## Once coordinates are shifted (see step 2 below), the Target box is
## always at the left-stim location and the Distractor box is always at
## the right-stim location -- so each tibble only needs one fixed entry
## per role, not per screen side.
aoi_boxes <- tibble(
  AOI  = c("Target", "Distractor"),
  xmin = c(aoi_left_topleft[1],               aoi_right_topleft[1]),
  xmax = c(aoi_left_topleft[1] + aoi_size[1], aoi_right_topleft[1] + aoi_size[1]),
  ymin = c(aoi_left_topleft[2],               aoi_right_topleft[2]),
  ymax = c(aoi_left_topleft[2] + aoi_size[2], aoi_right_topleft[2] + aoi_size[2])
)

stim_boxes <- tibble(
  AOI  = c("Target", "Distractor"),
  xmin = c(left_stim_coords[1] + stim_width[1],  right_stim_coords[1] + stim_width[1]),
  xmax = c(left_stim_coords[1] + stim_width[2],  right_stim_coords[1] + stim_width[2]),
  ymin = c(left_stim_coords[2] + stim_height[1], right_stim_coords[2] + stim_height[1]),
  ymax = c(left_stim_coords[2] + stim_height[2], right_stim_coords[2] + stim_height[2])
)

## ---- 1b. Custom geom: draw an image exactly inside a data-space box ---
## ggimage::geom_image() sizes images as a fraction of the panel, which
## makes it fiddly to line an image up exactly with a data-defined AOI
## box. This little geom instead reads each row's image file and places
## it with grid::rasterGrob() using the coordinates ggplot has already
## transformed for the panel, so it fills exactly [xmin,xmax] x [ymin,ymax]
## in data units -- i.e. exactly the AOI box -- on every facet panel that
## row belongs to.
.image_cache <- new.env(parent = emptyenv())

read_image_cached <- function(path) {
  if (!is.null(.image_cache[[path]])) return(.image_cache[[path]])
  img <- tryCatch(jpeg::readJPEG(path), error = function(e) {
    warning("Could not read image: ", path, " (", conditionMessage(e), ")")
    NULL
  })
  .image_cache[[path]] <- img
  img
}

GeomImageRect <- ggplot2::ggproto(
  "GeomImageRect", ggplot2::Geom,
  required_aes = c("xmin", "xmax", "ymin", "ymax", "image"),
  default_aes  = ggplot2::aes(alpha = 1),
  draw_key     = ggplot2::draw_key_blank,
  draw_panel = function(data, panel_params, coord) {
    grobs <- lapply(seq_len(nrow(data)), function(i) {
      row <- data[i, ]
      img <- read_image_cached(row$image)
      if (is.null(img)) return(grid::nullGrob())

      corners   <- data.frame(x = c(row$xmin, row$xmax), y = c(row$ymin, row$ymax))
      corners_t <- coord$transform(corners, panel_params)

      grid::rasterGrob(
        img,
        x      = mean(corners_t$x),
        y      = mean(corners_t$y),
        width  = abs(diff(corners_t$x)),
        height = abs(diff(corners_t$y)),
        interpolate = TRUE,
        gp = grid::gpar(alpha = row$alpha)
      )
    })
    do.call(grid::grobTree, grobs)
  }
)

geom_image_rect <- function(mapping = NULL, data = NULL, ...,
                             show.legend = FALSE, inherit.aes = FALSE) {
  ggplot2::layer(
    geom        = GeomImageRect,
    mapping     = mapping,
    data        = data,
    stat        = "identity",
    position    = "identity",
    show.legend = show.legend,
    inherit.aes = inherit.aes,
    params      = list(...)
  )
}

## ---- 2. Load & clean -------------------------------------------------
gaze <- read_csv(data_path, show_col_types = FALSE)

gaze_clean <- gaze %>%
  mutate(
    Condition       = factor(Condition, levels = c("Unrelated", "Related")),
    TargetObjectPos = factor(TargetObjectPos, levels = c("centerLeft", "centerRight")),
    TargetImage     = factor(TargetImage)
  ) %>%
  # keep samples that actually have a gaze point, were tracked, and fell
  # into one of the two AOIs (left/right)
  filter(
    !is.na(GazePointXMean),
    !is.na(GazePointYMean),
    !is.na(LookAOI),
    isTracked == "yes",
    isInterpolatedFrame == 0,
    TimeBinMs>=3550 & TimeBinMs<=5050
  ) %>%
  mutate(
    TargetAOI     = if_else(TargetObjectPos == "centerLeft", "left", "right"),
    DistracterAOI = if_else(TargetObjectPos == "centerLeft", "right", "left"),
    # every remaining valid look is on the target's side or the
    # distractor's side, since those are the only two AOIs
    AOIType = case_when(
      LookAOI == TargetAOI     ~ "Target",
      LookAOI == DistracterAOI ~ "Distractor",
      TRUE                     ~ NA_character_
    ),
    AOIType = factor(AOIType, levels = c("Target", "Distractor")),
    # Shift (translate) the x-coordinate for trials where the target was
    # on the right, so that every trial's Target look lands in the same
    # local frame as a left-placed trial -- this makes plots comparable
    # across trials/subjects regardless of which physical side the target
    # happened to be on. This is a plain shift, NOT a mirror reflection
    # (see x_shift comment above for why that distinction matters).
    # (y is unaffected -- both stim locations share the same y-center.)
    GazePointXMean_norm = if_else(
      TargetAOI == "right",
      GazePointXMean - x_shift,
      GazePointXMean
    )
  ) %>%
  filter(!is.na(AOIType))

cat(
  "Kept", nrow(gaze_clean), "of", nrow(gaze), "rows",
  "(valid gaze samples landing in either AOI)\n"
)

## ---- 3. Sample sizes per cell (sanity check) -------------------------
## Full breakdown (both AOIType and Condition) for reference, even though
## the plots below focus on Target only, collapsed across Condition.
sample_counts <- gaze_clean %>%
  count(TargetImage, Condition, AOIType, name = "n_samples")
print(sample_counts)

## Target-only subset used for plotting below.
gaze_target <- gaze_clean %>% filter(AOIType == "Target")

## Helper: draw the Target AOI box on any plot. Note: this box is drawn at
## its actual left-side coordinates; it is NOT re-derived from x_shift.
## Because the AOI hit-box's own left/right shift (1300 px, flush to each
## side's screen edge) differs slightly from the image's (1200 px, see
## x_shift above), a few right-placed trials' samples that legitimately
## counted as AOI hits may, after translation, land up to ~100px outside
## this drawn outline near an edge. This is a minor display artifact of
## the two rectangles not sharing one offset -- it does not affect which
## samples were included (that was already decided by LookAOI upstream).
aoi_box_layer <- geom_rect(
  data = aoi_boxes %>% filter(AOI == "Target"),
  aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
  inherit.aes = FALSE, fill = NA, color = "grey30", linetype = "dashed",
  linewidth = 0.5
)

## ---- 3b. Stimulus-image lookup: one Target image per TargetImage -----
## Placed in the Target stim_box (the image's real 500x500 render
## location -- see stim_boxes above, not aoi_boxes, which is the larger,
## offset 620x620 gaze hit-box). The Target image doesn't vary by
## Condition, so this is a plain 1:1 lookup with no ambiguity to worry
## about (unlike the distractor image, which is why we're leaving
## distractor looks out of these plots entirely).
target_box <- stim_boxes %>% filter(AOI == "Target")

stim_images <- gaze_target %>%
  distinct(TargetImage) %>%
  mutate(
    TargetImage = as.character(TargetImage),
    image = image_path(TargetImage),
    xmin = target_box$xmin, xmax = target_box$xmax,
    ymin = target_box$ymin, ymax = target_box$ymax,
    TargetImage = factor(TargetImage, levels = levels(gaze_clean$TargetImage))
  )

if (any(!file.exists(stim_images$image))) {
  stop("Missing stimulus images: ",
       paste(stim_images$image[!file.exists(stim_images$image)], collapse = ", "))
}

stim_image_layer <- geom_image_rect(
  data = stim_images,
  aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, image = image)
)

## ---- 4. Plot 1: raw gaze scatter within the Target AOI ---------------
## One panel per TargetImage; collapsed across Condition (Related +
## Unrelated trials pooled together); distractor looks excluded.
p_scatter <- ggplot(gaze_target, aes(x = GazePointXMean_norm, y = GazePointYMean)) +
  stim_image_layer +
  aoi_box_layer +
  geom_point(color = "#0072B2", alpha = 0.35, size = 0.7) +
  # Preserve the original increasing-Y plot convention used by the root
  # reference PNG. Do not change this without verifying acquisition geometry.
  coord_fixed() +
  facet_grid(TargetImage ~ .) +
  labs(
    title = "Gaze locations within the Target AOI box",
    subtitle = "Collapsed across Condition; distractor looks excluded; dashed box = gaze AOI hit-box; x-coords shifted (not mirrored) so Target is always on the left",
    x = "Gaze X, normalized so Target = left (screen px)", y = "Gaze Y (screen px)"
  ) +
  theme_minimal(base_size = 10) +
  theme(legend.position = "none")

ggsave(file.path(out_dir, "gaze_scatter_by_target.png"),
       p_scatter, width = 6, height = 16, dpi = 300, limitsize = FALSE)

## ---- 5. Plot 2: 2D density ("heatmap") within the Target AOI ---------
## One panel per TargetImage; collapsed across Condition (Related +
## Unrelated trials pooled together); distractor looks excluded.
make_density_plot <- function(samples, images, density_grid = NULL) {
  density_layer <- if (is.null(density_grid)) {
    # Match the original export: fixed opacity and default 100x100 KDE grid.
    stat_density_2d(
      aes(fill = after_stat(density)),
      geom = "raster", contour = FALSE, alpha = 0.75
    )
  } else {
    # Supplemental layouts reuse the exact reference KDE, including cells
    # in the buffer. Re-estimating per image changes the automatic grid bounds.
    geom_raster(
      data = density_grid, aes(x = x, y = y, fill = density),
      inherit.aes = FALSE, alpha = 0.75
    )
  }
  ggplot(samples, aes(x = GazePointXMean_norm, y = GazePointYMean)) +
    geom_image_rect(
      data = images,
      aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, image = image)
    ) +
    density_layer +
    aoi_box_layer +
    scale_fill_viridis_c(option = "magma") +
    coord_fixed() +
    labs(
      title = "Density of gaze within the Target AOI box",
      subtitle = "Collapsed across Condition; distractor looks excluded; dashed box = gaze AOI hit-box; x-coords shifted (not mirrored) so Target is always on the left",
      x = "Gaze X, normalized so Target = left (screen px)",
      y = "Gaze Y (screen px)",
      fill = "Density"
    ) +
    theme_minimal(base_size = 9) +
    theme(legend.position = "bottom", strip.text = element_text(size = 7))
}

p_density <- make_density_plot(gaze_target, stim_images) +
  facet_grid(TargetImage ~ .)

ggsave(file.path(out_dir, "gaze_density_by_target.png"),
       p_density, width = 6, height = 18, dpi = 300, limitsize = FALSE)

## Build the supplemental layouts from the same data-space bounds and
## density/color scale as the reference-sized plot. In particular, do not
## resize the 500px image to fill its 620px AOI or crop the AOI buffer off.
reference_build <- ggplot_build(p_density)
reference_x_range <- reference_build$layout$panel_params[[1]]$x.range
reference_y_range <- reference_build$layout$panel_params[[1]]$y.range
reference_fill_limits <- range(reference_build$data[[2]]$density)
reference_density <- reference_build$data[[2]] %>%
  mutate(TargetImage = reference_build$layout$layout$TargetImage[
    match(PANEL, reference_build$layout$layout$PANEL)
  ]) %>%
  select(x, y, density, TargetImage)
inspection_coord <- coord_fixed(
  xlim = reference_x_range, ylim = reference_y_range, expand = FALSE
)
inspection_fill <- scale_fill_viridis_c(
  option = "magma", limits = reference_fill_limits,
  labels = scales::label_scientific(digits = 1),
  guide = guide_colorbar(barwidth = grid::unit(5, "cm"))
)

ggsave(file.path(out_dir, "gaze_density_overview.png"),
       make_density_plot(gaze_target, stim_images, reference_density) +
         facet_wrap(~ TargetImage, ncol = 4) +
         inspection_coord + inspection_fill +
         labs(subtitle = "Original rendering; Related + Unrelated pooled; 3550-5050 ms"),
       width = 16, height = 9, dpi = 200)

per_target_dir <- file.path(out_dir, "by_target")
dir.create(per_target_dir, showWarnings = FALSE)
for (target in as.character(stim_images$TargetImage)) {
  target_samples <- gaze_target %>% filter(TargetImage == target)
  target_images <- stim_images %>% filter(TargetImage == target)
  target_density <- reference_density %>% filter(TargetImage == target)
  p_target <- make_density_plot(target_samples, target_images, target_density) +
    inspection_coord + inspection_fill +
    labs(
      title = paste0(target, " - ", nrow(target_samples), " target gaze samples"),
      subtitle = "Original rendering; Related + Unrelated pooled; 3550-5050 ms"
    )
  ggsave(file.path(per_target_dir, paste0(target, "_heatmap.png")),
         p_target, width = 8, height = 8, dpi = 200)
}

## ---- 7. (Optional) numeric summary: mean gaze location per cell ------
## Per-subject-averaged summary, useful for a follow-up stats model, e.g.
## lmer(GazePointXMean ~ Condition * AOIType + (1|subjCode))
summary_by_subj <- gaze_clean %>%
  group_by(subjCode, TargetImage, Condition, AOIType) %>%
  summarise(
    mean_x_norm = mean(GazePointXMean_norm),  # normalized so Target = left
    mean_y      = mean(GazePointYMean),
    n           = n(),
    .groups = "drop"
  )

write_csv(summary_by_subj, file.path(out_dir, "gaze_location_summary_by_subject.csv"))

writeLines(capture.output(sessionInfo()), file.path(out_dir, "session_info.txt"))

cat("\nDone. Plots and summary written to:", normalizePath(out_dir), "\n")
