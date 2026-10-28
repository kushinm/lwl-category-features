# Shared repository guidance for Codex and Claude

This file is the canonical agent guidance. Keep instructions here so both
Codex and Claude use the same rules; `CLAUDE.md` imports this file.

## Study context

This repository analyzes a Looking-While-Listening eye-tracking study.
Infants saw a target image paired with a distractor while hearing speech
that named the target. There are eight target items, each paired with a
related (same-category) and an unrelated (different-category) distractor.
The current goal is to generate gaze-density heatmaps for each image
using samples collected when that image was the target.

The items are apple, banana, carrots, peas, bathtub, comb, potty, and
toothbrush. `gaze_location_lwl.R` is the existing analysis entry point.

## Data and Git

- Never commit CSV files containing study data, including raw inputs,
  participant metadata, and derived participant-level summaries.
- Keep CSV files local and ignored. The repository intentionally ignores
  all CSV files and compressed CSV files, including future exports.
- Do not bypass ignore rules with `git add -f` for study data. Check staged
  files before any commit and stop if study data would be included.
- Avoid copying participant identifiers or individual records into tracked
  documentation, fixtures, notebooks, or other file formats. Prefer
  aggregate summaries when reviewing data.
- Do not delete or overwrite original study data during analysis.

## Commits

- Use the user's configured Git identity as the sole author and committer.
- Do not add `Co-authored-by` trailers or credit Claude, Codex, or other
  agents in commit attribution. Commits should credit only the user.
- Keep commit messages brief, normally a single short subject line.

## Analysis and documentation

- Preserve the distinction between target and distractor looks and between
  related and unrelated conditions. State explicitly when pooling groups.
- Check coordinate systems, stimulus placement, AOI definitions, and time
  references before changing gaze alignment or interpreting heatmaps.
- When running an existing analysis, reproduce its current behavior first.
  Use the root `gaze_density_by_target.png` as the visual reference. Preserve
  its coordinate convention and the offset between the 500px stimulus and
  620px AOI, including buffer space, unless a change is explicitly requested
  or supported by verified acquisition records.
- Treat comments about acquisition geometry or timing as assumptions until
  checked against the experiment source or other acquisition records.
- Keep shared agent instructions in this file. Use ordinary Markdown and
  relative repository paths in other documentation so both agents can
  work with it. Keep `CLAUDE.md` as the entry point to these shared rules.
