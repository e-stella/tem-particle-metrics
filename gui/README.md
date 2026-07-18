# napari desktop app

GUIs over the segmentation/review engine, on the `feature/napari-gui` branch.
(Env setup in `../requirements-gui.txt`.)

## Single-window app (recommended) — `napari_app.py`

The whole workflow in one window, no CLI: **segment → review → export.**

```bash
conda run -n tem-gui python gui/napari_app.py
# or point it at a folder to pre-fill it:
conda run -n tem-gui python gui/napari_app.py data/data_for_ml/tem_images/October_2024
```

- **Segment / Export** dock: pick a folder + tier (tiling toggle for tier-2), hit
  **Segment**. It runs the headless engine in the background — tier-2 shells out
  to the `tem-tier2` env (torch stays out of the GUI process), tier-1 runs with
  the current interpreter — writing a run folder under `runs/` (gitignored).
  Progress updates live from the manifest; when done, the run loads for review.
- **Review** dock: navigate by sample→frame, click a particle to drop/restore
  (outline view, tier colours), **Save review** per frame.
- **Export samples**: pools kept particles into `runs/<...>/samples/<id>_sample.csv`
  + `summary.csv`, and prints per-sample mean±sd (flagging unreviewed frames).

The three underlying stages are also runnable standalone (`scripts/segment_folder.py`,
`gui/napari_batch_review.py`, `scripts/aggregate_samples.py`) — same run folder.

## Consolidated env (lab distribution) — `tem-app`

For a single downloadable environment (no juggling `tem-gui`/`tem-tier2`), build
the consolidated `tem-app` env — one Python 3.11 env with tier-1 + tier-2 +
napari together (verified 2026-07-18):

```bash
bash scripts/make_app_env.sh            # conda env create + pre-fetch FastSAM weights
conda run -n tem-app python gui/napari_app.py
```

In this env tier-2 runs **in-process** (no subprocess) — the app detects npsam
and runs it in-place (`gui_engine._tier2_python`). The stack is uniform **PyQt6**
(npsam requires it; napari follows); the app selects it automatically
(`gui/_qtapi.py`), so no `QT_API` juggling. See `../environment.yml` and
`docs/batch_design.md` for the recipe and its sharp edges (PyQt6, numpy 1.26
pin, weight pre-fetch). Development still uses the split envs for faster iteration.

## Lower-level GUIs

`napari_review.py` — segment ONE image on demand (Phase-2, tier selector):

```bash
conda run -n tem-gui python gui/napari_review.py [folder]
```

`napari_batch_review.py` — PURE review of an existing run folder (no engine; the
light reviewer for a split-role / distributed workflow):

```bash
conda run -n tem-gui python gui/napari_batch_review.py runs/<run>
```

## What it does now

- **Open folder…** → lists `.dm3/.dm4/.png/.tif` images; pick one from the dropdown.
- Pick a **tier**: *Tier 1* (classical, in-process, ~1 s) or *Tier 2* (NP-SAM, for
  aggregated samples). The image is segmented and shown with a coloured mask
  overlay: **green = solo, red = touching**. (Increase the labels layer's
  **contour** control for outline-only.)
- **Click a particle to drop/restore it** (turns **yellow**). The docked histogram
  and the mean ± sd update live over kept particles.
- **Save review** → writes `outputs/gui/<stem>_reviewed_particles.csv`
  (`manual_status` column; drops excluded from stats).

Reuses the existing engine: `gui_engine.segment_image` (pipeline) and
`review.ReviewSession` (the toolkit-agnostic keep/drop logic).

## Tier-2 (NP-SAM)

Tier-2 needs torch, which can't live in the GUI env, so the GUI **shells out** to
the isolated `tem-tier2` env (`scripts/segment_image_cli.py`) and reads the result
back. It runs off the UI thread (the window stays responsive) but each image is a
fresh process (~20 s: torch import + model load + inference). Set
`TEM_TIER2_PYTHON` if that env's python isn't at the default path.

## Not yet (later phases)

- **Persistent tier-2 worker** to amortize the ~10 s model-load per image (keep one
  process alive instead of one-per-image).
- **Forward-spectrum panel** — predicted UV-Vis from the reviewed histogram,
  subprocess to the optics env.
- Sample-level aggregation across a folder's frames; restore-on-reopen; packaging.

## Notes

- napari 0.8 (`DirectLabelColormap`), PyQt5. Needs a desktop display — the
  interactive window can't run headless; the engine (`test_gui_engine.py`) is
  tested headless.
- This is a first draft of the interaction; expect to iterate on click/colour
  behaviour once you run it.
