# tem-particle-metrics

Automated size/shape segmentation of nanoparticle TEM micrographs into
per-particle measurement tables (diameter, shape class, touching groups),
replacing hand-measured histograms. Validated against hand-measured
ground truth; output feeds the physics forward model in
[aunp_speciation](https://github.com/e-stella/aunp_speciation), which
predicts UV-Vis spectra from these tables with zero fitted parameters.

## Demo

[![Watch the demo](https://img.youtube.com/vi/t74QXefrVq4/hqdefault.jpg)](https://youtu.be/t74QXefrVq4)

Raw micrographs and measurements are lab data and are never committed to this
repository.

## Quick start (fresh machine)

The desktop app — **segment → review → export** in one napari window — runs from
a single conda environment. From a clean Mac or Windows machine:

1. **Install Miniforge** (conda): https://github.com/conda-forge/miniforge
   (Mac: the `MacOSX-arm64` installer; Windows: the `Windows-x86_64` `.exe`).
2. **Get the code** — `git clone https://github.com/e-stella/tem-particle-metrics`
   (or download the ZIP), then `cd tem-particle-metrics`.
3. **Build the environment** (downloads torch + napari — a few minutes, a few GB):
   - **macOS / Linux:** `bash scripts/make_app_env.sh`
   - **Windows** (from the *Miniforge Prompt*): `conda env create -f environment.yml`
4. **Run the app:**
   ```bash
   conda activate tem-app
   python gui/napari_app.py
   ```

In the app: **Pick image folder** → choose a tier (Tier 1 fast / Tier 2 for
dense clusters) → **Segment** → click particles to drop → **Export samples**.

Notes:
- **Your images aren't in this repo** (lab data is private). Point the app at
  your own folder of `.dm3/.dm4/.png/.tif` files.
- The **first Tier-2 run downloads the FastSAM model** (~138 MB);
  `make_app_env.sh` pre-fetches it, otherwise answer `y` when prompted.
- Use `conda activate tem-app` — the exact interpreter paths in `gui/README.md`
  are developer-machine specific. More detail: `gui/README.md`.
