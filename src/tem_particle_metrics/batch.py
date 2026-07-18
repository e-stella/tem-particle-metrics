"""Batch-mode plumbing: a run folder + a manifest that ties frames into samples.

The manifest (`<run>/manifest.csv`) is the spine of the headless-engine /
light-reviewer split (docs/batch_design.md). One row per frame; the engine
fills seg_* and status, the reviewer flips review_status, the aggregator reads
it to pool frames into per-sample tables. No torch here — importable in the
light review/aggregate context.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

IMAGE_SUFFIXES = {".dm3", ".dm4", ".png", ".tif", ".tiff"}
MANIFEST_NAME = "manifest.csv"

# one row per frame
MANIFEST_COLUMNS = [
    "frame_path", "stem", "sample_id", "tier", "tiling",
    "nm_per_px", "calibration_source",
    "n_particles", "mean_nm", "sd_nm",
    "seg_status", "review_status", "seg_time_s", "timestamp",
]

# a frame filename ending in a single letter or a run of digits (scan_a,
# img_c, frame_10) is one frame OF a sample; strip that suffix to get the
# sample id. Anything else is its own sample.
_FRAME_SUFFIX = re.compile(r"_(?:[A-Za-z]|\d+)$")


def sample_id_from_filename(stem: str) -> str:
    """Group key for a frame: strip a trailing `_a`/`_c`/`_10`… suffix.

    `scan_a` → `scan`, `img_c` → `img`, `run_23_e` → `run_23`.
    A stem with no such suffix (`overview_map`) is its own sample. The
    result is written into the manifest and is hand-editable when a filename
    breaks the convention.
    """
    grouped = _FRAME_SUFFIX.sub("", stem)
    return grouped or stem     # never return empty (e.g. stem == "a")


def list_images(folder: str | Path) -> list[Path]:
    """Supported image files in a folder, sorted by name."""
    folder = Path(folder)
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


# --- manifest IO ---------------------------------------------------------
def manifest_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / MANIFEST_NAME


def build_manifest(image_paths, *, tier: int, tiling: bool) -> pd.DataFrame:
    """Fresh manifest (all frames pending) for a list of image paths."""
    rows = []
    for p in image_paths:
        p = Path(p)
        rows.append({
            "frame_path": str(p),
            "stem": p.stem,
            "sample_id": sample_id_from_filename(p.stem),
            "tier": tier,
            "tiling": bool(tiling),
            "nm_per_px": pd.NA, "calibration_source": pd.NA,
            "n_particles": pd.NA, "mean_nm": pd.NA, "sd_nm": pd.NA,
            "seg_status": "pending", "review_status": "unreviewed",
            "seg_time_s": pd.NA, "timestamp": pd.NA,
        })
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def load_manifest(run_dir: str | Path) -> pd.DataFrame:
    return pd.read_csv(manifest_path(run_dir))


def save_manifest(df: pd.DataFrame, run_dir: str | Path) -> Path:
    path = manifest_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


# --- per-frame artifact paths (mirror pipeline.run_image, out_tag="") ----
def particles_csv(run_dir: str | Path, stem: str) -> Path:
    return Path(run_dir) / f"{stem}_particles.csv"


def labels_npy(run_dir: str | Path, stem: str) -> Path:
    return Path(run_dir) / f"{stem}_labels.npy"


def overlay_png(run_dir: str | Path, stem: str) -> Path:
    return Path(run_dir) / f"{stem}_overlay.png"


def reviewed_csv(run_dir: str | Path, stem: str) -> Path:
    return Path(run_dir) / f"{stem}_reviewed_particles.csv"


def reviewed_or_raw(run_dir: str | Path, stem: str) -> tuple[Path | None, bool]:
    """Per-frame table for a consumer: (path, is_reviewed).

    Prefers `<stem>_reviewed_particles.csv` (human decisions) over the raw
    `<stem>_particles.csv`; returns (None, False) if neither exists. Callers
    exclude `manual_status == dropped` when is_reviewed.
    """
    rev = reviewed_csv(run_dir, stem)
    if rev.exists():
        return rev, True
    raw = particles_csv(run_dir, stem)
    if raw.exists():
        return raw, False
    return None, False
