"""Engine glue for the napari GUI: list a folder's images and segment one,
handing back everything the reviewer needs (image, id-image, per-particle
table) as a `SegmentResult`. No napari import here, so it stays unit-testable.

Tier-1 runs in-process (fast, no torch). Tier-2 (NP-SAM/torch) runs in the
isolated `tem-tier2` env via a subprocess to `scripts/segment_image_cli.py` —
keeping torch out of the GUI process — and the result is read back.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .image_io import load_dm3, load_grayscale
from .pipeline import run_image

ROOT = Path(__file__).resolve().parents[2]
IMAGE_SUFFIXES = {".dm3", ".dm4", ".png", ".tif", ".tiff"}
_METADATA_CAL = {".dm3", ".dm4"}

# Python interpreter of the isolated tier-2 (NP-SAM) env. Override with the
# TEM_TIER2_PYTHON env var if it lives elsewhere.
TIER2_PYTHON = os.environ.get(
    "TEM_TIER2_PYTHON",
    str(Path.home() / "miniforge3" / "envs" / "tem-tier2" / "bin" / "python"),
)
_CLI = ROOT / "scripts" / "segment_image_cli.py"
_ENGINE_CLI = ROOT / "scripts" / "segment_folder.py"


def _tier2_python() -> str:
    """Python to run tier-2 with: THIS interpreter if it already has npsam (the
    consolidated tem-app env), else the isolated tem-tier2 env. So the same app
    runs tier-2 in-place after env consolidation, or shells out to tem-tier2
    before it — no code change either way. find_spec doesn't import torch."""
    import importlib.util
    if importlib.util.find_spec("npsam") is not None:
        return sys.executable
    return TIER2_PYTHON


def default_run_dir(folder: str | Path, tier: int) -> Path:
    """Per-(folder, tier) run folder under runs/ (gitignored)."""
    base = Path(folder).name or "run"
    return ROOT / "runs" / f"{base}_tier{tier}"


def batch_engine_command(
    folder: str | Path,
    run_dir: str | Path,
    tier: int | str,
    *,
    tiling: bool | None = None,
    nm_per_px: float | None = None,
    bar_length_nm: float | None = None,
) -> list[str]:
    """Full subprocess command to run the headless engine over a folder.

    Tier-2 uses the isolated tem-tier2 python (TIER2_PYTHON); tier-1 uses the
    current interpreter (the light GUI env already has the tier-1 stack). This
    lets the single-window app fire the engine as ONE background process
    (model loaded once) rather than per image.
    """
    py = _tier2_python() if tier in (2, "auto") else sys.executable
    cmd = [py, str(_ENGINE_CLI), str(folder), "--out-dir", str(run_dir), "--tier", str(tier)]
    if tier in (2, "auto") and tiling is not None:
        cmd.append("--tiling" if tiling else "--no-tiling")
    if nm_per_px is not None:
        cmd += ["--nm-per-px", str(nm_per_px)]
    elif bar_length_nm is not None:
        cmd += ["--bar-length-nm", str(bar_length_nm)]
    return cmd


@dataclass
class SegmentResult:
    gray: np.ndarray
    id_image: np.ndarray            # pixel = particle_id + 1
    table: pd.DataFrame
    tier: int
    nm_per_px: float
    calibration_source: str


def list_images(folder: str | Path) -> list[Path]:
    """Supported image files in a folder, sorted by name."""
    folder = Path(folder)
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def load_gray(path: str | Path) -> np.ndarray:
    """The displayed image, [0,1] (dm3 via metadata reader, else generic)."""
    path = Path(path)
    return load_dm3(path)[0] if path.suffix.lower() in _METADATA_CAL else load_grayscale(path)


def _calibration_needed(path: Path) -> bool:
    return path.suffix.lower() not in _METADATA_CAL


def segment_image(
    path: str | Path,
    *,
    tier: int = 1,
    nm_per_px: float | None = None,
    bar_length_nm: float | None = None,
) -> SegmentResult:
    """Segment one image. dm3/dm4 self-calibrate; other formats need a scale."""
    path = Path(path)
    if _calibration_needed(path) and nm_per_px is None and bar_length_nm is None:
        raise ValueError(f"{path.name}: non-dm3 image needs nm_per_px or bar_length_nm")

    if tier == 1:
        kw: dict = {}
        if _calibration_needed(path):
            kw = {"nm_per_px": nm_per_px} if nm_per_px is not None else {"bar_length_nm": bar_length_nm}
        res = run_image(path, sample_id=path.stem, **kw)
        return SegmentResult(res.gray, res.id_image, res.table, 1,
                             res.calibration.nm_per_px, res.calibration.source)
    if tier == 2:
        return _segment_tier2(path, nm_per_px, bar_length_nm)
    raise ValueError(f"unknown tier {tier}")


def _segment_tier2(path: Path, nm_per_px, bar_length_nm) -> SegmentResult:
    py = _tier2_python()
    if not Path(py).exists():
        raise FileNotFoundError(
            f"tier-2 env python not found: {py}\n"
            "install the tem-tier2 env or set TEM_TIER2_PYTHON."
        )
    with tempfile.TemporaryDirectory() as td:
        cmd = [py, str(_CLI), str(path), "--tier", "2", "--out-dir", td]
        if _calibration_needed(path):
            cmd += (["--nm-per-px", str(nm_per_px)] if nm_per_px is not None
                    else ["--bar-length-nm", str(bar_length_nm)])
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"tier-2 segmentation failed:\n{proc.stderr[-1000:]}")
        stem = path.stem
        id_image = np.load(Path(td) / f"{stem}_labels.npy").astype(np.int32)
        table = pd.read_csv(Path(td) / f"{stem}_particles.csv")

    gray = load_gray(path)
    nmpx = float(table["nm_per_px"].iloc[0]) if len(table) else float("nan")
    src = str(table["calibration_source"].iloc[0]) if len(table) else "metadata"
    return SegmentResult(gray, id_image, table, 2, nmpx, src)
