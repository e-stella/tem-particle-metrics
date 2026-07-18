"""Glue: micrograph -> per-particle table + QC overlay.

One entry point, `run_image`, wires the I/O, calibration, segmentation,
analysis, and QC layers together. Calibration is by scale bar unless an
explicit nm_per_px is given.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from .analysis import build_table
from .calibration import (
    Calibration, annotation_mask, detect_scale_bar, manual_calibration, metadata_calibration,
)
from .image_io import load_dm3, load_grayscale
from .qc import save_overlay
from .segmentation import SegmentationResult, segment


class Segmenter(Protocol):
    """A segmentation backend: grayscale image -> SegmentationResult. Tier-1
    `segment` and tier-2 `segment_npsam` both satisfy this, so the analysis,
    table, and QC layers are shared across tiers."""

    def __call__(self, gray, *, exclude=None, **kwargs) -> SegmentationResult: ...


@dataclass
class RunResult:
    table: pd.DataFrame
    calibration: Calibration
    segmentation: SegmentationResult
    filter_report: dict
    gray: np.ndarray | None = None      # the loaded image, [0,1] (for the GUI/reviewer)
    id_image: np.ndarray | None = None  # pixel = particle_id + 1 (for the GUI/reviewer)
    csv_path: Path | None = None
    overlay_path: Path | None = None
    labels_path: Path | None = None


def run_image(
    path: str | Path,
    *,
    sample_id: str,
    frame_id: int = 0,
    bar_length_nm: float | None = None,
    nm_per_px: float | None = None,
    out_dir: str | Path | None = None,
    out_tag: str = "",
    segmenter: Segmenter | None = None,
    seg_kwargs: dict | None = None,
    analysis_kwargs: dict | None = None,
) -> RunResult:
    """Run one micrograph through the pipeline.

    `segmenter` selects the tier (defaults to tier-1 classical `segment`).
    Provide either `nm_per_px` (manual) or `bar_length_nm` (scale-bar
    detection). If `out_dir` is set, writes `<stem>{out_tag}_particles.csv`
    and `<stem>{out_tag}_overlay.png` there (`out_tag` keeps tiers from
    overwriting each other).
    """
    path = Path(path)

    if path.suffix.lower() in (".dm3", ".dm4"):
        # native microscope image: calibration from metadata, no scale bar
        gray, meta_nm_per_px = load_dm3(path)
        cal = metadata_calibration(meta_nm_per_px)
        exclude = None
    else:
        gray = load_grayscale(path)
        if nm_per_px is not None:
            cal = manual_calibration(nm_per_px)
            exclude = None
        elif bar_length_nm is not None:
            cal = detect_scale_bar(gray, bar_length_nm)
            exclude = annotation_mask(gray.shape, cal.bar_bbox)
        else:
            raise ValueError("provide either nm_per_px or bar_length_nm (or a dm3/dm4 file)")

    seg_fn: Callable = segmenter or segment
    seg = seg_fn(gray, exclude=exclude, **(seg_kwargs or {}))
    df, report, id_image = build_table(
        seg, cal, sample_id=sample_id, frame_id=frame_id,
        source_file=path.name, **(analysis_kwargs or {}),
    )

    csv_path = overlay_path = labels_path = None
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{path.stem}{out_tag}_particles.csv"
        overlay_path = out_dir / f"{path.stem}{out_tag}_overlay.png"
        labels_path = out_dir / f"{path.stem}{out_tag}_labels.npy"
        df.to_csv(csv_path, index=False)
        np.save(labels_path, id_image.astype(np.uint16))  # for the QC reviewer
        title = (
            f"{sample_id}: n={len(df)}, "
            f"mean={df['diameter_nm'].mean():.2f} nm, "
            f"sd={df['diameter_nm'].std(ddof=1):.2f} nm"
        )
        save_overlay(gray, seg, overlay_path, title=title)

    return RunResult(
        table=df, calibration=cal, segmentation=seg,
        filter_report=report, gray=gray, id_image=id_image, csv_path=csv_path,
        overlay_path=overlay_path, labels_path=labels_path,
    )
