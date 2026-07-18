"""Calibration: turn pixels into nanometres.

For the pptx-extracted PNGs the only calibration source is the burned-in
scale bar (a solid near-black horizontal rectangle, bottom-right, with a
"<value> nm" label under it). We detect the bar's pixel length and divide
the known physical length by it. Manual override is always available; dm4
metadata will be a third source later.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage import measure


@dataclass
class Calibration:
    """nm-per-pixel plus provenance. `source` matches the output-contract
    `calibration_source` field: 'scalebar' | 'manual' | 'metadata'."""

    nm_per_px: float
    source: str
    bar_length_px: float | None = None
    bar_length_nm: float | None = None
    bar_bbox: tuple[int, int, int, int] | None = None  # (minr, minc, maxr, maxc), full image


def detect_scale_bar(
    gray: np.ndarray,
    bar_length_nm: float,
    *,
    dark_thresh: float = 0.35,
    min_aspect: float = 4.0,
    min_width_frac: float = 0.10,
    min_solidity: float = 0.90,
    border_frac: float = 0.25,
) -> Calibration:
    """Detect a burned-in horizontal scale bar in any corner/edge.

    The bar is the widest dark component that is a solid horizontal rectangle
    (`min_aspect` wide-to-tall, `min_solidity` fill) spanning at least
    `min_width_frac` of the image width, with its centroid near a border
    (`border_frac`). Solidity is what separates a bar from an elongated
    aggregate chain (which is bumpy, so low solidity). Raises ValueError if
    none qualifies (fall back to manual nm/px).
    """
    h, w = gray.shape
    dark = gray < dark_thresh
    lbl = measure.label(dark)

    best_w = 0.0
    best_bbox: tuple[int, int, int, int] | None = None
    for r in measure.regionprops(lbl):
        minr, minc, maxr, maxc = r.bbox
        bw, bh = maxc - minc, maxr - minr
        if bh == 0 or bw / bh < min_aspect:
            continue
        if bw < min_width_frac * w or r.solidity < min_solidity:
            continue
        cy, cx = r.centroid
        near_border = (
            cy < border_frac * h or cy > (1 - border_frac) * h
            or cx < border_frac * w or cx > (1 - border_frac) * w
        )
        if not near_border or bw <= best_w:
            continue
        best_w = float(bw)
        best_bbox = (minr, minc, maxr, maxc)

    if best_bbox is None:
        raise ValueError("no scale bar found; pass nm_per_px manually")
    return Calibration(
        nm_per_px=bar_length_nm / best_w,
        source="scalebar",
        bar_length_px=best_w,
        bar_length_nm=float(bar_length_nm),
        bar_bbox=best_bbox,
    )


def manual_calibration(nm_per_px: float) -> Calibration:
    """Calibration from a user-supplied nm/px (e.g. read off Fiji)."""
    return Calibration(nm_per_px=float(nm_per_px), source="manual")


def metadata_calibration(nm_per_px: float) -> Calibration:
    """Calibration read from file metadata (dm3/dm4/emd)."""
    return Calibration(nm_per_px=float(nm_per_px), source="metadata")


def annotation_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Boolean mask covering the scale bar and the "<n> nm" text beneath it.

    Pads the bar bbox and extends a band downward for the label text, so it
    works for a bar in any corner (bottom or top). Side pad scales with bar
    width in case the text is wider than the bar.
    """
    minr, minc, maxr, maxc = bbox
    bh = maxr - minr
    side = max(25, int(0.15 * (maxc - minc)))
    text_drop = 8 * bh  # room for the label below the bar
    m = np.zeros(shape, dtype=bool)
    r0, r1 = max(0, minr - bh), min(shape[0], maxr + text_drop)
    c0, c1 = max(0, minc - side), min(shape[1], maxc + side)
    m[r0:r1, c0:c1] = True
    return m
