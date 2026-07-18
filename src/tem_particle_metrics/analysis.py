"""Turn a label image into the per-particle output-contract table.

Columns (one row per detected particle), in contract order:
    sample_id, frame_id, particle_id, diameter_nm, area_nm2, perimeter_nm,
    aspect_ratio, circularity, shape_class, touching_group_id,
    nm_per_px, calibration_source, source_file
Summary stats are DERIVED from this table, never stored as primary output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from skimage import measure

from .calibration import Calibration

CONTRACT_COLUMNS = [
    "sample_id", "frame_id", "particle_id",
    "diameter_nm", "area_nm2", "perimeter_nm",
    "aspect_ratio", "circularity", "shape_class",
    "touching_group_id", "nm_per_px", "calibration_source", "source_file",
]

# Tier-2 (NP-SAM/FastSAM) masks run slightly generous; multiplying linear
# dimensions by this factor removes the size-weighted mean oversizing. Derived
# empirically against hand-measured ground truth. Opt-in — see
# apply_size_calibration.
SIZE_CALIBRATION_FACTOR = 0.977


def apply_size_calibration(df: pd.DataFrame, factor: float = SIZE_CALIBRATION_FACTOR) -> pd.DataFrame:
    """Return a copy with linear sizes scaled by `factor` (area by factor²).

    Corrects the tier-2 mask-boundary oversizing. Dimensionless columns
    (aspect_ratio, circularity) are geometrically invariant and untouched.
    Off by default in the pipeline; apply explicitly to a tier-2 table.
    """
    out = df.copy()
    for col in ("diameter_nm", "perimeter_nm"):
        if col in out:
            out[col] = out[col] * factor
    if "area_nm2" in out:
        out["area_nm2"] = out["area_nm2"] * factor**2
    return out


def classify_shape(circularity: float, aspect_ratio: float, solidity: float) -> str:
    """Rule-based class from mask geometry: sphere | rod | triangle | irregular.

    Crude by design (scalar thresholds); citrate AuNP are overwhelmingly
    spheres, so this mainly flags the exceptions the human eye skips. Assumes
    upstream segmentation has already split touching pairs, so an elongated
    *single, convex* object is a genuine rod rather than an unsplit dimer.
    Tune against QC overlays as more shapes appear.
    """
    if not np.isfinite(circularity):
        circularity = 0.0
    if not np.isfinite(aspect_ratio):
        aspect_ratio = 1.0
    if aspect_ratio >= 1.7 and solidity >= 0.88:
        return "rod"
    if circularity >= 0.80 and aspect_ratio < 1.3:
        return "sphere"
    if solidity >= 0.92 and 0.55 <= circularity < 0.80 and aspect_ratio < 1.5:
        return "triangle"
    return "irregular"


def build_table(
    seg,
    cal: Calibration,
    *,
    sample_id: str,
    source_file: str,
    frame_id: int = 0,
    min_diameter_nm: float = 4.0,
    max_diameter_nm: float | None = None,
) -> tuple[pd.DataFrame, dict, np.ndarray]:
    """Build the per-particle DataFrame. Returns (df, filter_report, id_image).

    `filter_report` records how many regions were dropped and why, so the
    pipeline can log it — no silent truncation. `id_image` is a label image
    whose pixel value is `particle_id + 1` (0 = background), so a click in
    the QC reviewer maps straight to a table row.
    """
    nm = cal.nm_per_px
    props = measure.regionprops(seg.labels)

    id_image = np.zeros(seg.labels.shape, dtype=np.int32)
    rows = []
    dropped_small = dropped_large = 0
    pid = 0
    for p in props:
        area_px = float(p.area)
        d_px = 2.0 * np.sqrt(area_px / np.pi)
        diameter_nm = d_px * nm
        if diameter_nm < min_diameter_nm:
            dropped_small += 1
            continue
        if max_diameter_nm is not None and diameter_nm > max_diameter_nm:
            dropped_large += 1
            continue

        perim_px = float(p.perimeter)
        circ = 4.0 * np.pi * area_px / perim_px**2 if perim_px > 0 else np.nan
        major = float(p.axis_major_length)
        minor = float(p.axis_minor_length)
        ar = major / minor if minor > 0 else np.nan
        solidity = float(p.solidity)
        shape = classify_shape(min(circ, 1.0) if np.isfinite(circ) else np.nan, ar, solidity)

        rows.append({
            "sample_id": sample_id,
            "frame_id": frame_id,
            "particle_id": pid,
            "diameter_nm": diameter_nm,
            "area_nm2": area_px * nm * nm,
            "perimeter_nm": perim_px * nm,
            "aspect_ratio": ar,
            "circularity": circ,
            "shape_class": shape,
            "touching_group_id": seg.touching_group.get(p.label),
            "nm_per_px": nm,
            "calibration_source": cal.source,
            "source_file": source_file,
        })
        id_image[seg.labels == p.label] = pid + 1
        pid += 1

    df = pd.DataFrame(rows, columns=CONTRACT_COLUMNS)
    report = {
        "regions_total": len(props),
        "kept": len(df),
        "dropped_small": dropped_small,
        "dropped_large": dropped_large,
    }
    return df, report, id_image
