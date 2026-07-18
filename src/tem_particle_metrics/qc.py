"""QC overlays. Mandatory for every run: trust is the product.

Draws segmentation boundaries over the original micrograph so a human can
eyeball what the machine measured. Solo particles and touching-group
particles get different colours.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from skimage import color, exposure, segmentation  # noqa: E402


def save_overlay(
    gray: np.ndarray,
    seg,
    out_path: str | Path,
    *,
    title: str | None = None,
) -> None:
    """Write a boundary overlay PNG. Touching-group particles in red, solo
    particles in green."""
    disp = color.gray2rgb(exposure.rescale_intensity(gray, out_range=(0.0, 1.0)))

    solo = np.zeros_like(seg.labels)
    touch = np.zeros_like(seg.labels)
    for lab, gid in seg.touching_group.items():
        (touch if gid is not None else solo)[seg.labels == lab] = lab

    disp = segmentation.mark_boundaries(disp, solo, color=(0.1, 0.9, 0.1), mode="thick")
    if touch.max() > 0:
        disp = segmentation.mark_boundaries(disp, touch, color=(1.0, 0.0, 0.0), mode="thick")

    fig, ax = plt.subplots(figsize=(11, 11), dpi=120)
    ax.imshow(disp)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=13)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
