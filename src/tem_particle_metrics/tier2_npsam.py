"""Tier-2 segmentation via NP-SAM (Segment Anything, tuned for nanoparticle
EM). Targets the dense-aggregate images where tier-1's watershed cannot
separate heavily overlapping particles.

Heavy deps (torch, npsam, SAM weights) are imported lazily so importing the
package never requires them — tier-1 keeps running without torch. Install
into an isolated env (see requirements-tier2.txt); NP-SAM needs Python <3.13.

The backend emits the same `SegmentationResult` as tier-1, so analysis.py,
qc.py, and pipeline.run_image are reused unchanged.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from skimage import measure, morphology

from .segmentation import SegmentationResult


def masks_to_result(
    masks: list[np.ndarray],
    shape: tuple[int, int],
    *,
    gray: np.ndarray | None = None,
    exclude: np.ndarray | None = None,
    touch_gap: int = 2,
    min_area: int = 20,
    min_contrast: float = 0.0,
) -> SegmentationResult:
    """Convert a list of boolean instance masks into a SegmentationResult.

    Drops (a) instances mostly inside `exclude` (the scale bar) and (b) when
    `gray` and `min_contrast` are given, "ghost" masks — background patches
    FastSAM occasionally segments. A real particle is dark; a ghost sits on
    the light background, so we drop any mask whose interior is not at least
    `min_contrast` darker (in [0,1]) than the background (median of uncovered
    pixels). Overlaps are resolved by painting larger instances first so small
    particles are not swallowed; touching groups are recovered by closing gaps.
    """
    cand = [np.asarray(m, dtype=bool) for m in masks if int(np.asarray(m).sum()) >= min_area]
    if exclude is not None:
        cand = [m for m in cand if (m & exclude).sum() <= 0.5 * m.sum()]

    if gray is not None and min_contrast > 0 and cand:
        covered = np.zeros(shape, dtype=bool)
        for m in cand:
            covered |= m
        if exclude is not None:
            covered |= exclude
        bg = float(np.median(gray[~covered])) if (~covered).any() else float(np.median(gray))
        cand = [m for m in cand if bg - float(np.median(gray[m])) >= min_contrast]

    labels = np.zeros(shape, dtype=np.int32)
    for next_id, m in enumerate(sorted(cand, key=lambda m: int(m.sum()), reverse=True), 1):
        labels[m] = next_id

    touching = _touching_groups(labels, gap=touch_gap)
    return SegmentationResult(labels=labels, touching_group=touching)


def _touching_groups(labels: np.ndarray, *, gap: int) -> dict[int, int | None]:
    """A touching group is a set of >1 instances that fall in one blob after
    closing gaps of `gap` px (mirrors tier-1's contact-group semantics)."""
    fg = labels > 0
    closed = morphology.closing(fg, morphology.disk(gap))
    comp = measure.label(closed)

    comp_to_labels: dict[int, set[int]] = defaultdict(set)
    for lab in range(1, int(labels.max()) + 1):
        m = labels == lab
        if not m.any():
            continue
        c = int(np.bincount(comp[m]).argmax())
        comp_to_labels[c].add(lab)

    group: dict[int, int | None] = {}
    for c, labs in comp_to_labels.items():
        gid = c if len(labs) > 1 else None
        for lab in labs:
            group[lab] = gid
    return group


def segment_npsam(
    gray: np.ndarray,
    *,
    exclude: np.ndarray | None = None,
    device: str = "auto",
    sam_model: str = "auto",
    pps: int = 64,
    invert: bool = False,
    min_mask_region_area: int = 100,
    min_contrast: float = 0.15,
    crop_and_enlarge: bool = False,
    verbose: bool = False,
) -> SegmentationResult:
    """Run NP-SAM on a grayscale image and adapt to a SegmentationResult.

    Lazily imports npsam/torch. `device='auto'` uses MPS on Apple Silicon.
    `invert` matters for TEM: NP-SAM's characterization assumes bright
    objects; our particles are dark, so try invert=True if masks look wrong.
    `min_contrast` drops "ghost" masks on the light background (real particles
    are dark); 0.15 sits in the empty gap between ghosts (~0) and particles
    (≥0.3 darker than background). Set 0 to disable.

    `crop_and_enlarge` ("tiling") segments enlarged sub-crops instead of the
    whole downscaled frame — the recall lever for DENSE fields, where the
    backend (FastSAM here — `pps` is a no-op for it) otherwise smears crowded
    particles together. ~4–5× slower; safe on sparse frames (no over-count).
    See docs/batch_design.md; the batch engine turns it on for tier-2.
    """
    masks = _run_npsam(
        gray, exclude=exclude, device=device, sam_model=sam_model, pps=pps,
        invert=invert, min_mask_region_area=min_mask_region_area,
        crop_and_enlarge=crop_and_enlarge, verbose=verbose,
    )
    return masks_to_result(
        masks, gray.shape, gray=gray, exclude=exclude, min_contrast=min_contrast,
    )


def _run_npsam(
    gray, *, exclude=None, device="auto", sam_model="auto", pps=64,
    invert=False, min_mask_region_area=100, crop_and_enlarge=False, verbose=False,
) -> list[np.ndarray]:
    """Invoke NP-SAM's automatic mask generator and return instance masks.

    Writes the (scale-bar-blanked) frame to a temp PNG because NPSAMImage is
    file-based, runs SAM, then returns the filtered mask stack.
    """
    import os
    import tempfile

    from npsam import NPSAMImage
    from skimage import io as skio

    work = gray.copy()
    if exclude is not None:
        work[exclude] = float(np.median(gray))
    u8 = np.clip(work * 255.0, 0, 255).astype(np.uint8)

    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "frame.png")
        skio.imsave(fp, u8)
        img = NPSAMImage(fp)
        img.segment(
            device=device, SAM_model=sam_model, PPS=pps, invert=invert,
            min_mask_region_area=min_mask_region_area,
            crop_and_enlarge=crop_and_enlarge, verbose=verbose,
        )
        img.characterize(verbose=verbose)
        try:
            masks = img.get_filtered_masks()
        except Exception:
            masks = img.seg.data  # fall back to all masks post shape/edge filter

    return [np.asarray(m, dtype=bool) for m in masks]
