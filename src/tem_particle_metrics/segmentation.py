"""Tier-1 classical segmentation, rod-aware.

AuNP TEM is near-ideal for classical methods: dark compact particles on a
smooth light background. Pipeline:
  invert -> white-tophat background flattening -> Otsu -> fill/despeckle
  -> distance-transform watershed, deliberately OVER-seeded
  -> merge adjacent regions back where there is no real neck.

The merge step is what distinguishes a rod from a touching pair. A genuine
rod has a flat distance ridge, so a watershed cut across it sits at nearly
the full half-width (neck ratio ~1) and is merged away. A touching pair has
a pinched neck, so the cut sits at a distance well below the particle radii
(neck ratio < 1) and is kept. This fixes both failure modes at once: rods
shattered into fake spheres, and dimers merged into fake rods.

Everything here is pixel space; conversion to nm happens in analysis.py.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi
from skimage import feature, filters, measure, morphology, segmentation


@dataclass
class SegmentationResult:
    labels: np.ndarray  # int label image, 0 = background
    touching_group: dict[int, int | None] = field(default_factory=dict)
    # final particle label -> group id (shared by particles from the same
    # original mask component when it yielded >1 particle), else None


def _remove_small(bw: np.ndarray, min_area: int) -> np.ndarray:
    """Drop connected components with area < min_area (version-stable)."""
    lbl = measure.label(bw)
    counts = np.bincount(lbl.ravel())
    keep = counts >= min_area
    keep[0] = False
    return keep[lbl]


def _merge_by_neck(labels: np.ndarray, dist: np.ndarray, keep_ratio: float) -> np.ndarray:
    """Merge adjacent watershed regions separated by a shallow (fake) neck.

    For each adjacent label pair, the neck half-thickness is the max distance
    transform value along their shared interface. Divided by the smaller of
    the two regions' peak distances it gives a ratio near 1 for an artificial
    cut (rod or over-split sphere) and well below 1 for a real neck (contact
    between two particles). Pairs with ratio >= keep_ratio are merged.
    """
    n = int(labels.max())
    if n <= 1:
        return labels

    peak = np.zeros(n + 1)
    np.maximum.at(peak, labels.ravel(), dist.ravel())

    necks: dict[tuple[int, int], float] = {}
    for axis in (0, 1):
        a = labels
        b = np.roll(labels, -1, axis=axis)
        da, db = dist, np.roll(dist, -1, axis=axis)
        m = (a > 0) & (b > 0) & (a != b)
        if axis == 0:
            m[-1, :] = False
        else:
            m[:, -1] = False
        aa, bb = a[m], b[m]
        dd = np.maximum(da[m], db[m])
        for x, y, dv in zip(aa.tolist(), bb.tolist(), dd.tolist()):
            key = (x, y) if x < y else (y, x)
            if dv > necks.get(key, 0.0):
                necks[key] = dv

    parent = list(range(n + 1))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for (a, b), neck in necks.items():
        denom = min(peak[a], peak[b])
        if denom > 0 and neck / denom >= keep_ratio:
            parent[find(a)] = find(b)

    out = np.zeros_like(labels)
    remap: dict[int, int] = {}
    nxt = 1
    for lab in range(1, n + 1):
        root = find(lab)
        if root not in remap:
            remap[root] = nxt
            nxt += 1
        out[labels == lab] = remap[root]
    return out


def segment(
    gray: np.ndarray,
    *,
    exclude: np.ndarray | None = None,
    bg_radius: int = 40,
    min_object_area: int = 30,
    peak_min_distance: int = 7,
    neck_keep_ratio: float = 0.85,
) -> SegmentationResult:
    """Segment dark particles; split touching ones but keep rods whole.

    `exclude` blanks a region (e.g. the scale bar). `bg_radius` sizes the
    top-hat element (must exceed particle radius). `peak_min_distance`
    over-seeds the watershed; `neck_keep_ratio` sets how flat a neck must be
    to merge (higher = more conservative merging).
    """
    work = gray.copy()
    if exclude is not None:
        work[exclude] = float(np.median(gray))

    inv = 1.0 - work
    # Decomposed disk: same top-hat, ~50x faster. A radius-r opening is built
    # from a sequence of tiny separable elements, so cost is ~N*r not N*r^2.
    flat = morphology.white_tophat(inv, morphology.disk(bg_radius, decomposition="sequence"))

    thr = filters.threshold_otsu(flat)
    bw = flat > thr
    bw = ndi.binary_fill_holes(bw)
    bw = _remove_small(bw, min_object_area)

    pre = measure.label(bw)  # original contact groups, before splitting

    dist = ndi.distance_transform_edt(bw)
    coords = feature.peak_local_max(dist, min_distance=peak_min_distance, labels=bw)
    peaks = np.zeros(dist.shape, dtype=bool)
    peaks[tuple(coords.T)] = True
    markers = measure.label(peaks)
    ws = segmentation.watershed(-dist, markers, mask=bw)
    labels = _merge_by_neck(ws, dist, neck_keep_ratio)

    # Contact group = original mask component that ended as >1 final particle.
    comp_to_finals: dict[int, set[int]] = defaultdict(set)
    for lab in range(1, int(labels.max()) + 1):
        m = labels == lab
        if not m.any():
            continue
        comp = int(np.bincount(pre[m]).argmax())
        comp_to_finals[comp].add(lab)

    touching_group: dict[int, int | None] = {}
    for comp, finals in comp_to_finals.items():
        gid = comp if len(finals) > 1 else None
        for lab in finals:
            touching_group[lab] = gid

    return SegmentationResult(labels=labels, touching_group=touching_group)
