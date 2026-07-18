"""The tier-2 mask->SegmentationResult adapter must build a clean label image,
recover touching groups, drop scale-bar instances, and resolve overlaps
without swallowing small particles. Runs without npsam/torch.
"""
import sys
from pathlib import Path

import numpy as np
from skimage import draw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tem_particle_metrics.tier2_npsam import masks_to_result  # noqa: E402


def _disk_mask(cy, cx, r, shape=(200, 200)):
    m = np.zeros(shape, bool)
    rr, cc = draw.disk((cy, cx), r, shape=shape)
    m[rr, cc] = True
    return m


def test_adapter_labels_groups_and_exclusion():
    shape = (200, 200)
    a = _disk_mask(60, 60, 15)          # isolated
    b = _disk_mask(120, 95, 15)         # touching pair with c
    c = _disk_mask(120, 120, 15)
    bar = np.zeros(shape, bool)
    bar[185:195, 120:190] = True        # a "scale bar" instance to drop
    exclude = bar.copy()

    res = masks_to_result([a, b, c, bar], shape, exclude=exclude, touch_gap=3)

    assert int(res.labels.max()) == 3, "bar instance dropped, 3 particles kept"
    grouped = [lab for lab, g in res.touching_group.items() if g is not None]
    assert len(grouped) == 2, "the touching pair forms one group of 2"
    lone = [lab for lab, g in res.touching_group.items() if g is None]
    assert len(lone) == 1, "the isolated particle has no group"


def test_ghost_masks_dropped_by_contrast():
    shape = (200, 200)
    gray = np.full(shape, 0.85)          # light background
    real1, real2 = _disk_mask(60, 60, 15), _disk_mask(60, 140, 15)
    ghost = _disk_mask(140, 100, 15)     # a mask sitting on bare background
    gray[real1] = 0.2                    # real particles are dark
    gray[real2] = 0.2
    # ghost interior stays at background brightness (0.85) -> no contrast

    res = masks_to_result(
        [real1, real2, ghost], shape, gray=gray, min_contrast=0.15,
    )
    assert int(res.labels.max()) == 2, "the background ghost mask is dropped"
    # without the filter, all three survive
    res_off = masks_to_result([real1, real2, ghost], shape)
    assert int(res_off.labels.max()) == 3


def test_adapter_small_particle_survives_overlap():
    shape = (200, 200)
    big = _disk_mask(100, 100, 40)
    small = _disk_mask(100, 100, 8)  # fully inside big; must not be erased
    res = masks_to_result([big, small], shape, touch_gap=2, min_area=20)
    # small painted on top -> both labels present
    assert int(res.labels.max()) == 2
    assert (res.labels == 2).sum() > 0, "small overlapping particle survives"


if __name__ == "__main__":
    test_adapter_labels_groups_and_exclusion()
    test_adapter_small_particle_survives_overlap()
    print("ok")
