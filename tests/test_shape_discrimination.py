"""Synthetic test: the segmenter must keep a rod whole, split a touching
dimer, and leave isolated spheres intact. This pins the neck-merge behaviour
independent of messy real micrographs.
"""
import sys
from pathlib import Path

import numpy as np
from skimage import draw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tem_particle_metrics.analysis import build_table  # noqa: E402
from tem_particle_metrics.calibration import manual_calibration  # noqa: E402
from tem_particle_metrics.segmentation import segment  # noqa: E402


def _scene():
    """Dark objects on light background: 2 spheres, 1 dimer, 1 rod."""
    gray = np.ones((300, 300), dtype=float)
    obj = np.zeros((300, 300), dtype=bool)
    for cy, cx in [(50, 50), (50, 250)]:  # isolated spheres
        rr, cc = draw.disk((cy, cx), 18, shape=obj.shape)
        obj[rr, cc] = True
    for cy, cx in [(150, 120), (150, 150)]:  # touching dimer (visible neck)
        rr, cc = draw.disk((cy, cx), 18, shape=obj.shape)
        obj[rr, cc] = True
    rr, cc = draw.ellipse(230, 150, 15, 52, shape=obj.shape)  # rod, aspect ~3.5
    obj[rr, cc] = True
    gray[obj] = 0.1
    return gray


def test_rod_kept_dimer_split_spheres_intact():
    gray = _scene()
    seg = segment(gray, bg_radius=60, min_object_area=50)
    df, _, _ = build_table(
        seg, manual_calibration(1.0), sample_id="synthetic", source_file="synthetic",
    )
    # 2 spheres + 2 (split dimer) + 1 rod = 5 objects
    assert len(df) == 5, df[["diameter_nm", "aspect_ratio", "shape_class"]]

    rods = df[df.shape_class == "rod"]
    assert len(rods) == 1, "exactly one rod expected"
    assert rods.iloc[0].aspect_ratio > 2.5, "rod should be clearly elongated"

    # The dimer produced two spheres sharing a touching group; the rod does not.
    assert df.touching_group_id.notna().sum() == 2, "dimer -> 2 grouped particles"
    assert (df.shape_class == "sphere").sum() == 4, "4 spheres (2 lone + split dimer)"


if __name__ == "__main__":
    test_rod_kept_dimer_split_spheres_intact()
    print("ok")
