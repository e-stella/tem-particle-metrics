"""ReviewSession: dropping a particle removes it from the stats, colours flip
correctly, and decisions round-trip through the CSV. Headless (no GUI).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tem_particle_metrics.review import (  # noqa: E402
    COLOR_DROPPED, COLOR_SOLO, COLOR_TOUCHING, ReviewSession,
)


def _fixture():
    id_image = np.zeros((60, 60), dtype=np.int32)
    id_image[5:15, 5:15] = 1     # particle_id 0, solo
    id_image[5:15, 25:35] = 2    # particle_id 1, touching
    id_image[30:55, 30:55] = 3   # particle_id 2, big, touching
    df = pd.DataFrame({
        "particle_id": [0, 1, 2],
        "diameter_nm": [10.0, 12.0, 50.0],
        "touching_group_id": [np.nan, 5.0, 5.0],
    })
    return id_image, df


def test_drop_updates_stats_and_colors():
    id_image, df = _fixture()
    sess = ReviewSession(np.zeros((60, 60)), id_image, df)

    s0 = sess.stats()
    assert s0["n_kept"] == 3 and abs(s0["mean"] - 24.0) < 1e-9

    # click maps to the right particle, then drop the 50 nm outlier
    assert sess.particle_at(40, 40) == 2
    sess.toggle(2)
    s1 = sess.stats()
    assert s1["n_kept"] == 2 and s1["n_dropped"] == 1
    assert abs(s1["mean"] - 11.0) < 1e-9  # (10 + 12) / 2

    assert sess.color_for(2) == COLOR_DROPPED
    assert sess.color_for(1) == COLOR_TOUCHING
    assert sess.color_for(0) == COLOR_SOLO


def test_decisions_round_trip_through_csv(tmp_path):
    id_image, df = _fixture()
    sess = ReviewSession(np.zeros((60, 60)), id_image, df)
    sess.toggle(2)
    out = tmp_path / "reviewed.csv"
    sess.save_csv(out)

    reloaded = pd.read_csv(out)
    assert reloaded.loc[reloaded.particle_id == 2, "manual_status"].iloc[0] == "dropped"

    sess2 = ReviewSession(np.zeros((60, 60)), id_image, reloaded)
    assert sess2.stats()["n_dropped"] == 1
    assert sess2.color_for(2) == COLOR_DROPPED


if __name__ == "__main__":
    test_drop_updates_stats_and_colors()
    test_decisions_round_trip_through_csv(Path("/tmp"))
    print("ok")
