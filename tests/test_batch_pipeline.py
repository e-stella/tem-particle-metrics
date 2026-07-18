"""Integration: headless engine (tier-1) -> aggregator, with a reviewed frame.
Synthetic PNGs; no torch, no lab data."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import draw, io as skio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import segment_folder  # noqa: E402
import aggregate_samples  # noqa: E402
from tem_particle_metrics import batch  # noqa: E402


def _synthetic_png(path: Path) -> None:
    img = np.full((200, 200), 220, np.uint8)
    for cy, cx in [(50, 50), (50, 150), (150, 100)]:   # 3 separated dark disks
        rr, cc = draw.disk((cy, cx), 15, shape=img.shape)
        img[rr, cc] = 40
    skio.imsave(path, img)


def test_engine_then_aggregate(tmp_path):
    folder = tmp_path / "imgs"
    folder.mkdir()
    _synthetic_png(folder / "s1_a.png")
    _synthetic_png(folder / "s1_b.png")
    run = tmp_path / "run"

    rc = segment_folder.main([
        "segment_folder", str(folder), "--out-dir", str(run),
        "--tier", "1", "--nm-per-px", "1.0",
    ])
    assert rc == 0

    man = batch.load_manifest(run)
    assert (man["seg_status"] == "done").all()
    assert list(man["sample_id"]) == ["s1", "s1"]
    assert batch.particles_csv(run, "s1_a").exists()
    assert batch.labels_npy(run, "s1_a").exists()

    # aggregate raw (both frames unreviewed) -> all detections pooled, flagged
    summary = aggregate_samples.aggregate(run)
    assert len(summary) == 1
    r = summary.iloc[0]
    assert r["sample_id"] == "s1"
    assert r["n_frames"] == 2 and r["n_particles"] == 6     # 3 + 3
    assert bool(r["has_unreviewed_frames"]) is True
    assert (run / "samples" / "s1_sample.csv").exists()

    # "review" frame a: drop one particle -> aggregation must exclude it
    t = pd.read_csv(batch.particles_csv(run, "s1_a"))
    t["manual_status"] = ["kept", "kept", "dropped"]
    t.to_csv(batch.reviewed_csv(run, "s1_a"), index=False)
    man.loc[man["stem"] == "s1_a", "review_status"] = "reviewed"
    batch.save_manifest(man, run)

    summary2 = aggregate_samples.aggregate(run)
    r2 = summary2.iloc[0]
    assert r2["n_particles"] == 5                            # 2 kept (a) + 3 (b)
    assert r2["n_frames_reviewed"] == 1
    assert bool(r2["has_unreviewed_frames"]) is True         # b still unreviewed


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_engine_then_aggregate(Path(d))
    print("ok")
