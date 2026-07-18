"""batch.py units: sample grouping rule, manifest round-trip, reviewed/raw
precedence. No torch, no lab data."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tem_particle_metrics import batch  # noqa: E402


def test_sample_id_from_filename():
    assert batch.sample_id_from_filename("scan_a") == "scan"
    assert batch.sample_id_from_filename("img_c") == "img"
    assert batch.sample_id_from_filename("run_23_e") == "run_23"
    assert batch.sample_id_from_filename("frame_10") == "frame"          # numeric frame index
    assert batch.sample_id_from_filename("overview_map") == "overview_map"  # no frame suffix
    assert batch.sample_id_from_filename("img_ab") == "img_ab"          # two letters: not a frame suffix
    assert batch.sample_id_from_filename("a") == "a"                    # never empty


def test_manifest_roundtrip(tmp_path):
    paths = [tmp_path / "scan_a.dm3", tmp_path / "scan_b.dm3"]
    for p in paths:
        p.write_bytes(b"")
    man = batch.build_manifest(paths, tier=2, tiling=True)
    assert list(man["sample_id"]) == ["scan", "scan"]
    assert (man["seg_status"] == "pending").all()
    assert (man["review_status"] == "unreviewed").all()

    batch.save_manifest(man, tmp_path)
    man2 = batch.load_manifest(tmp_path)
    assert list(man2["stem"]) == ["scan_a", "scan_b"]
    assert int(man2["tier"].iloc[0]) == 2
    assert bool(man2["tiling"].iloc[0]) is True


def test_reviewed_or_raw_precedence(tmp_path):
    assert batch.reviewed_or_raw(tmp_path, "x") == (None, False)      # neither

    batch.particles_csv(tmp_path, "x").write_text("particle_id\n0\n")
    path, is_rev = batch.reviewed_or_raw(tmp_path, "x")               # raw only
    assert is_rev is False and path.name == "x_particles.csv"

    batch.reviewed_csv(tmp_path, "x").write_text("particle_id,manual_status\n0,kept\n")
    path, is_rev = batch.reviewed_or_raw(tmp_path, "x")               # reviewed wins
    assert is_rev is True and path.name == "x_reviewed_particles.csv"


if __name__ == "__main__":
    test_sample_id_from_filename()
    print("ok")
