"""gui_engine: folder listing + in-process segmentation for the napari GUI.
Self-contained (synthetic PNG); no napari, no lab data."""
import sys
from pathlib import Path

import numpy as np
import pytest
from skimage import draw, io as skio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tem_particle_metrics.gui_engine import list_images, segment_image  # noqa: E402


def _synthetic_png(path: Path) -> None:
    img = np.full((200, 200), 220, np.uint8)          # light background
    for cy, cx in [(50, 50), (50, 150), (150, 100)]:  # 3 dark, separated particles
        rr, cc = draw.disk((cy, cx), 15, shape=img.shape)
        img[rr, cc] = 40
    skio.imsave(path, img)


def test_list_images_filters_and_sorts(tmp_path):
    for name in ("b.png", "a.dm3", "notes.txt", "c.tif"):
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in list_images(tmp_path)] == ["a.dm3", "b.png", "c.tif"]


def test_segment_png_returns_image_and_ids(tmp_path):
    p = tmp_path / "s.png"
    _synthetic_png(p)
    res = segment_image(p, nm_per_px=1.0)
    assert len(res.table) == 3
    assert int(res.id_image.max()) == 3        # pixel = particle_id + 1
    assert res.gray is not None and res.gray.shape == (200, 200)


def test_segment_requires_calibration(tmp_path):
    p = tmp_path / "s.png"
    _synthetic_png(p)
    with pytest.raises(ValueError):
        segment_image(p)                        # non-dm3 needs nm_per_px


def test_tier2_missing_env_errors_clearly(tmp_path, monkeypatch):
    p = tmp_path / "s.png"
    _synthetic_png(p)
    monkeypatch.setattr("tem_particle_metrics.gui_engine.TIER2_PYTHON", "/no/such/python")
    with pytest.raises(FileNotFoundError):
        segment_image(p, tier=2, nm_per_px=1.0)  # tier-2 needs the isolated env


def test_default_run_dir():
    from tem_particle_metrics.gui_engine import default_run_dir
    p = default_run_dir("/a/b/October_2024", 2)
    assert p.name == "October_2024_tier2" and p.parent.name == "runs"


def test_batch_engine_command_tier2(monkeypatch):
    from tem_particle_metrics import gui_engine
    monkeypatch.setattr(gui_engine, "TIER2_PYTHON", "/envs/tem-tier2/bin/python")
    cmd = gui_engine.batch_engine_command("/imgs", "/run", 2, tiling=True)
    assert cmd[0] == "/envs/tem-tier2/bin/python"
    assert cmd[1].endswith("segment_folder.py") and cmd[2] == "/imgs"
    assert cmd[cmd.index("--tier") + 1] == "2" and "--tiling" in cmd


def test_batch_engine_command_tier1_uses_current_python(monkeypatch):
    from tem_particle_metrics.gui_engine import batch_engine_command
    cmd = batch_engine_command("/imgs", "/run", 1, nm_per_px=0.5)
    assert cmd[0] == sys.executable                       # tier-1 = light env, in-place
    assert "--tiling" not in cmd and "--no-tiling" not in cmd
    assert cmd[cmd.index("--nm-per-px") + 1] == "0.5"


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_list_images_filters_and_sorts(Path(d))
    print("ok")
