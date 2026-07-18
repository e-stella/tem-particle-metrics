#!/usr/bin/env python3
"""TEM particle metrics — single-window app (segment → review → export).

One window, the whole workflow, no CLI:
  1. Pick an image folder + tier → "Segment" runs the headless engine in the
     background (tier-2 in the isolated tem-tier2 env via subprocess; tier-1
     in-process env), writing a run folder + manifest. Progress updates live.
  2. When it finishes the run folder loads into the reviewer (reused
     BatchReviewDock): navigate by sample→frame, click to drop/restore, Save.
  3. "Export samples" pools the kept particles into per-sample tables +
     summary.csv.

Run in the GUI env:
    conda run -n tem-app python gui/napari_app.py [folder]
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # sibling gui modules

import _qtapi  # noqa: F401,E402  — pick/configure the Qt binding before napari
import napari  # noqa: E402
from napari.qt.threading import thread_worker  # noqa: E402
from qtpy.QtCore import QTimer  # noqa: E402
from qtpy.QtWidgets import (  # noqa: E402
    QCheckBox, QComboBox, QFileDialog, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget,
)

from tem_particle_metrics import batch  # noqa: E402
from tem_particle_metrics.gui_engine import (  # noqa: E402
    TIER2_PYTHON, batch_engine_command, default_run_dir,
)
from napari_batch_review import BatchReviewDock  # noqa: E402
from aggregate_samples import aggregate  # noqa: E402


@thread_worker
def _run_engine(cmd: list[str]):
    import subprocess
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout)[-1500:])
    return proc.stdout


class OrchestratorDock(QWidget):
    """Segment a folder + export samples; drives the shared review dock."""

    def __init__(self, viewer: napari.Viewer, review: BatchReviewDock):
        super().__init__()
        self.viewer = viewer
        self.review = review
        self.folder: str | None = None
        self.run_dir: Path | None = None
        self._worker = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._n_total = 0
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        b_pick = QPushButton("Pick image folder…")
        b_pick.clicked.connect(self._pick_folder)
        layout.addWidget(b_pick)
        self.folder_label = QLabel("No folder selected.")
        self.folder_label.setWordWrap(True)
        layout.addWidget(self.folder_label)

        self.tier_combo = QComboBox()
        self.tier_combo.addItems(["Tier 1 — classical (fast)", "Tier 2 — NP-SAM (aggregates)"])
        layout.addWidget(self.tier_combo)

        self.tiling_check = QCheckBox("tiling (tier-2, dense fields; slower)")
        self.tiling_check.setChecked(True)
        layout.addWidget(self.tiling_check)

        layout.addWidget(QLabel("nm/px (PNG/TIF only; dm3 auto-calibrates):"))
        self.nmpx_edit = QLineEdit()
        self.nmpx_edit.setPlaceholderText("e.g. 0.5   (leave blank for dm3)")
        layout.addWidget(self.nmpx_edit)

        self.b_seg = QPushButton("Segment")
        self.b_seg.clicked.connect(self.segment)
        layout.addWidget(self.b_seg)

        self.status = QLabel("Pick a folder, choose a tier, then Segment.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        b_export = QPushButton("Export samples")
        b_export.clicked.connect(self.export)
        layout.addWidget(b_export)
        layout.addStretch()

    # --- helpers ----------------------------------------------------------
    def _tier(self) -> int:
        return 1 if self.tier_combo.currentIndex() == 0 else 2

    def _nm_per_px(self) -> float | None:
        txt = self.nmpx_edit.text().strip()
        try:
            return float(txt) if txt else None
        except ValueError:
            return None

    def _pick_folder(self, _=None) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select image folder")
        if folder:
            self.folder = folder
            self.folder_label.setText(folder)
            self._n_total = len(batch.list_images(folder))
            self.status.setText(f"{self._n_total} image(s) found. Choose tier, then Segment.")

    # --- segment ----------------------------------------------------------
    def segment(self, _=None) -> None:
        if not self.folder:
            self.status.setText("Pick an image folder first.")
            return
        tier = self._tier()
        if tier == 2 and not Path(TIER2_PYTHON).exists():
            self.status.setText(f"Tier-2 needs the tem-tier2 env python:\n{TIER2_PYTHON}\n"
                                "(install it or set TEM_TIER2_PYTHON), or use Tier 1.")
            return
        self.run_dir = default_run_dir(self.folder, tier)
        cmd = batch_engine_command(
            self.folder, self.run_dir, tier,
            tiling=(self.tiling_check.isChecked() if tier == 2 else None),
            nm_per_px=self._nm_per_px(),
        )
        note = " (tier-2 in tem-tier2; first frame loads the model)" if tier == 2 else ""
        self.status.setText(f"Segmenting {self._n_total} frame(s), tier {tier}{note}…")
        self.b_seg.setEnabled(False)
        worker = _run_engine(cmd)
        worker.returned.connect(self._on_done)
        worker.errored.connect(self._on_error)
        worker.start()
        self._worker = worker
        self._timer.start(2000)   # poll the manifest for progress

    def _poll(self) -> None:
        if self.run_dir is None or not batch.manifest_path(self.run_dir).exists():
            return
        try:
            man = batch.load_manifest(self.run_dir)
        except Exception:
            return
        done = int((man["seg_status"] == "done").sum())
        err = int((man["seg_status"] == "error").sum())
        self.status.setText(f"Segmenting… {done}/{len(man)} done"
                            f"{f', {err} errored' if err else ''}")

    def _on_done(self, _stdout) -> None:
        self._timer.stop()
        self.b_seg.setEnabled(True)
        self.status.setText(f"Segmentation done → {self.run_dir.name}. Loading for review…")
        self.review.open_run(run_dir=str(self.run_dir))

    def _on_error(self, exc) -> None:
        self._timer.stop()
        self.b_seg.setEnabled(True)
        self.status.setText(f"Segmentation failed:\n{exc}")

    # --- export -----------------------------------------------------------
    def export(self, _=None) -> None:
        run_dir = self.review.run_dir or self.run_dir
        if run_dir is None:
            self.status.setText("Nothing to export — segment or open a run folder first.")
            return
        try:
            summary = aggregate(run_dir)
        except Exception as e:  # noqa: BLE001
            self.status.setText(f"Export failed: {e}")
            return
        lines = [f"Exported {len(summary)} sample(s) → {Path(run_dir)/'samples'}"]
        for r in summary.itertuples():
            flag = " ⚠unreviewed" if bool(r.has_unreviewed_frames) else ""
            lines.append(f"  {r.sample_id}: n={r.n_particles}, "
                         f"{r.mean_nm:.2f}±{r.sd_nm:.2f} nm{flag}")
        self.status.setText("\n".join(lines))


def main(argv: list[str]) -> int:
    viewer = napari.Viewer(title="TEM particle metrics")
    review = BatchReviewDock(viewer)
    controls = OrchestratorDock(viewer, review)
    viewer.window.add_dock_widget(controls, area="right", name="Segment / Export")
    viewer.window.add_dock_widget(review, area="right", name="Review")
    if len(argv) > 1:
        controls.folder = argv[1]
        controls.folder_label.setText(argv[1])
        controls._n_total = len(batch.list_images(argv[1]))
    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
