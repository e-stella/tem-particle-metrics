#!/usr/bin/env python3
"""Light batch reviewer — Stage ② of the batch split (docs/batch_design.md).

Opens a RUN FOLDER produced by scripts/segment_folder.py (a manifest.csv +
per-frame labels/csv/overlay) and lets a human curate it. It does NOT segment
— no torch, no model load — so it opens each frame instantly and runs in any
light env (tem-gui / dev).

Navigate BY SAMPLE (pick a sample → step through its frames); click a particle
to drop/restore (yellow = dropped); the histogram tracks kept particles live.
"Save review" writes <stem>_reviewed_particles.csv and flips the frame's
review_status in the manifest.

    conda run -n tem-gui python gui/napari_batch_review.py [run_dir]
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _qtapi  # noqa: F401,E402  — pick/configure the Qt binding before napari
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import napari  # noqa: E402
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from napari.utils.colormaps import DirectLabelColormap  # noqa: E402
from qtpy.QtWidgets import (  # noqa: E402
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from tem_particle_metrics import batch  # noqa: E402
from tem_particle_metrics.gui_engine import load_gray  # noqa: E402
from tem_particle_metrics.review import ReviewSession  # noqa: E402

_TRANSPARENT = (0.0, 0.0, 0.0, 0.0)
_ALPHA = 0.55


class BatchReviewDock(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.run_dir: Path | None = None
        self.man: pd.DataFrame | None = None
        self.session: ReviewSession | None = None
        self.stem: str | None = None
        self.labels_layer = None
        self._build_ui()
        try:  # one-key accept+advance; the button is the fallback if 'a' conflicts
            viewer.bind_key("a", lambda _v: self.accept_next(), overwrite=True)
        except Exception:
            pass

    # --- UI ---------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        b_open = QPushButton("Open run folder…")
        b_open.clicked.connect(self.open_run)
        layout.addWidget(b_open)

        self.sample_combo = QComboBox()
        self.sample_combo.currentIndexChanged.connect(self._on_sample)
        layout.addWidget(QLabel("sample:"))
        layout.addWidget(self.sample_combo)

        self.frame_combo = QComboBox()
        self.frame_combo.currentIndexChanged.connect(self._on_frame)
        layout.addWidget(QLabel("frame:"))
        layout.addWidget(self.frame_combo)

        self.status = QLabel("Open a run folder (must contain manifest.csv).")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.fig = Figure(figsize=(3.4, 2.6), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        b_accept = QPushButton("Accept ✓ + next frame   (key: a)")
        b_accept.clicked.connect(self.accept_next)
        layout.addWidget(b_accept)

        row = QHBoxLayout()
        b_save = QPushButton("Save review")
        b_save.clicked.connect(self.save)
        b_reset = QPushButton("Reset")
        b_reset.clicked.connect(self.reset)
        row.addWidget(b_save)
        row.addWidget(b_reset)
        layout.addLayout(row)
        layout.addStretch()

    # --- run folder / navigation -----------------------------------------
    def open_run(self, _=None, run_dir: str | None = None) -> None:
        run_dir = run_dir or QFileDialog.getExistingDirectory(self, "Select run folder")
        if not run_dir:
            return
        run_dir = Path(run_dir)
        if not batch.manifest_path(run_dir).exists():
            self.status.setText(f"No manifest.csv in {run_dir} — run segment_folder.py first.")
            return
        self.run_dir = run_dir
        self.man = batch.load_manifest(run_dir)
        self._samples = sorted(self.man["sample_id"].astype(str).unique())
        self.sample_combo.blockSignals(True)
        self.sample_combo.clear()
        self.sample_combo.addItems(self._samples)
        self.sample_combo.blockSignals(False)
        if self._samples:
            self.sample_combo.setCurrentIndex(0)
            self._on_sample(0)

    def _sample_frames(self, sid: str) -> pd.DataFrame:
        m = self.man
        return m[(m["sample_id"].astype(str) == sid) & (m["seg_status"] == "done")]

    def _on_sample(self, _idx: int) -> None:
        sid = self.sample_combo.currentText()
        frames = self._sample_frames(sid)
        self.frame_combo.blockSignals(True)
        self.frame_combo.clear()
        labels = [
            f"{r.stem}  [{'✓' if r.review_status == 'reviewed' else '·'}]"
            for r in frames.itertuples()
        ]
        self.frame_combo.addItems(labels)
        self._frame_stems = list(frames["stem"].astype(str))
        self.frame_combo.blockSignals(False)
        if self._frame_stems:
            self.frame_combo.setCurrentIndex(0)
            self._on_frame(0)
        else:
            self.status.setText(f"{sid}: no segmented frames.")

    def _on_frame(self, idx: int) -> None:
        if 0 <= idx < len(self._frame_stems):
            self.load_frame(self._frame_stems[idx])

    def load_frame(self, stem: str) -> None:
        row = self.man[self.man["stem"].astype(str) == stem].iloc[0]
        path, _is_reviewed = batch.reviewed_or_raw(self.run_dir, stem)  # reviewed restores decisions
        gray = load_gray(row["frame_path"])
        id_image = np.load(batch.labels_npy(self.run_dir, stem))
        table = pd.read_csv(path)
        self.stem = stem
        self.session = ReviewSession(gray, id_image, table)
        self.viewer.layers.clear()
        self.viewer.add_image(gray, name=Path(row["frame_path"]).name,
                              colormap="gray", contrast_limits=(0.0, 1.0))
        self.labels_layer = self.viewer.add_labels(id_image.astype(int), name=f"particles [{stem}]")
        self.labels_layer.editable = False
        self.labels_layer.contour = 3   # outline-only view: keeps merges visible
        self.labels_layer.mouse_drag_callbacks.append(self._on_click)
        self._refresh()

    # --- interaction ------------------------------------------------------
    def _on_click(self, layer, event):
        dragged = False
        yield
        while event.type == "mouse_move":
            dragged = True
            yield
        if dragged or self.session is None:
            return
        val = layer.get_value(
            event.position, view_direction=event.view_direction,
            dims_displayed=event.dims_displayed, world=True,
        )
        if val:
            self.session.toggle(int(val) - 1)
            self._refresh()

    def reset(self) -> None:
        if self.session is not None:
            self.session.reset()
            self._refresh()

    def save(self) -> None:
        if self.session is None or self.stem is None:
            return
        out = batch.reviewed_csv(self.run_dir, self.stem)
        self.session.save_csv(out)
        # flip review_status in the manifest and persist
        self.man.loc[self.man["stem"].astype(str) == self.stem, "review_status"] = "reviewed"
        batch.save_manifest(self.man, self.run_dir)
        s = self.session.stats()
        self.status.setText(f"Saved {out.name}  ({s['n_dropped']} dropped). manifest updated.")
        # reflect the ✓ in the frame dropdown
        i = self.frame_combo.currentIndex()
        self.frame_combo.setItemText(i, f"{self.stem}  [✓]")

    def accept_next(self) -> None:
        """Approve the current frame as-is (Save, even with zero drops) and jump
        to the next frame — the fast path for frames with nothing to drop.
        'Accepted' == Saved, so the frame counts as reviewed and its flag clears."""
        if self.session is None:
            return
        self.save()
        self._advance()

    def _advance(self) -> None:
        fi = self.frame_combo.currentIndex()
        if fi + 1 < self.frame_combo.count():
            self.frame_combo.setCurrentIndex(fi + 1)       # -> _on_frame loads it
            return
        si = self.sample_combo.currentIndex()
        if si + 1 < self.sample_combo.count():
            self.sample_combo.setCurrentIndex(si + 1)      # -> _on_sample loads first frame
            return
        self.status.setText("Reviewed the last frame — all samples done.")

    # --- rendering --------------------------------------------------------
    def _refresh(self) -> None:
        if self.session is None:
            return
        cdict = {0: _TRANSPARENT, None: _TRANSPARENT}
        for pid in self.session.status:
            r, g, b = self.session.color_for(pid)
            cdict[pid + 1] = (r, g, b, _ALPHA)
        self.labels_layer.colormap = DirectLabelColormap(color_dict=cdict)

        s = self.session.stats()
        n_reviewed = int((self._sample_frames(self.sample_combo.currentText())
                          ["review_status"] == "reviewed").sum())
        n_frames = len(self._frame_stems)
        self.status.setText(
            f"sample {self.sample_combo.currentIndex()+1}/{len(self._samples)} · "
            f"frame {self.frame_combo.currentIndex()+1}/{n_frames} "
            f"({n_reviewed} reviewed)\n"
            f"{self.stem}: kept {s['n_kept']}/{len(self.session.df)}   "
            f"mean {s['mean']:.2f} ± {s['sd']:.2f} nm\n"
            f"click a particle to drop/restore (yellow = dropped)"
        )
        self.ax.clear()
        kept = self.session.kept()["diameter_nm"].to_numpy()
        if len(kept):
            self.ax.hist(kept, bins=30, color="#3b7dd8", alpha=0.85)
            self.ax.axvline(kept.mean(), color="k", ls="--", lw=1)
        self.ax.set_xlabel("diameter (nm)")
        self.ax.set_ylabel("count")
        self.canvas.draw_idle()


def main(argv: list[str]) -> int:
    viewer = napari.Viewer(title="TEM batch reviewer")
    dock = BatchReviewDock(viewer)
    viewer.window.add_dock_widget(dock, area="right", name="Batch review")
    if len(argv) > 1:
        dock.open_run(run_dir=argv[1])
    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
