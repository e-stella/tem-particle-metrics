#!/usr/bin/env python3
"""napari desktop reviewer — Phase-2 (tier-1 + tier-2).

Pick a folder of TEM images and a tier; each image is segmented on demand and
shown with a coloured mask overlay (green = solo, red = touching). Click a
particle to drop/restore it (yellow = dropped); a docked histogram shows the
kept-particle size distribution live. "Save review" writes a
`<stem>_reviewed_particles.csv`.

Tier-1 (classical) runs in-process. Tier-2 (NP-SAM) runs in the isolated
tem-tier2 env via subprocess (~20 s/image), off the UI thread so the window
stays responsive.

Run in the GUI env:
    conda run -n tem-gui python gui/napari_review.py [folder]
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _qtapi  # noqa: F401,E402  — pick/configure the Qt binding before napari
import napari  # noqa: E402
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from napari.qt.threading import thread_worker  # noqa: E402
from napari.utils.colormaps import DirectLabelColormap  # noqa: E402
from qtpy.QtWidgets import (  # noqa: E402
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from tem_particle_metrics.gui_engine import list_images, segment_image  # noqa: E402
from tem_particle_metrics.review import ReviewSession  # noqa: E402

OUT = ROOT / "outputs" / "gui"
_TRANSPARENT = (0.0, 0.0, 0.0, 0.0)
_ALPHA = 0.55


@thread_worker
def _segment_worker(path: Path, tier: int):
    return segment_image(path, tier=tier)


class ReviewDock(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.paths: list[Path] = []
        self.session: ReviewSession | None = None
        self.image_path: Path | None = None
        self.labels_layer = None
        self._worker = None
        self._build_ui()

    # --- UI ---------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        b_open = QPushButton("Open folder…")
        b_open.clicked.connect(self.open_folder)
        layout.addWidget(b_open)

        self.combo = QComboBox()
        self.combo.currentIndexChanged.connect(self._on_select)
        layout.addWidget(self.combo)

        self.tier_combo = QComboBox()
        self.tier_combo.addItems(["Tier 1 — classical (fast)", "Tier 2 — NP-SAM (aggregates, ~20 s)"])
        self.tier_combo.currentIndexChanged.connect(self._reload_current)
        layout.addWidget(self.tier_combo)

        self.status = QLabel("No image loaded.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.fig = Figure(figsize=(3.4, 2.6), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        row = QHBoxLayout()
        b_save = QPushButton("Save review")
        b_save.clicked.connect(self.save)
        b_reset = QPushButton("Reset")
        b_reset.clicked.connect(self.reset)
        row.addWidget(b_save)
        row.addWidget(b_reset)
        layout.addLayout(row)
        layout.addStretch()

    def _current_tier(self) -> int:
        return 1 if self.tier_combo.currentIndex() == 0 else 2

    # --- folder / image loading ------------------------------------------
    def open_folder(self, _=None, folder: str | None = None) -> None:
        folder = folder or QFileDialog.getExistingDirectory(self, "Select image folder")
        if not folder:
            return
        self.paths = list_images(folder)
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItems([p.name for p in self.paths])
        self.combo.blockSignals(False)
        if self.paths:
            self.combo.setCurrentIndex(0)
            self._on_select(0)
        else:
            self.status.setText("No supported images (.dm3/.dm4/.png/.tif) in that folder.")

    def _on_select(self, idx: int) -> None:
        if 0 <= idx < len(self.paths):
            self.load_image(self.paths[idx])

    def _reload_current(self, _=None) -> None:
        if self.image_path is not None:
            self.load_image(self.image_path)

    def load_image(self, path: Path) -> None:
        tier = self._current_tier()
        note = "  (~20 s, NP-SAM)" if tier == 2 else ""
        self.status.setText(f"Segmenting {path.name}, tier-{tier}{note}…")
        worker = _segment_worker(path, tier)
        worker.returned.connect(lambda res, p=path, t=tier: self._on_segmented(p, t, res))
        worker.errored.connect(lambda exc, p=path: self.status.setText(f"Failed on {p.name}:\n{exc}"))
        worker.start()
        self._worker = worker  # keep a reference so it isn't GC'd

    def _on_segmented(self, path: Path, tier: int, res) -> None:
        self.image_path = path
        self.session = ReviewSession(res.gray, res.id_image, res.table)
        self.viewer.layers.clear()
        self.viewer.add_image(res.gray, name=path.name, colormap="gray", contrast_limits=(0.0, 1.0))
        self.labels_layer = self.viewer.add_labels(res.id_image.astype(int), name=f"particles (tier {tier})")
        self.labels_layer.editable = False
        self.labels_layer.contour = 3   # outline-only view: keeps merges visible
        self.labels_layer.mouse_drag_callbacks.append(self._on_click)
        self._refresh()

    # --- interaction ------------------------------------------------------
    def _on_click(self, layer, event):
        """Toggle the clicked particle (click, not drag)."""
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
        if val:  # 0/None = background
            self.session.toggle(int(val) - 1)
            self._refresh()

    def reset(self) -> None:
        if self.session is not None:
            self.session.reset()
            self._refresh()

    def save(self) -> None:
        if self.session is None or self.image_path is None:
            return
        OUT.mkdir(parents=True, exist_ok=True)
        out = OUT / f"{self.image_path.stem}_reviewed_particles.csv"
        self.session.save_csv(out)
        s = self.session.stats()
        self.status.setText(f"Saved {out.name}  ({s['n_dropped']} dropped).")

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
        self.status.setText(
            f"{self.image_path.name}  ·  tier {self.session and self._current_tier()}\n"
            f"kept {s['n_kept']}/{len(self.session.df)}   mean {s['mean']:.2f} ± {s['sd']:.2f} nm\n"
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
    viewer = napari.Viewer(title="TEM particle reviewer")
    dock = ReviewDock(viewer)
    viewer.window.add_dock_widget(dock, area="right", name="Review")
    if len(argv) > 1:
        dock.open_folder(folder=argv[1])
    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
