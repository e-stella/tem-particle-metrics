"""Manual QC review state: keep/drop individual particles by hand.

The machine proposes a segmentation; a human disposes. A `ReviewSession`
holds the per-particle table plus a kept/dropped decision per particle,
recomputes summary stats over kept particles only, and round-trips the
decisions through a `manual_status` column in the CSV. Nothing is deleted —
a dropped particle stays in the table, flagged out of the statistics.

The interactive review UI is in the napari app; this module is
the pure, testable core (no GUI backend needed).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from skimage import measure

# outline colours (RGB): kept-solo, kept-touching, human-dropped
COLOR_SOLO = (0.10, 0.90, 0.10)
COLOR_TOUCHING = (1.00, 0.00, 0.00)
COLOR_DROPPED = (1.00, 0.85, 0.00)

KEPT, DROPPED = "kept", "dropped"


class ReviewSession:
    """Keep/drop decisions over a per-particle table.

    `id_image` pixels are `particle_id + 1` (0 = background), so a clicked
    pixel maps straight to a row. Decisions initialise from a `manual_status`
    column if the table already has one (re-opening a reviewed sample).
    """

    def __init__(self, gray: np.ndarray, id_image: np.ndarray, df: pd.DataFrame):
        self.gray = gray
        self.id_image = id_image.astype(np.int32)
        self.df = df.reset_index(drop=True)
        if "manual_status" in self.df.columns:
            self.status = {
                int(r.particle_id): (DROPPED if r.manual_status == DROPPED else KEPT)
                for r in self.df.itertuples()
            }
        else:
            self.status = {int(pid): KEPT for pid in self.df["particle_id"]}
        self._touching = {
            int(r.particle_id): pd.notna(r.touching_group_id)
            for r in self.df.itertuples()
        }
        self.contours = _particle_contours(self.id_image)

    # --- decisions -------------------------------------------------------
    def toggle(self, pid: int) -> str | None:
        """Flip keep/drop for a particle; returns the new status (or None)."""
        if pid not in self.status:
            return None
        self.status[pid] = DROPPED if self.status[pid] == KEPT else KEPT
        return self.status[pid]

    def particle_at(self, row: int, col: int) -> int | None:
        """particle_id under an image pixel, or None for background."""
        if not (0 <= row < self.id_image.shape[0] and 0 <= col < self.id_image.shape[1]):
            return None
        v = int(self.id_image[row, col])
        return v - 1 if v > 0 else None

    def reset(self) -> None:
        for pid in self.status:
            self.status[pid] = KEPT

    def color_for(self, pid: int) -> tuple[float, float, float]:
        if self.status.get(pid) == DROPPED:
            return COLOR_DROPPED
        return COLOR_TOUCHING if self._touching.get(pid) else COLOR_SOLO

    # --- stats / IO ------------------------------------------------------
    def kept(self) -> pd.DataFrame:
        keep_ids = [pid for pid, s in self.status.items() if s == KEPT]
        return self.df[self.df["particle_id"].isin(keep_ids)]

    def stats(self) -> dict:
        d = self.kept()["diameter_nm"]
        n = len(d)
        return {
            "n_kept": n,
            "n_dropped": len(self.df) - n,
            "mean": float(d.mean()) if n else float("nan"),
            "sd": float(d.std(ddof=1)) if n > 1 else float("nan"),
        }

    def to_frame(self) -> pd.DataFrame:
        out = self.df.copy()
        out["manual_status"] = [self.status[int(pid)] for pid in out["particle_id"]]
        return out

    def save_csv(self, path: str | Path) -> None:
        self.to_frame().to_csv(path, index=False)


def _particle_contours(id_image: np.ndarray) -> dict[int, list[np.ndarray]]:
    """Per-particle boundary polylines in full-image (row, col) coordinates.

    Uses each region's cropped mask + bbox offset, so it scales to hundreds
    of particles without a full-image contour call each.
    """
    out: dict[int, list[np.ndarray]] = {}
    for r in measure.regionprops(id_image):
        minr, minc = r.bbox[0], r.bbox[1]
        padded = np.pad(r.image.astype(float), 1)  # pad so edge particles close
        cs = measure.find_contours(padded, 0.5)
        out[r.label - 1] = [c + [minr - 1, minc - 1] for c in cs]
    return out
