"""I/O layer: load micrographs as grayscale float images.

This is the ONE place pixel values live in native form. Everything
downstream works in physical units (nm) via a Calibration.

PNG/TIFF path only for now; dm4/emd (with metadata calibration) lands
when native microscope files are provided.
"""
from __future__ import annotations

from pathlib import Path

from skimage import color, io
from skimage.util import img_as_float
import numpy as np


def load_grayscale(path: str | Path) -> np.ndarray:
    """Load an image as a 2-D float array in [0, 1].

    RGB inputs are converted with the luminance-weighted rgb2gray; single
    channel inputs are scaled to [0, 1] by dtype range.
    """
    img = io.imread(str(path))
    if img.ndim == 3:
        return color.rgb2gray(img[..., :3])
    return img_as_float(img)


# spatial units -> nanometres
_UNIT_NM = {
    "nm": 1.0, "µm": 1e3, "μm": 1e3, "um": 1e3, "mm": 1e6, "m": 1e9,
    "å": 0.1, "a": 0.1, "angstrom": 0.1, "pm": 1e-3,
}


def _unit_to_nm(units: str) -> float:
    return _UNIT_NM.get(str(units).strip().lower(), 1.0)


def load_dm3(path: str | Path) -> tuple[np.ndarray, float]:
    """Load a Gatan dm3/dm4 micrograph. Returns (gray in [0,1], nm_per_px).

    Calibration comes from the file metadata (no scale bar needed); units are
    normalised to nm (dm3 may report nm or µm). Intensity is robustly rescaled
    (0.5–99.5 pct) to [0,1]; particles stay dark, matching the PNG path.
    """
    try:
        from rsciio.digitalmicrograph import file_reader
    except ImportError:  # older rosettasciio
        from rsciio.digital_micrograph import file_reader

    sig = file_reader(str(path))[0]
    data = np.squeeze(np.asarray(sig["data"], dtype=np.float64))
    if data.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {data.shape}")

    scales = [
        ax["scale"] * _unit_to_nm(ax.get("units", "nm"))
        for ax in sig["axes"]
        if not ax.get("navigate", False) and "scale" in ax
    ]
    if not scales:
        raise ValueError("no spatial calibration in dm3 metadata")
    nm_per_px = float(np.mean(scales))

    lo, hi = np.percentile(data, [0.5, 99.5])
    gray = np.clip((data - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    return gray, nm_per_px
