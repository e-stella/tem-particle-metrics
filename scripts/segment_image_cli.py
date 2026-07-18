#!/usr/bin/env python3
"""Segment ONE image and write <stem>_labels.npy + <stem>_particles.csv to
--out-dir. The GUI (tem-gui env, no torch) calls this in the tem-tier2 env via
subprocess to run tier-2 (NP-SAM) without importing torch into the GUI process.

    <env>/bin/python scripts/segment_image_cli.py <image> --tier 2 --out-dir <dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tem_particle_metrics import run_image  # noqa: E402
from tem_particle_metrics.gui_engine import _METADATA_CAL  # noqa: E402


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--tier", type=int, choices=[1, 2], default=2)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--nm-per-px", type=float)
    ap.add_argument("--bar-length-nm", type=float)
    args = ap.parse_args(argv[1:])

    segmenter = None
    if args.tier == 2:
        from tem_particle_metrics.tier2_npsam import segment_npsam
        segmenter = segment_npsam

    path = Path(args.image)
    kw: dict = {}
    if path.suffix.lower() not in _METADATA_CAL:
        if args.nm_per_px is not None:
            kw["nm_per_px"] = args.nm_per_px
        elif args.bar_length_nm is not None:
            kw["bar_length_nm"] = args.bar_length_nm

    run_image(path, sample_id=path.stem, segmenter=segmenter, out_dir=args.out_dir, **kw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
