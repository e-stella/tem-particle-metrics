#!/usr/bin/env python3
"""Headless batch segmenter: a folder of frames -> a run folder + manifest.

Stage ① of the batch split (docs/batch_design.md). Runs UNATTENDED: for every
frame writes <stem>_particles.csv + _labels.npy + _overlay.png and one manifest
row. Resumable — re-running skips frames already `done`, so a crash loses
nothing. The light reviewer (napari_batch_review.py) and aggregator
(aggregate_samples.py) read the run folder afterwards.

Tier-1 (classical) runs anywhere. Tier-2 (NP-SAM/FastSAM) must run in the
tem-tier2 env; tiling (crop_and_enlarge) is ON by default for tier-2 — the
dense-frame recall lever (docs/batch_design.md).

    # tier-1 (dev/tem-gui env):
    python scripts/segment_folder.py <images/> --out-dir <run/> --tier 1 [--nm-per-px N | --bar-length-nm N]
    # tier-2 (tem-tier2 env), tiling on by default:
    .../tem-tier2/bin/python scripts/segment_folder.py <images/> --out-dir <run/> --tier 2
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from tem_particle_metrics import run_image  # noqa: E402
from tem_particle_metrics.batch import (  # noqa: E402
    build_manifest, list_images, load_manifest, manifest_path, save_manifest,
)

_METADATA_CAL = {".dm3", ".dm4"}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", help="folder of TEM frames")
    ap.add_argument("--out-dir", required=True, help="run folder for outputs + manifest")
    ap.add_argument("--tier", type=int, choices=[1, 2], default=1)
    ap.add_argument("--tiling", dest="tiling", action="store_true",
                    help="force tiling on (tier-2; default for tier-2 anyway)")
    ap.add_argument("--no-tiling", dest="tiling", action="store_false",
                    help="force tiling off for tier-2")
    ap.set_defaults(tiling=None)
    ap.add_argument("--nm-per-px", type=float, help="calibration for non-dm3 frames")
    ap.add_argument("--bar-length-nm", type=float, help="scale-bar length for non-dm3 frames")
    ap.add_argument("--force", action="store_true", help="re-segment even done frames")
    args = ap.parse_args(argv[1:])

    run_dir = Path(args.out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    # tiling defaults on for tier-2, off for tier-1, unless overridden.
    tiling = (args.tier == 2) if args.tiling is None else args.tiling

    segmenter = None
    seg_kwargs: dict = {}
    if args.tier == 2:
        from tem_particle_metrics.tier2_npsam import segment_npsam
        segmenter = segment_npsam
        seg_kwargs = {"crop_and_enlarge": tiling}

    # build or resume the manifest
    if manifest_path(run_dir).exists() and not args.force:
        man = load_manifest(run_dir)
        print(f"resuming manifest: {(man['seg_status'] == 'done').sum()}/{len(man)} already done")
    else:
        man = build_manifest(list_images(args.folder), tier=args.tier, tiling=tiling)
        save_manifest(man, run_dir)
        print(f"new manifest: {len(man)} frames, tier {args.tier}, tiling={tiling}")

    for i, row in man.iterrows():
        if row["seg_status"] == "done" and not args.force:
            continue
        frame = Path(row["frame_path"])
        kw: dict = {}
        if frame.suffix.lower() not in _METADATA_CAL:
            if args.nm_per_px is not None:
                kw["nm_per_px"] = args.nm_per_px
            elif args.bar_length_nm is not None:
                kw["bar_length_nm"] = args.bar_length_nm
        t0 = time.time()
        try:
            res = run_image(
                str(frame), sample_id=str(row["sample_id"]), segmenter=segmenter,
                seg_kwargs=seg_kwargs, out_dir=str(run_dir), **kw,
            )
            d = res.table["diameter_nm"]
            man.loc[i, ["nm_per_px", "calibration_source", "n_particles",
                        "mean_nm", "sd_nm", "seg_status", "seg_time_s", "timestamp"]] = [
                res.calibration.nm_per_px, res.calibration.source, len(d),
                float(d.mean()) if len(d) else float("nan"),
                float(d.std(ddof=1)) if len(d) > 1 else float("nan"),
                "done", round(time.time() - t0, 1),
                datetime.now().isoformat(timespec="seconds"),
            ]
            print(f"  [{i+1}/{len(man)}] {frame.name}: n={len(d)} "
                  f"({man.loc[i, 'seg_time_s']}s)")
        except Exception as e:  # noqa: BLE001 — record, don't abort the batch
            man.loc[i, ["seg_status", "timestamp"]] = [
                "error", datetime.now().isoformat(timespec="seconds")]
            print(f"  [{i+1}/{len(man)}] {frame.name}: ERROR {type(e).__name__}: {e}")
        save_manifest(man, run_dir)   # persist after each frame -> resumable

    done = (man["seg_status"] == "done").sum()
    err = (man["seg_status"] == "error").sum()
    print(f"\ndone: {done}/{len(man)} frames"
          f"{f' ({err} errored)' if err else ''}. manifest: {manifest_path(run_dir)}")
    print(f"next: review with gui/napari_batch_review.py, then aggregate_samples.py")
    return 1 if err and done == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
