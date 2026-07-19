#!/usr/bin/env python3
"""Headless batch segmenter: a folder of frames -> a run folder + manifest.

Stage ① of the batch split. Runs UNATTENDED: for every frame writes
<stem>_particles.csv + _labels.npy + _overlay.png and one manifest row.
Resumable — re-running skips frames already `done`, so a crash loses nothing.
The light reviewer (napari_batch_review.py) and aggregator (aggregate_samples.py)
read the run folder afterwards.

Tier choice (`--tier`):
  1     classical only (runs anywhere; clean on sparse fields).
  2     NP-SAM/FastSAM only, tiling on by default (tem-tier2/tem-app env).
  auto  tier-1 on every frame, then ESCALATE only crowded frames to tier-2
        (crowding = fraction of particles in contact; threshold tunable). Sparse
        frames stay tier-1 (clean, no tiling speckles); aggregated frames get
        SAM separation. Needs the torch env. This is usually what you want for a
        mixed folder.

Tier-2 (and auto-escalated) frames that crash fall back to the tier-1 result, so
one bad frame never aborts the run or vanishes. The manifest records the tier
actually used and the crowding score per frame.

    # tier-1 (dev/tem-gui env):
    python scripts/segment_folder.py <images/> --out-dir <run/> --tier 1 [--nm-per-px N]
    # tier-2 or auto (torch env), tiling on by default:
    .../tem-tier2/bin/python scripts/segment_folder.py <images/> --out-dir <run/> --tier auto
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
from tem_particle_metrics.analysis import (  # noqa: E402
    AUTO_CROWDING_THRESHOLD, crowding_score, decide_tier,
)
from tem_particle_metrics.batch import (  # noqa: E402
    build_manifest, list_images, load_manifest, manifest_path, save_manifest,
)

_METADATA_CAL = {".dm3", ".dm4"}


def _segment_one(frame, sample_id, mode, tiling, threshold, cal_kw, run_dir):
    """Segment one frame per `mode` (1 | 2 | "auto").

    Returns (RunResult, actual_tier, crowding, note). `auto` runs tier-1 first
    and escalates to tier-2 only when crowding >= threshold. Tier-2 failures
    (tier-2 and auto) fall back to the tier-1 result — already on disk — so a
    bad frame never aborts the run.
    """
    def tier1():
        return run_image(str(frame), sample_id=sample_id, out_dir=str(run_dir), **cal_kw)

    def tier2():
        from tem_particle_metrics.tier2_npsam import segment_npsam
        return run_image(str(frame), sample_id=sample_id, segmenter=segment_npsam,
                         seg_kwargs={"crop_and_enlarge": tiling}, out_dir=str(run_dir), **cal_kw)

    if mode == 1:
        res = tier1()
        return res, 1, crowding_score(res.table), ""

    if mode == 2:
        try:
            res = tier2()
            return res, 2, crowding_score(res.table), ""
        except Exception as e:  # noqa: BLE001
            res = tier1()   # rewrites outputs with the tier-1 result
            return res, 1, crowding_score(res.table), f"tier-2 failed ({type(e).__name__}) -> tier-1"

    # auto: tier-1 base, escalate crowded frames
    res1 = tier1()
    crowd = crowding_score(res1.table)
    if decide_tier(crowd, threshold) == 1:
        return res1, 1, crowd, ""
    try:
        res2 = tier2()   # overwrites the tier-1 outputs on disk
        return res2, 2, crowd, ""
    except Exception as e:  # noqa: BLE001 — tier-1 outputs (res1) remain on disk
        return res1, 1, crowd, f"tier-2 failed ({type(e).__name__}) -> tier-1"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", help="folder of TEM frames")
    ap.add_argument("--out-dir", required=True, help="run folder for outputs + manifest")
    ap.add_argument("--tier", choices=["1", "2", "auto"], default="1")
    ap.add_argument("--tiling", dest="tiling", action="store_true",
                    help="force tiling on (tier-2/auto; default for those anyway)")
    ap.add_argument("--no-tiling", dest="tiling", action="store_false",
                    help="force tiling off for tier-2/auto")
    ap.set_defaults(tiling=None)
    ap.add_argument("--crowding-threshold", type=float, default=AUTO_CROWDING_THRESHOLD,
                    help=f"auto: escalate to tier-2 above this contact fraction (default {AUTO_CROWDING_THRESHOLD})")
    ap.add_argument("--nm-per-px", type=float, help="calibration for non-dm3 frames")
    ap.add_argument("--bar-length-nm", type=float, help="scale-bar length for non-dm3 frames")
    ap.add_argument("--force", action="store_true", help="re-segment even done frames")
    args = ap.parse_args(argv[1:])

    run_dir = Path(args.out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    mode = int(args.tier) if args.tier in ("1", "2") else "auto"
    # tiling defaults on for tier-2 and auto (its escalated frames), off for tier-1.
    tiling = (mode in (2, "auto")) if args.tiling is None else args.tiling

    if manifest_path(run_dir).exists() and not args.force:
        man = load_manifest(run_dir)
        print(f"resuming manifest: {(man['seg_status'] == 'done').sum()}/{len(man)} already done")
    else:
        man = build_manifest(list_images(args.folder), tier=mode, tiling=tiling)
        save_manifest(man, run_dir)
        print(f"new manifest: {len(man)} frames, tier={args.tier}, tiling={tiling}")

    for i, row in man.iterrows():
        if row["seg_status"] == "done" and not args.force:
            continue
        frame = Path(row["frame_path"])
        cal_kw: dict = {}
        if frame.suffix.lower() not in _METADATA_CAL:
            if args.nm_per_px is not None:
                cal_kw["nm_per_px"] = args.nm_per_px
            elif args.bar_length_nm is not None:
                cal_kw["bar_length_nm"] = args.bar_length_nm
        t0 = time.time()
        try:
            res, tier, crowd, note = _segment_one(
                frame, str(row["sample_id"]), mode, tiling, args.crowding_threshold, cal_kw, run_dir)
            d = res.table["diameter_nm"]
            man.loc[i, ["tier", "crowding", "nm_per_px", "calibration_source", "n_particles",
                        "mean_nm", "sd_nm", "seg_status", "seg_time_s", "timestamp"]] = [
                tier, round(crowd, 3), res.calibration.nm_per_px, res.calibration.source, len(d),
                float(d.mean()) if len(d) else float("nan"),
                float(d.std(ddof=1)) if len(d) > 1 else float("nan"),
                "done", round(time.time() - t0, 1), datetime.now().isoformat(timespec="seconds"),
            ]
            extra = (f"  crowd={crowd:.2f}" if mode == "auto" else "") + (f"  [{note}]" if note else "")
            print(f"  [{i+1}/{len(man)}] {frame.name}: tier-{tier} n={len(d)}{extra} "
                  f"({man.loc[i, 'seg_time_s']}s)")
        except Exception as e:  # noqa: BLE001 — record, don't abort the batch
            man.loc[i, ["seg_status", "timestamp"]] = [
                "error", datetime.now().isoformat(timespec="seconds")]
            print(f"  [{i+1}/{len(man)}] {frame.name}: ERROR {type(e).__name__}: {e}")
        save_manifest(man, run_dir)   # persist after each frame -> resumable

    done_rows = man[man["seg_status"] == "done"]
    done, err = len(done_rows), int((man["seg_status"] == "error").sum())
    routing = done_rows["tier"].value_counts().to_dict()
    print(f"\ndone: {done}/{len(man)} frames"
          f"{f' ({err} errored)' if err else ''}."
          f"{f' routing: {routing}' if mode == 'auto' else ''}")
    print(f"manifest: {manifest_path(run_dir)}")
    print("next: review with gui/napari_batch_review.py, then aggregate_samples.py")
    return 1 if err and done == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
