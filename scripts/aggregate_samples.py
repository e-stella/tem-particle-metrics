#!/usr/bin/env python3
"""Aggregate a run folder's per-frame tables into per-SAMPLE tables.

Stage ③ of the batch split — the "5 frames → 1 sample"
step. For each sample_id in the manifest, pool the KEPT particles across its
frames into `samples/<sample_id>_sample.csv` (the per-particle table the sister
optics repo consumes) plus `samples/summary.csv` (mean±sd/n per sample).

Per frame it prefers `<stem>_reviewed_particles.csv` (dropping
manual_status==dropped) over the raw `<stem>_particles.csv`; a sample built
with any unreviewed frame is flagged (nothing silently dropped).

    python scripts/aggregate_samples.py <run/> [--calibrate]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from tem_particle_metrics.batch import load_manifest, reviewed_or_raw  # noqa: E402


def aggregate(run_dir: str | Path, *, calibrate: bool = False):
    """Pool per-frame tables into per-sample tables; returns the summary df.

    Writes `<run>/samples/<sample_id>_sample.csv` and `.../summary.csv`.
    """
    run_dir = Path(run_dir)
    man = load_manifest(run_dir)
    out_dir = run_dir / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    if calibrate:
        from tem_particle_metrics.analysis import apply_size_calibration

    summary_rows = []
    for sid, grp in man.groupby("sample_id", sort=True):
        parts, n_frames, n_reviewed, missing = [], 0, 0, 0
        for stem in grp["stem"]:
            path, is_reviewed = reviewed_or_raw(run_dir, str(stem))
            if path is None:
                missing += 1          # frame not segmented (pending/error)
                continue
            t = pd.read_csv(path)
            if is_reviewed:
                n_reviewed += 1
                if "manual_status" in t.columns:
                    t = t[t["manual_status"] != "dropped"]
            n_frames += 1
            parts.append(t)

        if not parts:
            print(f"{sid}: no segmented frames, skipped")
            continue
        allp = pd.concat(parts, ignore_index=True)
        if calibrate:
            allp = apply_size_calibration(allp)
        allp.to_csv(out_dir / f"{sid}_sample.csv", index=False)

        d = allp["diameter_nm"]
        has_unreviewed = n_reviewed < n_frames
        summary_rows.append({
            "sample_id": sid,
            "n_frames": n_frames,
            "n_frames_reviewed": n_reviewed,
            "n_particles": len(d),
            "mean_nm": round(float(d.mean()), 3) if len(d) else float("nan"),
            "sd_nm": round(float(d.std(ddof=1)), 3) if len(d) > 1 else float("nan"),
            "has_unreviewed_frames": has_unreviewed,
            "frames_missing": missing,
        })
        flag = "  ⚠ has unreviewed frames" if has_unreviewed else ""
        print(f"{sid}: {n_frames} frames ({n_reviewed} reviewed), n={len(d)}, "
              f"mean={d.mean():.2f}, sd={d.std(ddof=1):.2f}{flag}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"\nwrote {len(summary)} sample tables + summary.csv to {out_dir}")
    return summary


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="run folder produced by segment_folder.py")
    ap.add_argument("--calibrate", action="store_true",
                    help="apply the tier-2 size-calibration factor")
    args = ap.parse_args(argv[1:])
    aggregate(args.run_dir, calibrate=args.calibrate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
