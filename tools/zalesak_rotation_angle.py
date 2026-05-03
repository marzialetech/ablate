#!/usr/bin/env python3
"""Compute the actual rotation angle of the Zalesak top disk per frame
by tracking its alphak-weighted centroid relative to (0.5, 0.5).

For omega=30 rad/s the *expected* angle is omega*t (mod 2pi). This script
reports both the expected angle and the *measured* angle (from centroid
position), and the cumulative rotation through the run. Useful to detect
any systematic lag from the source-term-imposed velocity vs the actual
advection.
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path

import h5py
import numpy as np


def latest(run_dir: Path) -> Path:
    if (run_dir / "domain").is_dir():
        return run_dir
    subs = [d for d in run_dir.iterdir() if d.is_dir() and (d / "domain").is_dir()]
    subs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return subs[0]


def disk_angle(snap: Path, disk_idx: int = 0) -> tuple[float, float, float, float]:
    """Return (t, cx, cy, angle_deg) where angle_deg is atan2(cy-0.5, cx-0.5)."""
    with h5py.File(snap, "r") as h:
        t = float(h["time"][0, 0])
        a = np.asarray(h["cell_fields/solution_alphak"][0])  # (ncell, nphases)
    n = a.shape[0]
    nx = int(round(math.sqrt(n)))
    if nx * nx != n:
        raise SystemExit(f"non-square mesh ncell={n}")
    ny = nx
    weights = a[:, disk_idx].clip(min=0.0)
    if weights.sum() < 1e-9:
        return t, math.nan, math.nan, math.nan
    # cell centroids on a uniform unit square
    xs = (np.arange(nx) + 0.5) / nx
    ys = (np.arange(ny) + 0.5) / ny
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    cx = float((X.ravel() * weights).sum() / weights.sum())
    cy = float((Y.ravel() * weights).sum() / weights.sum())
    ang = math.degrees(math.atan2(cy - 0.5, cx - 0.5))
    return t, cx, cy, ang


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("--disk", type=int, default=0, help="phase index to track (default top disk = 0)")
    ap.add_argument("--omega", type=float, default=30.0)
    ap.add_argument("--initial-deg", type=float, default=90.0,
                    help="expected initial angle of the disk centroid in degrees (default top = 90)")
    args = ap.parse_args()

    base = latest(args.run)
    snaps = sorted((base / "domain").glob("domain.*.hdf5"))
    if not snaps:
        raise SystemExit(f"no snapshots under {base}")

    print(f"# run: {args.run}")
    print(f"# tracking disk = {args.disk} (assumed initial centroid angle {args.initial_deg} deg)")
    print(f"# expected angular speed omega = {args.omega} rad/s")
    print()
    print(f"{'frame':>5} {'t':>8} {'cx':>8} {'cy':>8} {'meas_deg':>10} {'expect_deg':>11} {'cum_meas_deg':>13} {'lag_deg':>9}")

    last = None
    cum = 0.0  # cumulative rotation (signed) measured
    for i, s in enumerate(snaps):
        t, cx, cy, ang = disk_angle(s, args.disk)
        if not math.isfinite(ang):
            continue
        # expected: starts at args.initial_deg, advances by omega*t (CCW positive)
        exp_deg = args.initial_deg + math.degrees(args.omega * t)
        # cumulative measured: track delta to handle wraparound
        if last is not None:
            d = ang - last
            # unwrap: clamp delta to [-180, 180]
            while d > 180.0: d -= 360.0
            while d < -180.0: d += 360.0
            cum += d
        else:
            cum = 0.0
        last = ang
        cum_meas_total_deg = cum + args.initial_deg  # full from initial
        lag = exp_deg - cum_meas_total_deg
        print(f"{i:>5d} {t:>8.4f} {cx:>8.4f} {cy:>8.4f} {ang:>10.3f} {exp_deg:>11.3f} {cum_meas_total_deg:>13.3f} {lag:>9.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
