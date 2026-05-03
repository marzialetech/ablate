#!/usr/bin/env python3
"""Per-frame diagnostic for the n-phase Zalesak gates.

Designed for the tight Chiu-Lin vs PM iteration loop. For each frame, prints
and appends a single line to /tmp/chiulin_progress.log:

    ISO_TIMESTAMP | gate=N | run=LABEL | step=NNNN | t=T.TTTTT | mass_drift=...
                  | finite_frac=... | slot_fraction=... | max_fsharp=...
                  | max_alpha_disk=... | oob_count=... | verdict={pass,fail,nan}

Slot metric is borrowed from tools/sweep_leaderboard.py (convex-hull "missing
mass" for each disk phase, summed). max_fsharp pulls from the aux fsharpk
field if present; otherwise reports nan.

Usage:
    python tools/quick_zalesak_diag.py runs/sweeps/zalesak_2disk_3phase_pm \
        --label pm --gate 1 --frames last
    python tools/quick_zalesak_diag.py runs/sweeps/zalesak_2disk_3phase_chiulin \
        --label cl --gate 1 --frames all
"""

from __future__ import annotations

import argparse
import datetime as _dt
import math
import sys
from pathlib import Path

import h5py
import numpy as np

try:
    from skimage.morphology import convex_hull_image
except ImportError:
    convex_hull_image = None


PROGRESS_LOG = Path("/tmp/chiulin_progress.log")


def latest_tagged_subdir(run_dir: Path) -> Path:
    """Resolve a run dir (which may directly contain `domain/` or contain a
    tagDirectory-style timestamped subdir) to the path holding `domain/`."""
    if (run_dir / "domain").is_dir():
        return run_dir
    # Pick newest direct subdir that has a domain/ inside it.
    subs = [
        d for d in run_dir.iterdir()
        if d.is_dir() and (d / "domain").is_dir()
    ]
    if not subs:
        raise FileNotFoundError(f"no domain/ directory found anywhere under {run_dir}")
    subs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return subs[0]


def discover_snapshots(run_dir: Path) -> list[Path]:
    base = latest_tagged_subdir(run_dir)
    return sorted((base / "domain").glob("domain.*.hdf5"))


def step_of(snap: Path) -> int:
    # domain.NNNNN.hdf5 -> NNNNN
    try:
        return int(snap.stem.split(".")[-1])
    except Exception:
        return -1


def infer_grid(ncell: int) -> tuple[int, int]:
    nx = int(round(math.sqrt(ncell)))
    if nx * nx != ncell:
        return ncell, 1
    return nx, nx


def slot_fraction(alphak: np.ndarray) -> float:
    """Fraction of cells in disks' convex hulls where the disk loses argmax.
    Mirrors slot_fraction in tools/sweep_leaderboard.py."""
    if convex_hull_image is None:
        return float("nan")
    nphase = alphak.shape[1]
    s = alphak.sum(axis=1)
    finite = np.isfinite(alphak).all(axis=1) & (s > 1e-9)
    if not finite.any():
        return float("nan")
    nx, ny = infer_grid(alphak.shape[0])
    if nx * ny != alphak.shape[0]:
        return float("nan")
    argmax = np.full(alphak.shape[0], -1, dtype=int)
    argmax[finite] = np.argmax(alphak[finite], axis=1)
    argmax2d = argmax.reshape(ny, nx)
    total_holes = 0
    for k in range(nphase - 1):  # all but background
        mask = argmax2d == k
        if not mask.any():
            continue
        try:
            hull = convex_hull_image(mask)
        except Exception:
            continue
        notch = hull & ~mask
        total_holes += int(notch.sum())
    return total_holes / alphak.shape[0]


def max_fsharp(h: h5py.File) -> float:
    for key in (
        "cell_fields/aux_fsharpk",
        "cell_fields/aux_fsharpk0",
        "cell_fields/fsharpk",
    ):
        if key in h:
            arr = np.asarray(h[key][0])
            arr = arr[np.isfinite(arr)]
            return float(arr.max(initial=0.0)) if arr.size else float("nan")
    # search any field starting with aux_fsharpk
    cf = h.get("cell_fields")
    if cf is not None:
        for k in cf.keys():
            if k.startswith("aux_fsharpk") or k == "fsharpk":
                arr = np.asarray(cf[k][0])
                arr = arr[np.isfinite(arr)]
                return float(arr.max(initial=0.0)) if arr.size else float("nan")
    return float("nan")


def diagnose_frame(snap: Path, label: str, gate: int, threshold: float = 1e-3) -> dict:
    with h5py.File(snap, "r") as h:
        t = float(h["time"][0, 0])
        a = np.asarray(h["cell_fields/solution_alphak"][0])
        finite = np.isfinite(a).all(axis=1)
        finite_frac = float(finite.mean())
        if finite.any():
            s = a[finite].sum(axis=1)
            mass_drift = float(np.abs(s - 1.0).max())
            af = a[finite]
            oob_count = int(((af < -threshold) | (af > 1.0 + threshold)).any(axis=1).sum())
            disk_max = float(af[:, :-1].max(initial=0.0))
        else:
            mass_drift = float("nan")
            oob_count = int(a.shape[0])
            disk_max = float("nan")
        sf = slot_fraction(a)
        mf = max_fsharp(h)
    step = step_of(snap)
    if not np.isfinite(finite_frac) or finite_frac < 0.999:
        verdict = "nan"
    elif mass_drift > 1e-2 or oob_count > a.shape[0] * 0.01:
        verdict = "fail"
    else:
        verdict = "pass"
    rec = {
        "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "gate": gate,
        "run": label,
        "step": step,
        "t": t,
        "mass_drift": mass_drift,
        "finite_frac": finite_frac,
        "slot_fraction": sf,
        "max_fsharp": mf,
        "max_alpha_disk": disk_max,
        "oob_count": oob_count,
        "verdict": verdict,
        "snap": str(snap),
    }
    return rec


def write_log(rec: dict) -> None:
    line = (
        f"{rec['ts']} | gate={rec['gate']} | run={rec['run']:<3s} | "
        f"step={rec['step']:>5d} | t={rec['t']:.5f} | "
        f"mass_drift={rec['mass_drift']:.2e} | finite_frac={rec['finite_frac']:.4f} | "
        f"slot_fraction={rec['slot_fraction']:.5f} | max_fsharp={rec['max_fsharp']:.2e} | "
        f"max_alpha_disk={rec['max_alpha_disk']:.4f} | oob={rec['oob_count']:>5d} | "
        f"verdict={rec['verdict']}"
    )
    print(line)
    with PROGRESS_LOG.open("a") as f:
        f.write(line + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--label", required=True, help="short label, e.g. pm or cl")
    parser.add_argument("--gate", type=int, required=True)
    parser.add_argument("--frames", default="last", choices=("first", "last", "all"))
    parser.add_argument("--threshold", type=float, default=1e-3,
                        help="alpha out-of-bounds tolerance (default 1e-3)")
    args = parser.parse_args()

    snaps = discover_snapshots(args.run_dir)
    if not snaps:
        print(f"NO_SNAPSHOTS run={args.label} dir={args.run_dir}", file=sys.stderr)
        return 2
    if args.frames == "first":
        snaps = snaps[:1]
    elif args.frames == "last":
        snaps = snaps[-1:]
    rc = 0
    for snap in snaps:
        rec = diagnose_frame(snap, args.label, args.gate, args.threshold)
        write_log(rec)
        if rec["verdict"] != "pass":
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
