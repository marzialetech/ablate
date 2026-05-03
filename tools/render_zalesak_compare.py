#!/usr/bin/env python3
"""Side-by-side render of the alphak field for a list of zalesak runs at the
same step. For each run, plot:

  row 0: argmax(alphak) (which phase wins each cell)
  row 1: alphak[disk_phase_0] (continuous alpha for top disk)
  row 2: a sharpness diagnostic = max(alphak) - second-max(alphak), which
         shows how decisive the argmax is. Sharp interfaces show high values
         everywhere; diffuse interfaces show low values in the band.

Default disk_phase = 0 (top disk). Use --frame to pick the snapshot index.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def latest_tagged_subdir(run_dir: Path) -> Path:
    if (run_dir / "domain").is_dir():
        return run_dir
    subs = [d for d in run_dir.iterdir() if d.is_dir() and (d / "domain").is_dir()]
    if not subs:
        raise FileNotFoundError(f"no domain/ under {run_dir}")
    subs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return subs[0]


def discover_snapshots(run_dir: Path) -> list[Path]:
    base = latest_tagged_subdir(run_dir)
    return sorted((base / "domain").glob("domain.*.hdf5"))


def infer_grid(ncell: int) -> tuple[int, int]:
    nx = int(round(math.sqrt(ncell)))
    if nx * nx != ncell:
        return ncell, 1
    return nx, nx


def load_alphak(snap: Path) -> tuple[float, np.ndarray, int, int]:
    with h5py.File(snap, "r") as h:
        t = float(h["time"][0, 0])
        a = np.asarray(h["cell_fields/solution_alphak"][0])
    nx, ny = infer_grid(a.shape[0])
    return t, a, nx, ny


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path, help="run directories")
    ap.add_argument("--labels", nargs="+", default=None, help="labels for columns; default = run name")
    ap.add_argument("--frame", type=int, default=-1, help="snapshot index per run; -1 = last finite (default)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--disk", type=int, default=0)
    args = ap.parse_args()

    if args.labels is None:
        args.labels = [r.name for r in args.runs]
    assert len(args.labels) == len(args.runs)

    cols = []
    for run in args.runs:
        snaps = discover_snapshots(run)
        if not snaps:
            raise SystemExit(f"no snapshots in {run}")
        if args.frame == -1:
            chosen = None
            for s in reversed(snaps):
                t, a, nx, ny = load_alphak(s)
                if np.isfinite(a).all(axis=1).mean() > 0.99:
                    chosen = (s, t, a, nx, ny)
                    break
            if chosen is None:
                t, a, nx, ny = load_alphak(snaps[-1])
                chosen = (snaps[-1], t, a, nx, ny)
        else:
            idx = max(0, min(len(snaps) - 1, args.frame))
            t, a, nx, ny = load_alphak(snaps[idx])
            chosen = (snaps[idx], t, a, nx, ny)
        cols.append(chosen)

    nrows = 3
    ncols = len(cols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.4 * nrows), squeeze=False)

    for ci, (snap, t, a, nx, ny) in enumerate(cols):
        finite = np.isfinite(a).all(axis=1)
        amat = a.reshape(ny, nx, -1)
        argmax = np.argmax(amat, axis=2)
        argmax_finite = np.where(finite.reshape(ny, nx), argmax, -1)
        amat_disk = amat[..., args.disk]
        sorted_a = np.sort(amat, axis=2)
        sharpness = sorted_a[..., -1] - sorted_a[..., -2]

        axes[0, ci].imshow(argmax_finite, origin="lower", cmap="viridis", vmin=-1, vmax=amat.shape[-1] - 1, extent=(0, 1, 0, 1))
        axes[0, ci].set_title(f"{args.labels[ci]}\nstep={int(snap.stem.split('.')[-1])*25 if 'domain' in snap.stem else int(snap.stem.split('.')[-1])} t={t:.4f}", fontsize=10)
        axes[0, ci].set_xticks([]); axes[0, ci].set_yticks([])

        im1 = axes[1, ci].imshow(amat_disk, origin="lower", cmap="magma", vmin=0, vmax=1, extent=(0, 1, 0, 1))
        axes[1, ci].set_xticks([]); axes[1, ci].set_yticks([])

        im2 = axes[2, ci].imshow(sharpness, origin="lower", cmap="cividis", vmin=0, vmax=1, extent=(0, 1, 0, 1))
        axes[2, ci].set_xticks([]); axes[2, ci].set_yticks([])

    axes[0, 0].set_ylabel("argmax(alphak)\n(-1 = NaN)", fontsize=9)
    axes[1, 0].set_ylabel(f"alphak[{args.disk}]\n(top disk)", fontsize=9)
    axes[2, 0].set_ylabel("sharpness\n=max-2nd_max", fontsize=9)

    cbar1 = fig.colorbar(im1, ax=axes[1, :], shrink=0.7, pad=0.02)
    cbar1.set_label(f"alphak[{args.disk}]")
    cbar2 = fig.colorbar(im2, ax=axes[2, :], shrink=0.7, pad=0.02)
    cbar2.set_label("sharpness")

    plt.suptitle(f"Zalesak 3-phase / 2-disk: alphak rendering at last finite frame", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
