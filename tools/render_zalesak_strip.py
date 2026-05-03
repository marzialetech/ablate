#!/usr/bin/env python3
"""Multi-frame horizontal strip of alphak argmax for a single Zalesak run.

Useful for inspecting interface evolution across an entire rotation in one
figure. Optionally compares against a second run (NOSHARP control) row-by-row.
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
    ap.add_argument("runs", nargs="+", type=Path, help="run directories (1+)")
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.labels is None:
        args.labels = [r.name for r in args.runs]
    assert len(args.labels) == len(args.runs)

    nrows = len(args.runs)
    rows_data = []
    for run in args.runs:
        snaps = discover_snapshots(run)
        if not snaps:
            raise SystemExit(f"no snapshots in {run}")
        if len(snaps) <= args.frames:
            chosen = snaps
        else:
            idx = np.linspace(0, len(snaps) - 1, args.frames).astype(int)
            chosen = [snaps[i] for i in idx]
        rows_data.append(chosen)

    ncols = max(len(r) for r in rows_data)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.4 * nrows), squeeze=False)

    for ri, (snaps, label) in enumerate(zip(rows_data, args.labels)):
        for ci in range(ncols):
            ax = axes[ri, ci]
            if ci < len(snaps):
                t, a, nx, ny = load_alphak(snaps[ci])
                amat = a.reshape(ny, nx, -1)
                argmax = np.argmax(amat, axis=2)
                finite = np.isfinite(amat).all(axis=2)
                argmax = np.where(finite, argmax, -1)
                ax.imshow(argmax, origin="lower", cmap="viridis", vmin=-1, vmax=amat.shape[-1] - 1, extent=(0, 1, 0, 1))
                step_num = int(snaps[ci].stem.split(".")[-1])
                ax.set_title(f"frame {step_num}\nt={t:.4f}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            if ci == 0:
                ax.set_ylabel(label, fontsize=10)

    plt.suptitle(f"Zalesak 3-phase / 2-disk: alphak argmax across rotation", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
