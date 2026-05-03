#!/usr/bin/env python3
"""SIDI multi-frame strip renderer.

For one or more SIDI runs (4-phase, 3 disks 120 deg apart), render the
continuous component indicator I(x,y) = sum_k k*alpha_k(x,y) at K equally
spaced frames. Inferno_r colormap. Overlays the analytical disk-edge (alpha=0.5
contour for each disk in the IC) as a white dotted circle since the IC is
static in time.
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


def load_indicator(snap: Path) -> tuple[float, np.ndarray, int, int, int]:
    with h5py.File(snap, "r") as h:
        t = float(h["time"][0, 0])
        a = np.asarray(h["cell_fields/solution_alphak"][0])  # (ncell, K)
    ncell, K = a.shape
    nx = int(round(math.sqrt(ncell)))
    if nx * nx != ncell:
        raise RuntimeError(f"non-square ncell={ncell}")
    amat = a.reshape(nx, nx, K)
    ks = np.arange(K, dtype=np.float64)
    indicator = np.einsum("ijk,k->ij", amat, ks)
    return t, indicator, nx, nx, K


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path, help="run directories (1+)")
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument(
        "--centers",
        nargs="+",
        required=True,
        help="comma-separated 'x,y' centers for each disk, e.g. '0.5,0.77' '0.27,0.37' '0.73,0.37'",
    )
    ap.add_argument("--R", type=float, required=True, help="disk radius for analytical overlay contour")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument(
        "--lower", nargs=2, type=float, default=[0.0, 0.0],
    )
    ap.add_argument(
        "--upper", nargs=2, type=float, default=[1.0, 1.0],
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cmap", default="inferno_r")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    if len(args.labels) != len(args.runs):
        raise SystemExit("--labels length must match number of runs")

    centers: list[tuple[float, float]] = []
    for cs in args.centers:
        a, b = cs.split(",")
        centers.append((float(a), float(b)))

    rows_data: list[list[Path]] = []
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

    nrows = len(args.runs)
    ncols = max(len(r) for r in rows_data)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.4 * ncols, 2.4 * nrows), squeeze=False
    )

    nphases_seen: set[int] = set()
    last_mesh = None

    theta = np.linspace(0.0, 2.0 * math.pi, 200)

    for ri, (snaps, label) in enumerate(zip(rows_data, args.labels)):
        for ci in range(ncols):
            ax = axes[ri, ci]
            if ci < len(snaps):
                t, ind, nx, ny, K = load_indicator(snaps[ci])
                nphases_seen.add(K)
                xe = np.linspace(args.lower[0], args.upper[0], nx + 1)
                ye = np.linspace(args.lower[1], args.upper[1], ny + 1)
                last_mesh = ax.pcolormesh(
                    xe, ye, ind,
                    cmap=args.cmap,
                    vmin=0.0,
                    vmax=max(0.0, K - 1),
                    shading="flat",
                    rasterized=True,
                )
                # Static analytical disk contours (white dotted).
                for (xc, yc) in centers:
                    ax.plot(
                        xc + args.R * np.cos(theta),
                        yc + args.R * np.sin(theta),
                        color="white",
                        linestyle=":",
                        linewidth=1.0,
                    )
                ax.set_aspect("equal")
                ax.set_xlim(args.lower[0], args.upper[0])
                ax.set_ylim(args.lower[1], args.upper[1])
                step_num = int(snaps[ci].stem.split(".")[-1])
                ax.set_title(f"frame {step_num}\nt={t:.4f}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            if ci == 0:
                ax.set_ylabel(label, fontsize=10)

    if last_mesh is not None:
        cax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
        cbar = fig.colorbar(last_mesh, cax=cax)
        if len(nphases_seen) == 1:
            n = nphases_seen.pop()
            cbar.set_ticks(list(range(n)))
        cbar.set_label(r"$\sum_k k\,\alpha_k$", fontsize=10)

    plt.suptitle(
        r"SIDI: $I = \sum_k k\,\alpha_k$ vs time (white dotted = analytical $\alpha=0.5$ disk circles, static IC)",
        fontsize=11,
    )
    fig.subplots_adjust(left=0.06, right=0.88, top=0.90, bottom=0.10, wspace=0.12, hspace=0.25)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
