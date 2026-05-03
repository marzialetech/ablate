#!/usr/bin/env python3
"""Multi-frame strip rendering of the *continuous* component indicator

    I(x, y) = sum_k k * alpha_k(x, y)

for one or more Zalesak runs. For 3-phase / 2-disk runs phases are ordered
(top-disk = 0, bottom-disk = 1, background = 2), so I = 0 in the top disk,
I = 1 in the bottom disk, I = 2 in the background, with smooth interpolation
across mixed/diffused interfaces.

Optional: overlay the 0.5 contour of the *analytical* rigid rotation of the
initial Zalesak geometry (same as YAML IC: disk + notch), matching the
prescribed ZalesakTest velocity u = -ω(y-yc), v = ω(x-xc).

Uses pcolormesh on the cell-edge grid. Default cmap is inferno_r (reversed inferno).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import h5py
import matplotlib.lines as mlines
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


def load_indicator(snap: Path) -> tuple[float, np.ndarray, int, int, int]:
    """Return (t, indicator[ny, nx], nx, ny, nphases) where indicator = sum_k k*alpha_k."""
    with h5py.File(snap, "r") as h:
        t = float(h["time"][0, 0])
        a = np.asarray(h["cell_fields/solution_alphak"][0])  # (ncell, nphases)
    ncell, nphases = a.shape
    nx, ny = infer_grid(ncell)
    amat = a.reshape(ny, nx, nphases)
    ks = np.arange(nphases, dtype=np.float64)
    indicator = np.einsum("ijk,k->ij", amat, ks)  # ny x nx
    return t, indicator, nx, ny, nphases


def inverse_rotate_to_ic(
    x: np.ndarray,
    y: np.ndarray,
    t: float,
    omega: float,
    xc: float,
    yc: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map physical (x,y) at time t back to t=0 frame; same IC as forward CCW rotation."""
    c = math.cos(omega * t)
    s = math.sin(omega * t)
    x0 = xc + c * (x - xc) + s * (y - yc)
    y0 = yc - s * (x - xc) + c * (y - yc)
    return x0, y0


def analytical_alphak_perfect(
    x: np.ndarray,
    y: np.ndarray,
    t: float,
    omega: float,
    *,
    xc: float = 0.5,
    yc: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks (float 0/1) for phase 0 and 1 in reference frame, then... no, evaluated at physical x,y."""
    x0, y0 = inverse_rotate_to_ic(x, y, t, omega, xc, yc)
    # YAML Ic0  top disk
    in_disk0 = np.sqrt((x0 - 0.5) ** 2 + (y0 - 0.75) ** 2) < 0.15
    in_slit0 = (x0 >= 0.475) & (x0 <= 0.525) & (y0 <= 0.80)
    a0 = (in_disk0 & ~in_slit0).astype(np.float64)
    # YAML Ic1 bottom disk
    in_disk1 = np.sqrt((x0 - 0.5) ** 2 + (y0 - 0.25) ** 2) < 0.15
    in_slit1 = (x0 >= 0.475) & (x0 <= 0.525) & (y0 >= 0.20)
    a1 = (in_disk1 & ~in_slit1).astype(np.float64)
    return a0, a1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path, help="run directories (1+)")
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cmap", default="inferno")
    ap.add_argument(
        "--lower", nargs=2, type=float, default=[0.0, 0.0], help="domain lower-left (xmin ymin)"
    )
    ap.add_argument(
        "--upper", nargs=2, type=float, default=[1.0, 1.0], help="domain upper-right (xmax ymax)"
    )
    ap.add_argument("--dpi", type=int, default=180, help="raster fallback dpi")
    ap.add_argument(
        "--omega",
        type=float,
        default=30.0,
        help="Zalesak rigid-body rate (rad/s), must match YAML prescribing field",
    )
    ap.add_argument(
        "--rotation-center",
        nargs=2,
        type=float,
        default=[0.5, 0.5],
        help="(xc, yc) for rotation center",
    )
    ap.add_argument(
        "--truth-contour",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="overlay analytical alpha_k=0.5 contours (rigid rotation of IC)",
    )
    ap.add_argument(
        "--no-suptitle",
        action="store_true",
        help="omit the figure-level suptitle (cleaner look for figures destined for papers)",
    )
    ap.add_argument(
        "--truth-phases",
        type=str,
        default="0,1",
        help='comma-separated phase indices for analytical overlay (e.g. "0", "0,1")',
    )
    ap.add_argument(
        "--contour-color",
        nargs="+",
        default=["white"],
        help="line colors for each truth phase (cycles if short; default: white)",
    )
    ap.add_argument(
        "--contour-lw",
        type=float,
        default=1.25,
        help="linewidth for analytical contours",
    )
    ap.add_argument(
        "--contour-linestyle",
        default=":",
        help="matplotlib linestyle for analytical contours (default: ':' dotted)",
    )
    args = ap.parse_args()

    truth_phase_ids = [int(s.strip()) for s in args.truth_phases.split(",") if s.strip()]

    if args.labels is None:
        args.labels = [r.name for r in args.runs]
    if len(args.labels) != len(args.runs):
        raise SystemExit("--labels length must match number of runs")

    rx, ry = args.rotation_center

    nrows = len(args.runs)
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

    ncols = max(len(r) for r in rows_data)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.4 * ncols, 2.4 * nrows), squeeze=False
    )

    nphases_seen: set[int] = set()
    last_mesh = None

    for ri, (snaps, label) in enumerate(zip(rows_data, args.labels)):
        for ci in range(ncols):
            ax = axes[ri, ci]
            if ci < len(snaps):
                t, ind, nx, ny, nphases = load_indicator(snaps[ci])
                nphases_seen.add(nphases)
                xe = np.linspace(args.lower[0], args.upper[0], nx + 1)
                ye = np.linspace(args.lower[1], args.upper[1], ny + 1)
                xc = 0.5 * (xe[:-1] + xe[1:])
                yc_cent = 0.5 * (ye[:-1] + ye[1:])
                XX, YY = np.meshgrid(xc, yc_cent, indexing="xy")
                vmax = max(0.0, float(nphases - 1))
                last_mesh = ax.pcolormesh(
                    xe,
                    ye,
                    ind,
                    cmap=args.cmap,
                    vmin=0.0,
                    vmax=vmax,
                    shading="flat",
                    rasterized=True,
                )
                if args.truth_contour:
                    a0_p, a1_p = analytical_alphak_perfect(XX, YY, t, args.omega, xc=rx, yc=ry)
                    arrs = [a0_p, a1_p]
                    for j, k in enumerate(truth_phase_ids):
                        if k < 0 or k >= len(arrs):
                            continue
                        col = args.contour_color[j % len(args.contour_color)]
                        ax.contour(
                            XX,
                            YY,
                            arrs[k],
                            levels=[0.5],
                            colors=[col],
                            linewidths=args.contour_lw,
                            linestyles=args.contour_linestyle,
                        )
                ax.set_aspect("equal")
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

    if args.truth_contour and truth_phase_ids:
        handles = []
        for j, k in enumerate(truth_phase_ids):
            if k < 0 or k > 1:
                continue
            col = args.contour_color[j % len(args.contour_color)]
            handles.append(
                mlines.Line2D(
                    [], [],
                    color=col,
                    linewidth=args.contour_lw,
                    linestyle=args.contour_linestyle,
                    label=rf"analytical $\alpha_{k}=0.5$ (rigid IC)",
                )
            )
        if handles:
            fig.legend(
                handles=handles,
                loc="lower center",
                ncol=len(handles),
                fontsize=8,
                frameon=True,
            )

    if not args.no_suptitle:
        ttl = (
            r"Zalesak: $I = \sum_k k\,\alpha_k$ vs rigid-rotation IC"
            if args.truth_contour
            else r"Zalesak: continuous component indicator $I = \sum_k k\,\alpha_k$"
        )
        plt.suptitle(ttl + rf" ($\omega={args.omega}$ rad/s)", fontsize=11)
    fig.subplots_adjust(left=0.06, right=0.88, top=0.90, bottom=0.10, wspace=0.12, hspace=0.25)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
