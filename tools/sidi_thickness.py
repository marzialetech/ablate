#!/usr/bin/env python3
"""SIDI interface-thickness diagnostic.

For each snapshot of an n-phase SIDI run with circular disk(s), compute the
characteristic interface thickness per disk via dissertation Eq. 2.20:

    eps_char_k = |r_{0.9}^k - r_{0.1}^k| / (4 ln 3)

where r_c^k is the radial distance from disk-k center to the alpha_k = c
contour. The contour is extracted with skimage.measure.find_contours on the
2D cell-centered alpha_k field and averaged over its sample points.

Plots eps_char(t)/h per disk per Gamma run on a single figure; expected
behavior is monotonic decay from ~ eps0/h to ~ 1.

Usage:
    python3 tools/sidi_thickness.py \\
        runs/sidi/sidi_4phase_pm_g4e2 \\
        runs/sidi/sidi_4phase_pm_g5e1 \\
        runs/sidi/sidi_4phase_pm_g2e0 \\
        --labels "G=4e-2" "G=5e-1" "G=2.0" \\
        --centers 0.5,0.77 0.26617,0.365 0.73383,0.365 \\
        --h 0.005 \\
        --eps0-over-h 8 \\
        --out tools/figs/sidi_pm_gamma_sweep_thickness.pdf
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure


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


def load_alphak(snap: Path) -> tuple[float, np.ndarray]:
    with h5py.File(snap, "r") as h:
        a = np.asarray(h["cell_fields/solution_alphak"][0])  # (ncell, K)
        t = float(h["time"][0, 0])
    ncell, K = a.shape
    nx = int(round(math.sqrt(ncell)))
    if nx * nx != ncell:
        raise RuntimeError(f"non-square ncell={ncell}")
    return t, a.reshape(nx, nx, K)  # [iy, ix, k]


def contour_mean_radius(
    field2d: np.ndarray,
    level: float,
    xc: float,
    yc: float,
    domain_lower: tuple[float, float],
    domain_upper: tuple[float, float],
) -> float | None:
    """Return mean Euclidean distance from (xc,yc) to the field=level contour.

    field2d has shape (ny, nx) where row index i corresponds to y, column j to x
    (cell-centered, with grid spacing implied by domain_lower/upper).
    """
    ny, nx = field2d.shape
    fmin, fmax = float(np.nanmin(field2d)), float(np.nanmax(field2d))
    if not (fmin <= level <= fmax):
        return None
    contours = measure.find_contours(field2d, level)
    if not contours:
        return None
    radii: list[float] = []
    for c in contours:
        # c is (N, 2) array of (i_row, j_col) pixel indices in *grid* coords.
        i_row = c[:, 0]
        j_col = c[:, 1]
        x_phys = domain_lower[0] + (j_col + 0.5) / nx * (domain_upper[0] - domain_lower[0])
        y_phys = domain_lower[1] + (i_row + 0.5) / ny * (domain_upper[1] - domain_lower[1])
        radii.extend(np.sqrt((x_phys - xc) ** 2 + (y_phys - yc) ** 2).tolist())
    if not radii:
        return None
    return float(np.mean(radii))


def eps_char_per_disk(
    alpha_k: np.ndarray,  # (ny, nx)
    xc: float,
    yc: float,
    domain_lower: tuple[float, float],
    domain_upper: tuple[float, float],
    levels_high: float = 0.9,
    levels_low: float = 0.1,
) -> float | None:
    r_hi = contour_mean_radius(alpha_k, levels_high, xc, yc, domain_lower, domain_upper)
    r_lo = contour_mean_radius(alpha_k, levels_low, xc, yc, domain_lower, domain_upper)
    if r_hi is None or r_lo is None:
        return None
    return abs(r_lo - r_hi) / (4.0 * math.log(3.0))


def trajectory_for_run(
    run_dir: Path,
    centers: list[tuple[float, float]],
    domain_lower: tuple[float, float],
    domain_upper: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (times[Nt], eps_char[Nt, Ndisks]) for run_dir."""
    snaps = discover_snapshots(run_dir)
    if not snaps:
        raise FileNotFoundError(f"no snapshots in {run_dir}")
    times = []
    eps_grid = np.full((len(snaps), len(centers)), np.nan)
    for ti, snap in enumerate(snaps):
        t, alpha_xyk = load_alphak(snap)
        times.append(t)
        for k, (xc, yc) in enumerate(centers):
            ek = eps_char_per_disk(alpha_xyk[:, :, k], xc, yc, domain_lower, domain_upper)
            if ek is not None:
                eps_grid[ti, k] = ek
    return np.array(times), eps_grid


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
    ap.add_argument("--h", type=float, required=True, help="grid spacing magnitude (for eps_char/h normalization)")
    ap.add_argument(
        "--eps0-over-h",
        type=float,
        default=None,
        help="initial diffuse-band thickness eps0/h (just for an annotation line; informational)",
    )
    ap.add_argument(
        "--lower", nargs=2, type=float, default=[0.0, 0.0], help="domain lower-left (xmin ymin)"
    )
    ap.add_argument(
        "--upper", nargs=2, type=float, default=[1.0, 1.0], help="domain upper-right (xmax ymax)"
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    if len(args.labels) != len(args.runs):
        raise SystemExit("--labels length must match number of runs")

    centers: list[tuple[float, float]] = []
    for cs in args.centers:
        a, b = cs.split(",")
        centers.append((float(a), float(b)))

    domain_lower = (args.lower[0], args.lower[1])
    domain_upper = (args.upper[0], args.upper[1])

    all_results: list[tuple[str, np.ndarray, np.ndarray]] = []  # label, t, eps_grid
    for run, label in zip(args.runs, args.labels):
        print(f"[{label}] processing {run} ...")
        t, eps_grid = trajectory_for_run(run, centers, domain_lower, domain_upper)
        all_results.append((label, t, eps_grid))
        for k in range(eps_grid.shape[1]):
            ts_str = ", ".join(
                f"{tt:.3f}={eps_grid[i,k]/args.h:.2f}" for i, tt in enumerate(t) if np.isfinite(eps_grid[i, k])
            )
            print(f"  disk{k}: eps_char/h(t) = [{ts_str}]")

    # Plot: one panel per disk, lines per Gamma run.
    Nk = len(centers)
    fig, axes = plt.subplots(1, Nk, figsize=(4.5 * Nk, 4.0), squeeze=False, sharey=True)
    cmap = plt.get_cmap("viridis")
    colors = [cmap(0.15 + 0.7 * i / max(1, len(all_results) - 1)) for i in range(len(all_results))]

    for k in range(Nk):
        ax = axes[0, k]
        for ci, (label, t, eps_grid) in enumerate(all_results):
            ax.plot(t, eps_grid[:, k] / args.h, "-o", color=colors[ci], label=label, ms=4, lw=1.5)
        ax.axhline(1.0, color="0.4", ls="--", lw=1, label=r"target $\varepsilon_{char}/h = 1$")
        if args.eps0_over_h is not None:
            ax.axhline(args.eps0_over_h, color="0.7", ls=":", lw=1, label=rf"IC $\varepsilon_0/h = {args.eps0_over_h:.0f}$")
        ax.set_xlabel("t")
        if k == 0:
            ax.set_ylabel(r"$\varepsilon_{\mathrm{char}} / h$")
        xc, yc = centers[k]
        ax.set_title(rf"disk {k}, center $({xc:.3f},{yc:.3f})$", fontsize=10)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        if k == Nk - 1:
            ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        rf"SIDI: characteristic interface thickness vs time ($h={args.h:g}$)", fontsize=12
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
