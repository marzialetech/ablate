#!/usr/bin/env python3
"""
Post-process finger2d droplet-pinchoff runs into a dissertation-Fig-4.7-style
4-row x N-col grid.

For each Weber number row, picks snapshots at the requested timestamps
(default 0, 15, 30, 45, 60, 75 x 1e-3 s) and an additional pinchoff-onset
column.  Pinchoff onset is detected as the first frame where the alpha>0.5
mask has more connected components than the t=0 mask (so a child droplet has
appeared or the finger has split).

Usage
-----
    python tools/postprocess_finger2d.py \
        50:runs/droplet/finger2d/we50_s50 \
        30:runs/droplet/finger2d/we30_s83 \
        15:runs/droplet/finger2d/we15_s167 \
        10:runs/droplet/finger2d/we10_s250 \
        --out runs/droplet/finger2d/finger2d_fig47.png

Each positional argument is "WE:run_dir" where the title-based snapshot
folder is auto-discovered.

Optional flags:
    --times 0,15,30,45,60,75   # in milliseconds (default = dissertation Fig 4.7)
    --no-pinchoff               # skip the pinchoff-onset column

The colormap is a yellow-on-purple "viridis_r" tweak that visually matches
the published Fig 4.7 dual-tone style: alpha>0.5 (paraffin/finger) -> bright
yellow, alpha<=0.5 (gas) -> dark purple.
"""
from __future__ import annotations

import argparse
import os
import sys
from glob import glob
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def find_snapshots(run_dir: Path) -> list[Path]:
    """Snapshots are at <run_dir>/<title>/domain/domain.*.hdf5 where <title>
    is whatever environment.title was set to in the input.yaml.  We scan for
    any unique single-subdirectory match rather than hardcoding the title.
    """
    candidates = list(run_dir.glob("*/domain/domain.*.hdf5"))
    return sorted(candidates)


def load_snapshot(path: Path):
    with h5py.File(path, "r") as f:
        verts = f["geometry/vertices"][:]
        cells = f["viz/topology/cells"][:]
        cx = verts[cells, 0].mean(axis=1)
        cy = verts[cells, 1].mean(axis=1)
        vof = f["cell_fields/solution_volumeFraction"][0, :]
        t = float(f["time"][0, 0])
    return cx, cy, vof, t


def gridify(cx, cy, vof, nx=200, ny=200, lo=(0.0, 0.0), hi=(0.2, 0.2)):
    """Reshape unstructured-FV cell centroids into a regular nx by ny image
    by binning. The finger2d mesh is structured 200x200 so this is exact.
    """
    ix = np.clip(((cx - lo[0]) / (hi[0] - lo[0]) * nx).astype(int), 0, nx - 1)
    iy = np.clip(((cy - lo[1]) / (hi[1] - lo[1]) * ny).astype(int), 0, ny - 1)
    img = np.full((ny, nx), np.nan, dtype=float)
    img[iy, ix] = vof
    return img


def count_components(mask: np.ndarray) -> int:
    """4-connected component count via BFS (avoid scipy dependency)."""
    visited = np.zeros_like(mask, dtype=bool)
    n_comp = 0
    rows, cols = mask.shape
    for r in range(rows):
        for c in range(cols):
            if mask[r, c] and not visited[r, c]:
                n_comp += 1
                stack = [(r, c)]
                while stack:
                    rr, cc = stack.pop()
                    if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                        continue
                    if visited[rr, cc] or not mask[rr, cc]:
                        continue
                    visited[rr, cc] = True
                    stack.extend([(rr+1, cc), (rr-1, cc), (rr, cc+1), (rr, cc-1)])
    return n_comp


def pick_frame_at_time(snaps: list[Path], t_target_s: float):
    """Pick snapshot whose stored time is closest to t_target_s; return
    (path, t_actual) or (None, None) if outside range."""
    if not snaps:
        return None, None
    best, best_dt = None, None
    best_t = None
    for s in snaps:
        with h5py.File(s, "r") as f:
            t = float(f["time"][0, 0])
        dt = abs(t - t_target_s)
        if best is None or dt < best_dt:
            best, best_dt, best_t = s, dt, t
    return best, best_t


def detect_pinchoff(snaps: list[Path], baseline_components: int):
    """Walk snapshots in time order; return (path, t) of first frame with
    components > baseline_components.  None if no such frame in this run."""
    for s in snaps:
        cx, cy, vof, t = load_snapshot(s)
        img = gridify(cx, cy, vof)
        mask = (img > 0.5)
        c = count_components(mask)
        if c > baseline_components:
            return s, t
    return None, None


def make_dissertation_cmap():
    """Yellow-on-purple two-tone resembling dissertation Fig 4.7."""
    return LinearSegmentedColormap.from_list(
        "diss_finger",
        [(0.21, 0.07, 0.30), (0.99, 0.84, 0.18)],
        N=256,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows", nargs="+",
                    help="WE:run_dir entries, top-to-bottom row order; e.g. "
                         "50:runs/droplet/finger2d/we50_s50")
    ap.add_argument("--times", default="0,15,30,45,60,75",
                    help="comma-separated time values in milliseconds for the "
                         "uniformly-spaced columns (dissertation default).")
    ap.add_argument("--no-pinchoff", action="store_true",
                    help="omit the pinchoff-onset column.")
    ap.add_argument("--out", required=True, help="output PNG path.")
    ap.add_argument("--mesh-nx", type=int, default=200)
    ap.add_argument("--mesh-ny", type=int, default=200)
    args = ap.parse_args()

    rows = []
    for spec in args.rows:
        we_str, path = spec.split(":", 1)
        rows.append((float(we_str), Path(path)))

    times_ms = [float(x) for x in args.times.split(",")]
    times_s = [x * 1e-3 for x in times_ms]
    n_time_cols = len(times_s)
    n_cols = n_time_cols + (0 if args.no_pinchoff else 1)
    n_rows = len(rows)

    cmap = make_dissertation_cmap()
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(1.2 * n_cols + 0.6, 1.2 * n_rows + 0.6),
                             squeeze=False, gridspec_kw={"wspace": 0.05, "hspace": 0.05})

    for ri, (we, run_dir) in enumerate(rows):
        snaps = find_snapshots(run_dir)
        if not snaps:
            print(f"WARNING: no snapshots in {run_dir}", file=sys.stderr)
        # baseline component count from the t=0 frame
        if snaps:
            cx0, cy0, vof0, t0 = load_snapshot(snaps[0])
            img0 = gridify(cx0, cy0, vof0, args.mesh_nx, args.mesh_ny)
            baseline_comp = count_components(img0 > 0.5)
        else:
            baseline_comp = 1
        # uniform time columns
        for ci, t_s in enumerate(times_s):
            ax = axes[ri][ci]
            snap, t_actual = pick_frame_at_time(snaps, t_s)
            if snap is None:
                ax.text(0.5, 0.5, "no snapshot", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            cx, cy, vof, t = load_snapshot(snap)
            img = gridify(cx, cy, vof, args.mesh_nx, args.mesh_ny)
            ax.imshow(img, origin="lower", cmap=cmap, vmin=0, vmax=1,
                      extent=(0, 0.2, 0, 0.2), aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(rf"$10^3 t={times_ms[ci]:g}$", fontsize=9)
            if ci == 0:
                ax.set_ylabel(rf"We$={we:g}$", fontsize=10, rotation=0,
                              ha="right", va="center", labelpad=15)
        # pinchoff column
        if not args.no_pinchoff:
            ax = axes[ri][n_time_cols]
            print(f"[We={we}] detecting pinchoff...", flush=True)
            snap, t_p = detect_pinchoff(snaps, baseline_comp)
            if snap is None:
                ax.text(0.5, 0.5, "no pinchoff\nin range",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=8, color="white")
                ax.set_facecolor((0.21, 0.07, 0.30))
                ax.set_xticks([]); ax.set_yticks([])
            else:
                cx, cy, vof, _ = load_snapshot(snap)
                img = gridify(cx, cy, vof, args.mesh_nx, args.mesh_ny)
                ax.imshow(img, origin="lower", cmap=cmap, vmin=0, vmax=1,
                          extent=(0, 0.2, 0, 0.2), aspect="equal")
                ax.set_xticks([]); ax.set_yticks([])
                ax.text(0.96, 0.96, f"{t_p*1e3:.0f}",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=11, color="red", weight="bold")
            if ri == 0:
                ax.set_title("pinchoff\nonset", fontsize=9)

    fig.suptitle("finger2d droplet pinchoff (recon/dissertation-2026-04-30)\n"
                 "rows: We; columns: $10^3 t$ in s; right column: pinchoff onset",
                 fontsize=10)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
