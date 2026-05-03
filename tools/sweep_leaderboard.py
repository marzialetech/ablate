#!/usr/bin/env python3
"""
Compare a set of NPhaseIntSharp sweep variants at a single time-of-interest
(quarter rotation, t=pi/2/omega) and produce a leaderboard.

Metrics
-------
- ``diffuse_band``: # cells with 0.05 < alpha_k < 0.95, summed over phases.
  Lower = sharper magnitude transitions.
- ``max_alpha_disk``: max alpha_k (any disk phase 0..3) over all cells.
  1.0 = pure-disk cores survive; <1.0 = cores are bleeding to background.
- ``slot_max_alpha_bg``: for each disk k in {0..3}, find the cells where
  argmax==k, then take max alpha_4 over those cells. If the Zalesak slot
  inside disk k is preserved, alpha_4 should approach 1 there. Reported as
  the average over the four disks. **This is the slot-preservation metric.**
- ``slot_count_bg``: total # cells with (argmax==k for k in 0..3) AND
  (alpha_4 > 0.5). The integer "slot mass" across all four disks.
- ``mass_drift``: max |sum_k alpha_k - 1| over the snapshot.
- ``finite_frac``: fraction of cells with all-finite alpha.

Inputs
------
A list of run directories (each containing a ``domain/`` folder with HDF5
snapshots). The script picks the snapshot whose time is closest to
``--target-t``. By default ``--target-t = pi/(2*30) = 0.05236s`` (90 deg
rotation at omega=30).

Outputs
-------
- ``leaderboard.json`` -- machine-readable table.
- ``leaderboard.png`` -- visual side-by-side of phase-4 (background)
  alpha_4 heatmaps at the target time, one panel per variant.
- ``leaderboard.md`` -- markdown table.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import convex_hull_image


def discover_snapshots(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "domain").glob("domain.*.hdf5"))


def read_snapshot(path: Path):
    with h5py.File(path, "r") as f:
        alphak = f["cell_fields/solution_alphak"][0]
        time = float(f["time"][0, 0])
    return alphak, time


def infer_grid(ncell: int) -> tuple[int, int]:
    nx = int(round(math.sqrt(ncell)))
    if nx * nx != ncell:
        raise ValueError(f"{ncell} cells does not factor as a square mesh")
    return nx, nx


def pick_snapshot(run_dir: Path, target_t: float) -> tuple[Path, float]:
    snaps = discover_snapshots(run_dir)
    if not snaps:
        raise FileNotFoundError(f"no snapshots under {run_dir}/domain")
    times = []
    for s in snaps:
        with h5py.File(s, "r") as f:
            times.append(float(f["time"][0, 0]))
    idx = int(np.argmin(np.abs(np.asarray(times) - target_t)))
    return snaps[idx], times[idx]


def compute_metrics(alphak: np.ndarray) -> dict:
    """All metrics for a single snapshot.

    Slot metric (the one we actually care about):
      The Zalesak slot is an *open* notch, not an enclosed hole, so
      ``binary_fill_holes`` doesn't see it. Instead, for each disk phase
      ``k`` we take the 2D convex hull of the ``argmax==k`` mask and
      report ``hull.sum() - mask.sum()`` -- the number of cells inside
      the disk's bounding silhouette where the disk does NOT win argmax.
      For a perfectly circular disk this is zero (or close to it, modulo
      mesh discretization error of the hull); for a Zalesak disk with the
      slot intact it is the slot area. Sum over the four disks.
    """
    s = alphak.sum(axis=1)
    finite = np.isfinite(alphak).all(axis=1) & (s > 1e-9)
    nphase = alphak.shape[1]

    in_band = (alphak > 0.05) & (alphak < 0.95)
    diffuse_band = int(in_band.sum())

    disk_max = float(max(alphak[finite, k].max() for k in range(nphase - 1))) if finite.any() else float("nan")

    argmax = np.full(alphak.shape[0], -1, dtype=int)
    argmax[finite] = np.argmax(alphak[finite], axis=1)

    nx, ny = infer_grid(alphak.shape[0])
    argmax2d = argmax.reshape(ny, nx)
    bg = nphase - 1

    slot_hole_per_disk = []
    slot_hole_total = 0
    for k in range(nphase - 1):
        mask = argmax2d == k
        if not mask.any():
            slot_hole_per_disk.append(0)
            continue
        hull = convex_hull_image(mask)
        notch = hull & ~mask  # cells inside the disk's silhouette where it doesn't win argmax
        notch_count = int(notch.sum())
        slot_hole_per_disk.append(notch_count)
        slot_hole_total += notch_count

    mass_drift = float(np.max(np.abs(s[finite] - 1.0))) if finite.any() else float("nan")
    finite_frac = float(finite.mean())
    ncells = alphak.shape[0]
    slot_fraction = slot_hole_total / ncells
    diffuse_fraction = diffuse_band / ncells
    return {
        "diffuse_band": diffuse_band,
        "diffuse_fraction": diffuse_fraction,
        "max_alpha_disk": disk_max,
        "slot_hole_total": slot_hole_total,
        "slot_hole_per_disk": slot_hole_per_disk,
        "slot_fraction": slot_fraction,
        "mass_drift": mass_drift,
        "finite_frac": finite_frac,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-t", type=float, default=math.pi / (2 * 30.0),
                        help="time-of-interest in seconds (default = quarter rotation at omega=30)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("variants", nargs="+",
                        help="LABEL=PATH pairs, e.g. baseline=runs/plan-a-full-intsharp/nphase-zalesak-5phase_2026-04-30T22-30-44")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    snapshots = []
    for v in args.variants:
        if "=" not in v:
            raise SystemExit(f"variant must be LABEL=PATH; got {v!r}")
        label, _, path = v.partition("=")
        run = Path(path).expanduser()
        snap, t = pick_snapshot(run, args.target_t)
        alphak, _ = read_snapshot(snap)
        m = compute_metrics(alphak)
        m["label"] = label
        m["snapshot"] = str(snap)
        m["t"] = t
        m["ncell"] = alphak.shape[0]
        rows.append(m)
        snapshots.append((label, alphak, t))
        print(f"[sweep] {label:<24s} t={t:.5f}s diffuse_band={m['diffuse_band']:>7d}  "
              f"slot_hole={m['slot_hole_total']:>5d}  "
              f"max_alpha_disk={m['max_alpha_disk']:.4f}  "
              f"per-disk-holes={m['slot_hole_per_disk']}")

    # Visualization: phase-4 (background) heatmap per variant.
    n = len(snapshots)
    cols = min(n, 4)
    rows_plot = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_plot, cols, figsize=(4.0 * cols, 4.0 * rows_plot), squeeze=False)
    for i, (label, alphak, t) in enumerate(snapshots):
        nphase = alphak.shape[1]
        nx, ny = infer_grid(alphak.shape[0])
        bg = nphase - 1
        ax = axes[i // cols, i % cols]
        ax.imshow(alphak[:, bg].reshape(ny, nx), origin="lower", cmap="viridis",
                  vmin=0.0, vmax=1.0, aspect="equal")
        ax.set_title(f"{label}\nt={t:.4f}s, mesh={nx}x{ny}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(n, rows_plot * cols):
        axes[j // cols, j % cols].axis("off")
    fig.suptitle("alpha_4 (background) at quarter rotation -- slot-preservation comparison", y=1.0)
    fig.tight_layout()
    fig.savefig(args.out / "leaderboard.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Also dump argmax montage.
    fig, axes = plt.subplots(rows_plot, cols, figsize=(4.0 * cols, 4.0 * rows_plot), squeeze=False)
    for i, (label, alphak, t) in enumerate(snapshots):
        nphase = alphak.shape[1]
        nx, ny = infer_grid(alphak.shape[0])
        s = alphak.sum(axis=1)
        finite = np.isfinite(alphak).all(axis=1) & (s > 1e-9)
        argmax = np.full(alphak.shape[0], -1, dtype=int)
        argmax[finite] = np.argmax(alphak[finite], axis=1)
        ax = axes[i // cols, i % cols]
        ax.imshow(argmax.reshape(ny, nx), origin="lower", cmap="tab10",
                  vmin=-1, vmax=nphase - 1, aspect="equal")
        ax.set_title(f"{label}\nt={t:.4f}s, mesh={nx}x{ny}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(n, rows_plot * cols):
        axes[j // cols, j % cols].axis("off")
    fig.suptitle("argmax phase at quarter rotation -- topology comparison", y=1.0)
    fig.tight_layout()
    fig.savefig(args.out / "leaderboard_argmax.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    rows.sort(key=lambda r: -r["slot_fraction"])
    (args.out / "leaderboard.json").write_text(json.dumps(rows, indent=2))

    md = ["| rank | variant | mesh | slot% (mesh-indep) | diffuse% | slot_hole | diffuse_band | max_alpha_disk | mass_drift |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for rank, r in enumerate(rows, 1):
        nx = int(round(math.sqrt(r["ncell"])))
        md.append(f"| {rank} | {r['label']} | {nx}x{nx} | "
                  f"{r['slot_fraction']*100:.3f}% | {r['diffuse_fraction']*100:.2f}% | "
                  f"{r['slot_hole_total']:,} | {r['diffuse_band']:,} | "
                  f"{r['max_alpha_disk']:.4f} | {r['mass_drift']:.2e} |")
    (args.out / "leaderboard.md").write_text("\n".join(md) + "\n")

    print(f"[sweep] wrote {args.out / 'leaderboard.json'} and leaderboard.{{md,png,argmax.png}}")
    print("[sweep] ranked by slot_fraction (higher = more slot survives, mesh-independent):")
    for rank, r in enumerate(rows, 1):
        nx = int(round(math.sqrt(r["ncell"])))
        print(f"  {rank}. {r['label']:<22s} mesh={nx}x{nx}  "
              f"slot={r['slot_fraction']*100:.3f}%  diffuse={r['diffuse_fraction']*100:.2f}%  "
              f"max_alpha_disk={r['max_alpha_disk']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
