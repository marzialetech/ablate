#!/usr/bin/env python3
"""Side-by-side comparison plot of two slab runs at matching simulation times.

Designed to compare slab72-asis (Part 1, no fixes) against slab72-fixed
(Part 2, all corrections).  Produces a single PNG with two rows (asis on top,
fixed on bottom), each row showing the full domain at the final snapshot,
with the alpha=0.5 contour overlaid in black and the leading-edge anchor
(x_int, y_int) marked.
"""
import argparse
import os
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

# Make sure we can import postprocess_slab.compute_anchors_thesis
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from postprocess_slab import (
    compute_anchors_thesis,
    compute_theta_thesis,
    display_field,
    find_snapshots,
    load_snapshot,
)


def panel(ax, run_dir, label, x0, invert):
    snaps = find_snapshots(run_dir)
    if not snaps:
        ax.text(0.5, 0.5, f"no snapshots in {run_dir}",
                transform=ax.transAxes, ha="center")
        return
    x, y, vof, t = load_snapshot(snaps[-1])
    triang = tri.Triangulation(x, y)
    theta = compute_theta_thesis(triang, x0, vof)
    x_int, y_int, x_top, y_top = compute_anchors_thesis(triang, x0, vof)
    disp = display_field(vof, invert)
    ax.tricontourf(triang, disp, levels=20, cmap="coolwarm")
    ax.tricontour(triang, vof, levels=[0.5], colors="k", linewidths=1.2)
    if x_int is not None:
        ax.plot(x_int, y_int, "rx", ms=10, mew=2)
    ax.axvline(x0, color="y", lw=0.6, alpha=0.6)
    ax.set_xlim(0.0, 0.1)
    ax.set_ylim(0.0, 0.025)
    ax.set_aspect("auto")
    ax.set_xlabel("x [m]", fontsize=9)
    ax.set_ylabel("y [m]", fontsize=9)
    inv_note = "  (display: 1-alpha; lobe=red)" if invert else "  (display: alpha)"
    ax.set_title(rf"{label}   $t={t:.4e}$ s   "
                 rf"$\theta_{{\rm sim}}={theta:.2f}^\circ$" + inv_note,
                 fontsize=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("asis_dir", help="path to slab72-asis run directory")
    ap.add_argument("fixed_dir", help="path to slab72-fixed run directory")
    ap.add_argument("--out", default="slab72_asis_vs_fixed.png")
    ap.add_argument("--x0", type=float, default=25.5e-3)
    args = ap.parse_args()

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5),
                             gridspec_kw={"hspace": 0.5})
    fig.suptitle(
        rf"slab2d We=1000 reproduction - dissertation reports $\theta=29.43^\circ$ (Fig. 4.10)",
        fontsize=11,
    )
    panel(axes[0], args.asis_dir,
          "slab72-asis (Part 1; surface tension off, BCs inconsistent)",
          args.x0, invert=False)
    panel(axes[1], args.fixed_dir,
          "slab72-fixed (Part 2; sigma=50, BCs consistent, level-set sign fixed, smooth ramp, Cv corrected)",
          args.x0, invert=True)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.06)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
