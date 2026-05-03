#!/usr/bin/env python3
"""Overlay theta(t) trajectories from any number of slab runs.

Used to synthesize the asis -> fixed -> extended progression in one plot.
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

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from postprocess_slab import (
    compute_theta_thesis,
    detect_equilibrium,
    find_snapshots,
    load_snapshot,
)


def trajectory(run_dir, x0):
    snaps = find_snapshots(run_dir)
    times, thetas = [], []
    for p in snaps:
        cx, cy, vof, t = load_snapshot(p)
        triang = tri.Triangulation(cx, cy)
        th = compute_theta_thesis(triang, x0, vof)
        times.append(t)
        thetas.append(th)
    return np.array(times), np.array(thetas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+",
                    help="run directories (dir or label=dir); plotted in argv order")
    ap.add_argument("--out", default="theta_overlay.png")
    ap.add_argument("--x0", type=float, default=25.5e-3)
    ap.add_argument("--tau-adv", type=float, default=1.41e-4,
                    help="advective timescale for equilibrium check (default: "
                         "L_lobe/U_shear = 0.01/70.71 = 1.41e-4 s).")
    ap.add_argument("--diss-tf", type=float, default=5.0e-4)
    ap.add_argument("--diss-theta", type=float, default=29.43)
    ap.add_argument("--detect-equilibrium", action="store_true")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    parsed = []
    for spec in args.runs:
        if "=" in spec:
            label, path = spec.split("=", 1)
        else:
            label, path = os.path.basename(os.path.normpath(spec)), spec
        parsed.append((label, path))

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    for k, (label, run_dir) in enumerate(parsed):
        times, thetas = trajectory(run_dir, args.x0)
        c = colors[k % len(colors)]
        ax.plot(times, thetas, "o-", color=c, ms=3, lw=1.2,
                label=f"{label}  (final theta = {thetas[-1]:.2f} deg @ "
                      f"t = {times[-1]:.2e} s)")
        if args.detect_equilibrium:
            det = detect_equilibrium(times, thetas, tau_adv=args.tau_adv)
            if det["converged"]:
                ax.axvline(det["t_converged"], color=c, ls="--", lw=1.0, alpha=0.6)
                print(f"[{label}] {det['message']}")
            else:
                print(f"[{label}] {det['message']}")

    ax.axhline(args.diss_theta, color="k", ls="--", lw=1.1, alpha=0.7,
               label=f"dissertation Fig. 4.10 @ We=1000: {args.diss_theta} deg")
    ax.axvline(args.diss_tf, color="0.5", ls=":", lw=1.0, alpha=0.7,
               label=rf"dissertation $t_f = {args.diss_tf:g}$ s")
    ax.set_xlabel("time [s]")
    ax.set_ylabel(r"$\theta$ [deg]")
    if args.title is None:
        args.title = "slab2d We=1000: receding-angle trajectories"
    ax.set_title(args.title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
