#!/usr/bin/env python3
"""Composite Weber-vs-receding-angle plot (dissertation Fig. 4.11).

Inputs are one or more slab run directories (each containing
slab2dcoords/domain/domain.*.hdf5). Either pass an explicit We per run
via --pairs "RUNDIR=WE", or let the tool extract `shearRate:` from the
run's input.yaml and compute We = rho_l * u0^2 * L / sigma with
defaults rho_l=1000, L=0.01, sigma=50.

For each run:
  - theta_extended = theta(t = t_final), i.e. the run's actual final time
                     (1.4e-3 s for the slab72-extended convention).
  - theta_diss     = theta interpolated at the dissertation t_f = 5e-4 s.

Both columns are written into a single tools/figs/we_vs_theta.pdf with the
dissertation interpolant theta_sim = 0.864 * We^0.506 + 0.845 overlaid for
reference.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from glob import glob
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

# Import compute_theta_thesis from postprocess_slab.py
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from postprocess_slab import compute_theta_thesis, load_snapshot  # type: ignore


SHEAR_RE = re.compile(r"shearRate:\s*([0-9.eE+\-]+)")


def we_from_yaml(yaml_path: Path, rho_l: float, L: float, sigma: float) -> float:
    txt = yaml_path.read_text()
    m = SHEAR_RE.search(txt)
    if not m:
        raise ValueError(f"no shearRate: in {yaml_path}")
    u0 = float(m.group(1))
    return rho_l * u0 * u0 * L / sigma


def find_snaps(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "slab2dcoords" / "domain").glob("domain.*.hdf5"))


def theta_trajectory(run_dir: Path, x0: float = 25.5e-3) -> tuple[np.ndarray, np.ndarray]:
    snaps = find_snaps(run_dir)
    times = []
    thetas = []
    for sp in snaps:
        x, y, vof, t = load_snapshot(str(sp))
        triang = tri.Triangulation(x, y)
        th = compute_theta_thesis(triang, x0, vof)
        times.append(t)
        thetas.append(th)
    return np.array(times), np.array(thetas)


def interp_theta(times: np.ndarray, thetas: np.ndarray, t_target: float) -> float:
    finite = np.isfinite(thetas)
    if not finite.any():
        return float("nan")
    times_f = times[finite]
    thetas_f = thetas[finite]
    if t_target <= times_f.min():
        return float(thetas_f[np.argmin(times_f)])
    if t_target >= times_f.max():
        return float(thetas_f[np.argmax(times_f)])
    return float(np.interp(t_target, times_f, thetas_f))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path, help="run directories (1+)")
    ap.add_argument(
        "--pairs",
        nargs="*",
        default=None,
        help="optional explicit RUNDIR=WE overrides (e.g. 'runs/slab/we10=10')",
    )
    ap.add_argument("--rho-l", type=float, default=1000.0)
    ap.add_argument("--L", type=float, default=0.01)
    ap.add_argument("--sigma", type=float, default=50.0)
    ap.add_argument("--diss-tf", type=float, default=5e-4, help="dissertation t_f")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--diss-points",
        nargs="+",
        type=float,
        default=None,
        help="dissertation Fig 4.10 (We, theta_deg) pairs interleaved as flat list",
    )
    ap.add_argument("--dpi", type=int, default=180)
    args = ap.parse_args()

    overrides: dict[str, float] = {}
    if args.pairs:
        for s in args.pairs:
            k, v = s.split("=")
            overrides[Path(k).resolve().as_posix()] = float(v)

    rows = []
    for run in args.runs:
        run = run.resolve()
        run_id = run.as_posix()
        if run_id in overrides:
            We = overrides[run_id]
        else:
            We = we_from_yaml(run / "input.yaml", args.rho_l, args.L, args.sigma)
        try:
            t, th = theta_trajectory(run)
        except FileNotFoundError as e:
            print(f"  {run.name}: SKIP ({e})")
            continue
        if len(t) == 0:
            print(f"  {run.name}: no snapshots")
            continue
        th_final = float(th[np.where(np.isfinite(th))[0][-1]]) if np.any(np.isfinite(th)) else float("nan")
        t_final = float(t[-1])
        th_diss = interp_theta(t, th, args.diss_tf)
        rows.append((run.name, We, t_final, th_final, th_diss))
        print(f"  {run.name}: We={We:8.2f}  t_final={t_final:.3e}s  theta_final={th_final:.3f}  theta@t_diss={th_diss:.3f}")

    rows.sort(key=lambda r: r[1])

    Wes = np.array([r[1] for r in rows])
    th_finals = np.array([r[3] for r in rows])
    th_disses = np.array([r[4] for r in rows])

    # Dissertation interpolant theta_sim = 0.864 * We^0.506 + 0.845 (Fig 4.11).
    We_curve = np.logspace(0, 3.2, 200)
    th_interp = 0.864 * We_curve ** 0.506 + 0.845

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        We_curve, th_interp,
        "r-", lw=1.5, alpha=0.7,
        label=r"diss. interpolant $\theta_{\mathrm{sim}}=0.864\,\mathrm{We}^{0.506}+0.845$ (Fig. 4.11)",
    )
    if args.diss_points:
        dp = np.array(args.diss_points).reshape(-1, 2)
        ax.plot(dp[:, 0], dp[:, 1], "rs", ms=7, mfc="white", mew=1.4, label="diss. Fig 4.10 raw points")
    ax.plot(Wes, th_finals, "ko-", ms=7, lw=1.4, label=rf"sim, $\theta(t_{{\mathrm{{final}}}})$ ($t_f = $ extended)")
    ax.plot(Wes, th_disses, "b^--", ms=7, lw=1.0, alpha=0.8, label=rf"sim, $\theta(t_{{\mathrm{{diss}}}}={args.diss_tf:g}\,\mathrm{{s}})$")

    ax.set_xscale("log")
    ax.set_xlabel(r"We")
    ax.set_ylabel(r"$\theta$ [deg]")
    ax.set_title("Receding angle vs leading-edge Weber number")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"\nwrote {args.out}")

    # Also write a CSV companion for archival.
    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w") as f:
        f.write("name,We,t_final,theta_final,theta_at_t_diss\n")
        for name, We, tF, thF, thD in rows:
            f.write(f"{name},{We:.6f},{tF:.6e},{thF:.6f},{thD:.6f}\n")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
