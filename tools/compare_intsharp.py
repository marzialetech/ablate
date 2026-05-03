#!/usr/bin/env python3
"""
Compare an `nphase-zalesak-5phase` run with NPhaseIntSharp enabled against a
matching control run with NPhaseIntSharp disabled.

What this validates
-------------------
1. **Sharpness metric** (objective): for each phase k and snapshot t, count cells
   where ``0.05 < alphak[k] < 0.95``. This is the "diffuse-band cell count" --
   the number of cells where the volume fraction is partially mixed. Without
   sharpening, numerical advection diffuses interfaces over time and this
   count grows. Effective sharpening should keep it bounded or growing
   strictly slower than the control.

   We report:
     - ``total_band(t)``: sum across all phases.
     - ``per_phase_band(t, k)``: per phase (so we can see whether one disk is
       receiving disproportionate sharpening).

2. **Mass conservation**: sum_k alphak should remain 1.0 to roundoff. Any
   drift > 1e-8 is a bug.

3. **Visual side-by-side**: argmax-phase maps at t in {0, T/8, T/4} for both
   runs, so you can eyeball whether the disks are still recognizable.

Outputs
-------
Written under ``--out`` (default: parent of the intsharp dir):
- ``band_metric.png`` -- diffuse-band cell counts vs time, intsharp vs control.
- ``side_by_side_T0.png``, ``side_by_side_TmidEighth.png``, ``side_by_side_TquarterEnd.png``
  -- two-row argmax-phase panels (top: intsharp, bottom: control).
- ``summary.json`` -- machine-readable verdict.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def discover_snapshots(run_dir: Path) -> list[Path]:
    snaps = sorted((run_dir / "domain").glob("domain.*.hdf5"))
    if not snaps:
        raise FileNotFoundError(f"no HDF5 snapshots under {run_dir}/domain")
    return snaps


def read_snapshot(path: Path):
    with h5py.File(path, "r") as f:
        alphak = f["cell_fields/solution_alphak"][0]  # (ncell, nphase)
        time = float(f["time"][0, 0])
    return alphak, time


def infer_grid(ncell: int) -> tuple[int, int]:
    nx = int(round(math.sqrt(ncell)))
    if nx * nx != ncell:
        raise ValueError(f"{ncell} cells does not factor as a square mesh")
    return nx, nx


def diffuse_band_cells(alphak: np.ndarray, lo: float = 0.05, hi: float = 0.95) -> np.ndarray:
    """Per-phase count of cells with lo < alpha < hi."""
    in_band = (alphak > lo) & (alphak < hi)
    return in_band.sum(axis=0)  # (nphase,)


def collect_metrics(run_dir: Path) -> dict:
    snaps = discover_snapshots(run_dir)
    times = []
    band_per_phase = []
    band_total = []
    sum_min = []
    sum_max = []
    finite_frac = []
    nphase = None
    for snap in snaps:
        alphak, t = read_snapshot(snap)
        if nphase is None:
            nphase = alphak.shape[1]
        s = alphak.sum(axis=1)
        finite = np.isfinite(alphak).all(axis=1)
        band = diffuse_band_cells(alphak[finite] if finite.any() else alphak)
        times.append(t)
        band_per_phase.append(band)
        band_total.append(int(band.sum()))
        sum_min.append(float(s[finite].min()) if finite.any() else float("nan"))
        sum_max.append(float(s[finite].max()) if finite.any() else float("nan"))
        finite_frac.append(float(finite.mean()))
    return {
        "run_dir": str(run_dir),
        "times": np.asarray(times),
        "band_per_phase": np.asarray(band_per_phase),  # (nsnap, nphase)
        "band_total": np.asarray(band_total),
        "sum_min": np.asarray(sum_min),
        "sum_max": np.asarray(sum_max),
        "finite_frac": np.asarray(finite_frac),
        "snapshot_paths": snaps,
        "nphase": nphase,
    }


def plot_band_metric(intsharp: dict, control: dict, out: Path) -> None:
    fig, (ax_total, ax_per) = plt.subplots(1, 2, figsize=(14, 5))

    ax_total.plot(intsharp["times"], intsharp["band_total"], "-o", label="NPhaseIntSharp on", color="tab:blue")
    ax_total.plot(control["times"], control["band_total"], "-s", label="control (no NPhaseIntSharp)", color="tab:red")
    ax_total.set_xlabel("simulation time t [s]")
    ax_total.set_ylabel("# cells with 0.05 < alpha_k < 0.95  (summed over phases)")
    ax_total.set_title("Diffuse-band cell count (lower = sharper interface)")
    ax_total.grid(True, alpha=0.3)
    ax_total.legend(loc="best")

    nphase = intsharp["nphase"]
    cmap = plt.get_cmap("tab10")
    for k in range(nphase):
        ax_per.plot(intsharp["times"], intsharp["band_per_phase"][:, k], "-",
                    color=cmap(k), label=f"phase {k} (intsharp)")
        ax_per.plot(control["times"], control["band_per_phase"][:, k], "--",
                    color=cmap(k), label=f"phase {k} (control)")
    ax_per.set_xlabel("simulation time t [s]")
    ax_per.set_ylabel("# diffuse-band cells, per phase")
    ax_per.set_title("Per-phase diffuse-band counts (solid = intsharp, dashed = control)")
    ax_per.grid(True, alpha=0.3)
    ax_per.legend(loc="best", ncol=2, fontsize=8)

    fig.suptitle("NPhaseIntSharp validation: sharper run keeps the band lower", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def render_argmax_panel(ax, alphak: np.ndarray, nx: int, ny: int, title: str) -> None:
    s = alphak.sum(axis=1)
    finite = np.isfinite(alphak).all(axis=1) & (s > 1e-9)
    argmax = np.full(alphak.shape[0], -1, dtype=int)
    argmax[finite] = np.argmax(alphak[finite], axis=1)
    img = argmax.reshape(ny, nx)
    ax.imshow(img, origin="lower", cmap="tab10", vmin=-1, vmax=alphak.shape[1] - 1, aspect="equal")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def render_side_by_side(intsharp: dict, control: dict, fraction: float, label: str, out: Path) -> None:
    target_t = intsharp["times"][-1] * fraction
    i_idx = int(np.argmin(np.abs(intsharp["times"] - target_t)))
    c_idx = int(np.argmin(np.abs(control["times"] - target_t)))
    i_alpha, i_t = read_snapshot(intsharp["snapshot_paths"][i_idx])
    c_alpha, c_t = read_snapshot(control["snapshot_paths"][c_idx])

    nx, ny = infer_grid(i_alpha.shape[0])
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    render_argmax_panel(axes[0], i_alpha, nx, ny, f"NPhaseIntSharp on  |  t={i_t:.4f}s  ({label})")
    render_argmax_panel(axes[1], c_alpha, nx, ny, f"control (off)        |  t={c_t:.4f}s  ({label})")
    fig.suptitle(f"argmax phase per cell  ({label})")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def render_alpha_heatmap(intsharp: dict, control: dict, fraction: float, label: str, out: Path) -> None:
    """Per-phase alpha heatmaps side-by-side at fixed time."""
    target_t = intsharp["times"][-1] * fraction
    i_idx = int(np.argmin(np.abs(intsharp["times"] - target_t)))
    c_idx = int(np.argmin(np.abs(control["times"] - target_t)))
    i_alpha, i_t = read_snapshot(intsharp["snapshot_paths"][i_idx])
    c_alpha, c_t = read_snapshot(control["snapshot_paths"][c_idx])
    nx, ny = infer_grid(i_alpha.shape[0])
    nphase = i_alpha.shape[1]

    fig, axes = plt.subplots(2, nphase, figsize=(3 * nphase, 6.4))
    for k in range(nphase):
        axes[0, k].imshow(i_alpha[:, k].reshape(ny, nx), origin="lower", cmap="viridis", vmin=0, vmax=1, aspect="equal")
        axes[0, k].set_title(f"intsharp  phase {k}", fontsize=9)
        axes[0, k].set_xticks([]); axes[0, k].set_yticks([])
        axes[1, k].imshow(c_alpha[:, k].reshape(ny, nx), origin="lower", cmap="viridis", vmin=0, vmax=1, aspect="equal")
        axes[1, k].set_title(f"control  phase {k}", fontsize=9)
        axes[1, k].set_xticks([]); axes[1, k].set_yticks([])
    fig.suptitle(f"alphak heatmaps  |  t={i_t:.4f}s  ({label})", y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def verdict(intsharp: dict, control: dict) -> dict:
    end = -1
    band_int = int(intsharp["band_total"][end])
    band_ctl = int(control["band_total"][end])
    delta = band_ctl - band_int
    rel = delta / max(band_ctl, 1)
    finite_int = float(intsharp["finite_frac"].min())
    finite_ctl = float(control["finite_frac"].min())
    sum_drift_int = float(np.max(np.abs(np.r_[intsharp["sum_min"], intsharp["sum_max"]] - 1.0)))
    sum_drift_ctl = float(np.max(np.abs(np.r_[control["sum_min"], control["sum_max"]] - 1.0)))
    return {
        "intsharp_band_total_at_T": band_int,
        "control_band_total_at_T": band_ctl,
        "intsharp_minus_control_band": -delta,
        "relative_band_reduction": rel,
        "intsharp_min_finite_fraction": finite_int,
        "control_min_finite_fraction": finite_ctl,
        "intsharp_max_sum_drift": sum_drift_int,
        "control_max_sum_drift": sum_drift_ctl,
        "verdict": (
            "intsharp is sharper than control" if rel > 0.05
            else "intsharp is comparable to control" if abs(rel) <= 0.05
            else "intsharp is more diffuse than control (unexpected; check Gammak)"
        ),
        "notes": (
            "rel > 0.05 means intsharp produced at least 5% fewer diffuse-band cells "
            "at the final snapshot; this is the threshold for 'nonnegligible sharpening'."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intsharp", required=True, type=Path, help="intsharp run dir (the one containing domain/)")
    parser.add_argument("--control", required=True, type=Path, help="control (no-intsharp) run dir")
    parser.add_argument("--out", type=Path, default=None, help="output dir (default: parent of intsharp)")
    args = parser.parse_args()

    out = args.out or args.intsharp.parent / "compare"
    out.mkdir(parents=True, exist_ok=True)

    intsharp = collect_metrics(args.intsharp)
    control = collect_metrics(args.control)

    plot_band_metric(intsharp, control, out / "band_metric.png")
    render_side_by_side(intsharp, control, 0.00, "t=0",     out / "side_by_side_t00.png")
    render_side_by_side(intsharp, control, 0.25, "T/4 = 90°",  out / "side_by_side_t25.png")
    render_side_by_side(intsharp, control, 0.50, "T/2 = 180°", out / "side_by_side_t50.png")
    render_side_by_side(intsharp, control, 0.75, "3T/4 = 270°", out / "side_by_side_t75.png")
    render_side_by_side(intsharp, control, 1.00, "T = 360°",   out / "side_by_side_t100.png")
    render_alpha_heatmap(intsharp, control, 1.00, "T = 360°", out / "alphak_heatmap_T.png")

    summary = verdict(intsharp, control)
    summary["intsharp_run_dir"] = str(args.intsharp)
    summary["control_run_dir"] = str(args.control)
    summary["nsnap"] = int(len(intsharp["times"]))
    summary["t_max"] = float(intsharp["times"][-1])
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"[compare] verdict: {summary['verdict']}")
    print(f"  band_total at T:  intsharp={summary['intsharp_band_total_at_T']:>7d}   control={summary['control_band_total_at_T']:>7d}")
    print(f"  rel reduction:    {summary['relative_band_reduction']:+.3%}")
    print(f"  finite cells min (intsharp): {summary['intsharp_min_finite_fraction']:.6f}")
    print(f"  finite cells min (control):  {summary['control_min_finite_fraction']:.6f}")
    print(f"  max |sum(alpha)-1| (intsharp): {summary['intsharp_max_sum_drift']:.3e}")
    print(f"  max |sum(alpha)-1| (control):  {summary['control_max_sum_drift']:.3e}")
    print(f"[compare] wrote outputs to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
