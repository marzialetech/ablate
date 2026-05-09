"""Time evolution of the vof field with the final-equilibrium radius overlaid."""
from __future__ import annotations
import os, sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/Users/jjmarzia/Downloads/marzialeIP/audit")

import ablate_io as aio
from finger2d_metrics import _resample_to_uniform


CASE_DOMAIN = os.path.join(HERE, "domain")
OUT = os.path.join(HERE, "vof_evolution.png")
TARGET_TIMES_MS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]


case = aio.AblateCase(CASE_DOMAIN)
steps = list(case.steps())
if not steps:
    raise RuntimeError(f"No dumps in {CASE_DOMAIN}")

step_times_ms = np.array([s.time * 1e3 for s in steps])
max_t_ms = float(step_times_ms.max())
target_times_ms = [t for t in TARGET_TIMES_MS if t <= max_t_ms + 0.1]


def alpha_band_radius(step, lo=0.4, hi=0.6):
    xy = step.cellxy()
    a  = step.vof()
    mi = step.interior_mask()
    xyi = xy[mi]; ai = a[mi]
    sel = (ai > lo) & (ai < hi)
    if not sel.any():
        return np.nan, np.nan
    rs = np.sqrt(xyi[sel, 0] ** 2 + xyi[sel, 1] ** 2)
    return float(rs.mean()), float(rs.std())


R_final, R_std = alpha_band_radius(steps[-1])
print(f"final equilibrium radius: R = {R_final * 1e3:.3f} mm (std {R_std * 1e3:.4f} mm) at t = {steps[-1].time * 1e3:.3f} ms")

ncols = len(target_times_ms)
fig, axes = plt.subplots(1, ncols, figsize=(2.4 * ncols, 3.4), squeeze=False)

theta = np.linspace(0, 2 * np.pi, 361)
xc = R_final * np.cos(theta)
yc = R_final * np.sin(theta)

for ax, t_target in zip(axes[0], target_times_ms):
    idx = int(np.argmin(np.abs(step_times_ms - t_target)))
    s = steps[idx]
    xy = s.cellxy()
    a  = s.vof()
    mi = s.interior_mask()
    xy_i = xy[mi]
    a_i  = a[mi]
    h = aio.estimate_h(xy_i) or 1e-3
    Xg, Yg, Vg, _, _ = _resample_to_uniform(xy_i, a_i, h=h)
    Vg = np.where(np.isfinite(Vg), Vg, 0.0)
    extent = [xy_i[:, 0].min(), xy_i[:, 0].max(),
              xy_i[:, 1].min(), xy_i[:, 1].max()]

    ax.imshow(Vg, origin="lower", extent=extent, vmin=0, vmax=1,
              cmap="Blues_r", aspect="auto")
    ax.plot(xc, yc, color="red", lw=1.4, ls="--")
    ax.set_xlim(-0.035, 0.035)
    ax.set_ylim(-0.035, 0.035)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
fig.savefig(OUT, dpi=140, bbox_inches="tight", facecolor="white")
print(f"[fig] {OUT}")
