"""Time evolution of the alpha=0.5 contour (red on white) for this run."""
from __future__ import annotations
import os, sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/Users/jjmarzia/Downloads/marzialeIP/audit")

import ablate_io as aio
from finger2d_metrics import _resample_to_uniform


CASE_DOMAIN = os.path.join(HERE, "domain")
OUT = os.path.join(HERE, "alpha05_evolution.png")
TARGET_TIMES_MS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]


case = aio.AblateCase(CASE_DOMAIN)
steps = list(case.steps())
if not steps:
    raise RuntimeError(f"No dumps in {CASE_DOMAIN}")

step_times_ms = np.array([s.time * 1e3 for s in steps])
max_t_ms = float(step_times_ms.max())

target_times_ms = [t for t in TARGET_TIMES_MS if t <= max_t_ms + 0.05]
ncols = len(target_times_ms)

fig, axes = plt.subplots(1, ncols, figsize=(2.4 * ncols, 3.4), squeeze=False)

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

    ax.set_facecolor("white")
    ax.contour(Xg, Yg, Vg, levels=[0.5], colors="red", linewidths=1.4)
    ax.set_xlim(0.0, 0.2)
    ax.set_ylim(0.0, 0.2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
fig.savefig(OUT, dpi=140, bbox_inches="tight", facecolor="white")
print(f"[fig] {OUT}")
