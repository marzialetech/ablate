#!/usr/bin/env python3
"""receding-angle geometry kernel for the slab2d case.

extracted verbatim from the dissertation notebook ch4/lobe.ipynb
(`compute_theta_from_vf` cell). only the helpers required by
`animate_slab.py` are included here:

    _thesis_y_at_x0
    _thesis_vertex_on_contour
    compute_anchors_thesis
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _thesis_y_at_x0(paths, x0):
    """Topmost alpha=0.5 crossing on the vertical line x = x0, or None."""
    candidates = []
    for path in paths:
        v = path.vertices
        if len(v) < 2:
            continue
        xs = v[:, 0]
        ys = v[:, 1]
        d = xs - x0
        for i in range(len(v) - 1):
            di, dj = d[i], d[i + 1]
            if di == 0.0:
                candidates.append(float(ys[i]))
            elif di * dj < 0.0:
                t = (x0 - xs[i]) / (xs[i + 1] - xs[i])
                candidates.append(float(ys[i] + t * (ys[i + 1] - ys[i])))
        if d[-1] == 0.0:
            candidates.append(float(ys[-1]))
    return max(candidates) if candidates else None


def _thesis_vertex_on_contour(paths, x_int, y_int,
                              x_search=1.0e-2, fit_half_window=2.0e-3):
    """Thesis-faithful vertex: point (xv, yv) on the alpha=0.5 contour where
    `partial_alpha/partial_x = 0`, i.e. where the contour tangent is locally
    horizontal (|dy/dx| -> 0).

    Strategy:
      1. Scan every contour-polyline segment whose midpoint lies in the
         search window  x in (x_int, x_int + x_search],  y > y_int  (restricts
         to the upper side of the lobe).
      2. Track the segment with minimum |dy/dx|.
      3. Refine sub-grid: fit y = a*x^2 + b*x + c to all contour points within
         +/- fit_half_window in x of that segment's midpoint, return
         (-b/2a, c - b^2/4a). Falls back to the segment midpoint if the fit
         is degenerate.
    Returns (xv, yv) or (None, None).
    """
    best = None
    for path in paths:
        v = path.vertices
        if len(v) < 2:
            continue
        for i in range(len(v) - 1):
            x1, y1 = float(v[i, 0]), float(v[i, 1])
            x2, y2 = float(v[i + 1, 0]), float(v[i + 1, 1])
            xm = 0.5 * (x1 + x2)
            ym = 0.5 * (y1 + y2)
            if not (x_int < xm <= x_int + x_search):
                continue
            if ym <= y_int:
                continue
            dxs = x2 - x1
            if abs(dxs) < 1.0e-15:
                continue
            ab = abs((y2 - y1) / dxs)
            if best is None or ab < best[0]:
                best = (ab, xm, ym)
    if best is None:
        return None, None
    _, xm, ym = best

    near = []
    for path in paths:
        for px, py in path.vertices:
            if abs(px - xm) <= fit_half_window:
                near.append((float(px), float(py)))
    if len(near) < 3:
        return xm, ym
    near = np.array(near)
    order = np.argsort(near[:, 0])
    xs, ys = near[order, 0], near[order, 1]
    if xs[-1] - xs[0] < 1.0e-6:
        return xm, ym
    a, b, c = np.polyfit(xs, ys, 2)
    if a >= 0 or abs(a) < 1.0e-9:
        return xm, ym
    xv = -b / (2.0 * a)
    yv = c - b * b / (4.0 * a)
    if not (xs[0] - 1e-9 <= xv <= xs[-1] + 1e-9):
        return xm, ym
    return float(xv), float(yv)


def compute_anchors_thesis(triang, x0, vf_array, vertex_method="min_slope"):
    """Return (x_int, y_int, x_top, y_top) following the thesis verbatim.

    Anchor:
      x_int = x0 (fixed in time); y_int = y where alpha(x0, y) = 0.5,
      interpolated from the alpha=0.5 contour-polyline segments straddling
      the vertical line x = x0.

    Vertex (xv, yv): selectable via `vertex_method`:
      - "min_slope"   : thesis-faithful, the contour point where |dy/dx| is
                        minimized within a window past the anchor (i.e.
                        partial_alpha/partial_x = 0 on the contour), refined
                        by a local parabolic fit.
      - "argmin_xhi"  : verbatim lobe.ipynb heuristic, the contour vertex
                        nearest x0 + 3.5*dx.

    Returns (None, None, None, None) on no contour.
    """
    _scratch_fig, _scratch_ax = plt.subplots()
    cs = _scratch_ax.tricontour(triang, vf_array, levels=[0.5], colors="none")
    plt.close(_scratch_fig)
    if hasattr(cs, "collections") and cs.collections:
        paths = cs.collections[0].get_paths()
    else:
        paths = cs.get_paths()
    if not paths:
        return None, None, None, None
    y0 = _thesis_y_at_x0(paths, x0)
    if y0 is None:
        return None, None, None, None
    x_int, y_int = float(x0), y0

    if vertex_method == "min_slope":
        x_top, y_top = _thesis_vertex_on_contour(paths, x_int, y_int)
        if x_top is None:
            return None, None, None, None
    elif vertex_method == "argmin_xhi":
        pts = np.vstack([p.vertices for p in paths])
        dx = 2e-3
        xhi = x_int + 3.5 * dx
        idx_top = np.argmin(np.abs(pts[:, 0] - xhi))
        x_top, y_top = float(pts[idx_top, 0]), float(pts[idx_top, 1])
    else:
        raise ValueError(f"unknown vertex_method: {vertex_method}")
    return x_int, y_int, x_top, y_top
