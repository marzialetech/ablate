#!/usr/bin/env python3
"""
Receding-angle postproc for the slab2d case.

Algorithm: VERBATIM copy of `compute_theta_from_vf` from
    /Users/jjmarzia/Downloads/marzialeIP_noHDF5/notebooks_python/ch4/lobe.ipynb
(see the cell whose source begins at line 831 of that notebook). Only HDF5
reading glue is new.

Usage:
    python tools/postprocess_slab.py <run_dir> [--mode summary|plot] [--x0 X0]

<run_dir> contains slab2dcoords/domain/domain.*.hdf5
"""
import argparse
import os
import sys
from glob import glob

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np


# -------------------------------------------------------------------
# BEGIN VERBATIM COPY of compute_theta_from_vf from
#   marzialeIP_noHDF5/notebooks_python/ch4/lobe.ipynb (cell @ line 831)
# Only modifications:
#   - the docstring is preserved
#   - this is a Python file, not a notebook cell
# -------------------------------------------------------------------
def compute_theta_from_vf(xm, ym, triang, x0, vf_array):
    """
    compute theta for a given vf field
    return degrees or np.nan if no contour exists
    """
    try:
        # see compute_anchors_thesis: do not touch the caller's figure
        _scratch_fig, _scratch_ax = plt.subplots()
        cs = _scratch_ax.tricontour(triang, vf_array, levels=[0.5], colors="none")
        plt.close(_scratch_fig)
        # matplotlib >=3.8 removed `cs.collections`; the dissertation notebook
        # used `cs.collections[0].get_paths()` against mpl 3.7.x. The single
        # ContourSet now exposes `get_paths()` directly, so try the modern API
        # first and fall back to the legacy attribute.
        if hasattr(cs, "collections") and cs.collections:
            paths = cs.collections[0].get_paths()
        else:
            paths = cs.get_paths()
        if len(paths) == 0:
            return np.nan

        pts = np.vstack([p.vertices for p in paths])

        # nearest point on contour to x0
        idx = np.argmin(np.abs(pts[:, 0] - x0))
        x_int, y_int = pts[idx]

        # window matching your main code
        dx = 2e-3
        dy = 2e-3
        xhi = x_int + 3.5 * dx

        # find the "vertex" point for parabola
        idx_top = np.argmin(np.abs(pts[:, 0] - xhi))
        x_top, y_top = pts[idx_top]

        Xv = x_top - x_int
        Yv = y_top - y_int
        if abs(Xv) < 1e-12:
            return np.nan

        A = -Yv / (Xv ** 2)

        slope = -2 * A * Xv
        theta_rad = np.arctan(slope)
        return np.degrees(theta_rad)

    except Exception:
        return np.nan
# -------------------------------------------------------------------
# END VERBATIM COPY
# -------------------------------------------------------------------


# -------------------------------------------------------------------
# Thesis-faithful variant. Per dissertation Sec. 4.4:
#     "(x0, y0) ... x0 = 25e-3, y0 = y : { alpha(x0, y) = 0.5 }"
# i.e. x_int is fixed in time at x0 and y_int is the y-coord of the
# alpha=0.5 contour ON the vertical line x = x0. The verbatim notebook
# function above approximates this with `argmin(|pts[:,0] - x0|)` which
# leaves x_int wiggling at mesh resolution; this variant pins x_int = x0
# by linearly interpolating each alpha=0.5 contour-segment that straddles
# x = x0.
# -------------------------------------------------------------------
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
    best = None       # (abs_slope, xm, ym, x_neighbors, y_neighbors)
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

    # local-parabola refinement on contour points within the fit window
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
    # Use a private off-screen figure so we never disturb whatever the caller
    # is drawing on plt.gcf(); the previous behaviour of plt.tricontour + plt.clf()
    # would nuke the caller's active figure (e.g. compare_slab.py panels).
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


def compute_theta_thesis(triang, x0, vf_array, vertex_method="min_slope"):
    """Thesis-faithful theta. See compute_anchors_thesis for vertex_method."""
    x_int, y_int, x_top, y_top = compute_anchors_thesis(
        triang, x0, vf_array, vertex_method=vertex_method)
    if x_int is None:
        return np.nan
    Xv = x_top - x_int
    Yv = y_top - y_int
    if abs(Xv) < 1e-12:
        return np.nan
    A = -Yv / (Xv ** 2)
    slope = -2 * A * Xv
    return float(np.degrees(np.arctan(slope)))


def detect_equilibrium(times, thetas,
                       tau_adv,
                       iqr_threshold_deg=0.5,
                       drift_threshold_deg=0.5,
                       min_floor_factor=3.0,
                       window_factor=1.0,
                       persistence=2):
    """Detect quasi-steady lobe morphology in a theta(t) trajectory.

    Walks non-overlapping sliding windows of width ~window_factor*tau_adv,
    starting at t >= min_floor_factor*tau_adv, looking for the first run of
    `persistence` consecutive windows that BOTH:

      (a) have interquartile range  IQR(theta)  <  iqr_threshold_deg
          (intra-window stationarity in spread; IQR is robust to the isolated
          single-frame satellite-blob spikes documented in
          slab72-fixed/README.md), AND
      (b) have median(theta) within  drift_threshold_deg  of the immediately
          preceding window's median (inter-window stationarity in trend; this
          catches the case where a slow monotonic climb has small per-window
          IQR but is plainly still drifting upward).

    Both checks are required because a slow drift can keep IQR small inside
    every individual window while the underlying trajectory is far from
    settled.

    Returns a dict:
      converged       : bool
      t_converged     : float or None  (end of the n-th persistent window)
      theta_converged : float or None  (median theta across all snapshots in
                                        the persistent run)
      window_samples  : int            (snapshots per window)
      first_eligible_t: float          (the min-time floor)
      qualifying_windows : list of (i_start, i_end, median, iqr, t_window_end)
      message         : str
    """
    times = np.asarray(times, dtype=float)
    thetas = np.asarray(thetas, dtype=float)
    if len(times) < 2:
        return {"converged": False, "t_converged": None, "theta_converged": None,
                "window_samples": 0, "first_eligible_t": np.nan,
                "qualifying_windows": [],
                "message": "trajectory too short (<2 snapshots)"}

    dt_snap = float(np.median(np.diff(times)))
    if dt_snap <= 0:
        return {"converged": False, "t_converged": None, "theta_converged": None,
                "window_samples": 0, "first_eligible_t": np.nan,
                "qualifying_windows": [],
                "message": "non-monotonic snapshot times"}

    win_samples = max(3, int(round(window_factor * tau_adv / dt_snap)))
    min_floor_t = min_floor_factor * tau_adv
    floor_idx = int(np.searchsorted(times, min_floor_t, side="left"))

    qualifying = []
    streak = 0
    streak_start_window_i = None
    prev_streak_median = None  # median of the previous window in the current streak
    finite = np.isfinite(thetas)

    i = floor_idx
    while i + win_samples <= len(times):
        sl = slice(i, i + win_samples)
        th = thetas[sl][finite[sl]]
        if len(th) >= 3:
            q25, q50, q75 = np.percentile(th, [25, 50, 75])
            iqr = q75 - q25
            iqr_ok = iqr < iqr_threshold_deg
        else:
            q50, iqr = np.nan, np.inf
            iqr_ok = False

        # Intra-window IQR is necessary; inter-window drift is also required for
        # streaks of length >= 2 (the very first window in a streak has no
        # predecessor so it is graded on IQR alone).
        if iqr_ok and streak >= 1:
            drift_ok = abs(float(q50) - prev_streak_median) < drift_threshold_deg
        else:
            drift_ok = True

        qualifies = iqr_ok and drift_ok

        if qualifies:
            if streak == 0:
                streak_start_window_i = i
            streak += 1
            qualifying.append((i, i + win_samples, float(q50), float(iqr),
                               float(times[i + win_samples - 1])))
            prev_streak_median = float(q50)
            if streak >= persistence:
                streak_end = i + win_samples
                streak_start = streak_start_window_i
                run_thetas = thetas[streak_start:streak_end]
                run_thetas = run_thetas[np.isfinite(run_thetas)]
                t_conv = float(times[streak_end - 1])
                theta_conv = float(np.median(run_thetas))
                return {
                    "converged": True,
                    "t_converged": t_conv,
                    "theta_converged": theta_conv,
                    "window_samples": win_samples,
                    "first_eligible_t": float(min_floor_t),
                    "qualifying_windows": qualifying,
                    "message": (
                        f"converged at t={t_conv:.4e}s "
                        f"(median theta = {theta_conv:.3f} deg "
                        f"over {persistence} consecutive {win_samples}-frame windows; "
                        f"IQR < {iqr_threshold_deg:g} deg, "
                        f"|delta-median| < {drift_threshold_deg:g} deg)"
                    ),
                }
        else:
            streak = 0
            streak_start_window_i = None
            prev_streak_median = None
        i += win_samples

    return {
        "converged": False,
        "t_converged": None,
        "theta_converged": None,
        "window_samples": win_samples,
        "first_eligible_t": float(min_floor_t),
        "qualifying_windows": qualifying,
        "message": (
            f"no convergence: {len(qualifying)} qualifying window(s) "
            f"out of {(len(times) - floor_idx) // max(1, win_samples)} eligible; "
            f"required {persistence} consecutive."
        ),
    }


def theta_at_time(times, thetas, t_target):
    """Linear interpolation of theta(t) at t_target; NaN if out of range or
    surrounding snapshots are NaN."""
    times = np.asarray(times, dtype=float)
    thetas = np.asarray(thetas, dtype=float)
    if t_target < times[0] or t_target > times[-1]:
        return float("nan")
    j = int(np.searchsorted(times, t_target))
    if j == 0:
        return float(thetas[0])
    a, b = j - 1, j
    if not (np.isfinite(thetas[a]) and np.isfinite(thetas[b])):
        return float("nan")
    f = (t_target - times[a]) / (times[b] - times[a])
    return float(thetas[a] + f * (thetas[b] - thetas[a]))


def load_snapshot(path):
    """Read centroid coords + per-cell volumeFraction + sim time from an HDF5 snapshot.

    Centroids are computed from geometry/vertices and viz/topology/cells (a
    triangle index list of shape [Ncells, 3]) instead of relying on the
    aux_xpos / aux_ypos fields, because the locations process only populates
    those for the flow region; combining all cell groups via the topology
    avoids holes from boundary-extruded cells.
    """
    with h5py.File(path, "r") as f:
        verts = f["geometry/vertices"][:]
        cells = f["viz/topology/cells"][:]
        cx = verts[cells, 0].mean(axis=1)
        cy = verts[cells, 1].mean(axis=1)
        vof = f["cell_fields/solution_volumeFraction"][0, :]
        t = float(f["time"][0, 0])
    return cx, cy, vof, t


def display_field(vof, invert):
    """Convention-invariant view: display_field(vof, True) returns 1-vof so the
    lobe (paraffin) appears with indicator=1, matching dissertation Fig. 4.10.

    The angle calculation is unaffected (alpha=0.5 level set is symmetric under
    alpha -> 1-alpha), so this only changes the colormap display, not the
    physics or the metric.
    """
    return (1.0 - vof) if invert else vof


def find_snapshots(run_dir):
    pat = os.path.join(run_dir, "slab2dcoords", "domain", "domain.*.hdf5")
    return sorted(glob(pat))


def cmd_summary(args):
    snaps = find_snapshots(args.run_dir)
    if not snaps:
        print(f"no snapshots in {args.run_dir}", file=sys.stderr)
        sys.exit(2)
    print(f"{len(snaps)} snapshots in {args.run_dir}")
    print(f"x0 = {args.x0:g}")
    print(f"{'idx':>4}  {'time':>10}  {'theta_deg':>10}")
    for k, p in enumerate(snaps):
        x, y, vof, t = load_snapshot(p)
        triang = tri.Triangulation(x, y)
        theta = compute_theta_thesis(triang, args.x0, vof)
        print(f"{k:>4}  {t:>10.4e}  {theta:>10.3f}")


def cmd_plot_final(args):
    snaps = find_snapshots(args.run_dir)
    if not snaps:
        print(f"no snapshots in {args.run_dir}", file=sys.stderr)
        sys.exit(2)
    p = snaps[-1]
    x, y, vof, t = load_snapshot(p)
    triang = tri.Triangulation(x, y)
    theta = compute_theta_thesis(triang, args.x0, vof)
    os.makedirs(args.out_dir, exist_ok=True)
    disp = display_field(vof, args.invert_display)
    disp_name = "1-alpha (paraffin)" if args.invert_display else "alpha"

    # full-domain VOF + alpha=0.5 contour
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.tricontourf(triang, disp, levels=20, cmap="coolwarm")
    ax.tricontour(triang, vof, levels=[0.5], colors="k", linewidths=1.5)
    ax.axvline(args.x0, color="y", lw=0.6, alpha=0.6)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    label = args.label or os.path.basename(os.path.abspath(args.run_dir))
    ax.set_title(
        f"{label}  t={t:.4e}s   theta = {theta:.2f} deg   "
        f"(dissertation Fig. 4.10 @ We=1000: 29.43 deg)"
    )
    out_full = os.path.join(args.out_dir, "lobe_final.png")
    fig.savefig(out_full, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_full}")

    # zoomed leading-edge view with parabola fit overlay
    x_int, y_int, x_top, y_top = compute_anchors_thesis(triang, args.x0, vof)
    dx = 2e-3

    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    ax2.tricontourf(triang, disp, levels=20, cmap="coolwarm")
    ax2.tricontour(triang, vof, levels=[0.5], colors="k", linewidths=1.5)
    ax2.plot(x_int, y_int, "ro", ms=8, label=rf"ref $(x_0,y_0)$ @ $x_0={args.x0:g}$")
    ax2.plot(x_top, y_top, "g^", ms=8, label=r"vertex $(x_v,y_v)$, $\partial\alpha/\partial x=0$")
    # Parabola from dissertation Eq. 4.8:
    #   f(x) = A*(x - xv)^2 + yv,  A = (y0 - yv)/(x0 - xv)^2
    # opens downward (A < 0) anchored at the right-side peak (xv,yv).
    Xv = x_top - x_int
    Yv = y_top - y_int
    A = -Yv / (Xv ** 2)
    xpl = np.linspace(x_int - 0.3 * dx, x_top + 0.5 * dx, 200)
    ypl = A * (xpl - x_top) ** 2 + y_top
    ax2.plot(xpl, ypl, "y-", lw=2.2,
             label=rf"$f(x)=A(x-x_v)^2+y_v$,  $\theta={theta:.2f}^\circ$")
    # tangent at ref point (slope = 2A*(x0-xv))
    slope = 2 * A * (x_int - x_top)
    L = 1.5 * dx
    ax2.plot([x_int - L, x_int + L], [y_int - L * slope, y_int + L * slope],
             "r--", lw=1.4, alpha=0.75, label=r"tangent at $(x_0,y_0)$")
    ax2.set_xlim(x_int - 2 * dx, x_int + 4 * dx)
    ax2.set_ylim(y_int - dx, y_int + 3 * dx)
    ax2.set_aspect("equal")
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    ax2.set_title(f"{label}  leading edge zoom  t={t:.4e}s")
    ax2.legend(loc="lower right", fontsize=8)
    out_zoom = os.path.join(args.out_dir, "lobe_zoom.png")
    fig2.savefig(out_zoom, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"wrote {out_zoom}")

    # theta(t) plot
    times, thetas = [], []
    for sp in snaps:
        xx, yy, vv, tt = load_snapshot(sp)
        ttri = tri.Triangulation(xx, yy)
        th = compute_theta_thesis(ttri, args.x0, vv)
        times.append(tt)
        thetas.append(th)
    fig3, ax3 = plt.subplots(figsize=(8, 4.5))
    label = args.label or os.path.basename(os.path.abspath(args.run_dir))
    ax3.plot(times, thetas, "ko-", lw=1.2, ms=3, label=label)
    ax3.axhline(29.43, color="r", ls="--", lw=1.2,
                label="dissertation Fig. 4.10 @ We=1000: 29.43 deg")

    title_extra = ""
    if args.detect_equilibrium:
        det = detect_equilibrium(
            times, thetas,
            tau_adv=args.tau_adv,
            iqr_threshold_deg=args.equil_iqr_threshold,
            drift_threshold_deg=args.equil_drift_threshold,
            min_floor_factor=args.equil_min_floor,
            window_factor=args.equil_window,
            persistence=args.equil_persistence,
        )
        ax3.axvline(det["first_eligible_t"], color="0.4", ls=":", lw=1.0,
                    label=rf"min-time floor ({args.equil_min_floor:g}$\tau_{{adv}}$)")
        if det["converged"]:
            ax3.axvline(det["t_converged"], color="g", ls="-", lw=1.4, alpha=0.85,
                        label=(rf"equil. $t={det['t_converged']:.3e}$ s, "
                               rf"$\bar\theta={det['theta_converged']:.2f}^\circ$"))
            ax3.axhline(det["theta_converged"], color="g", ls=":", lw=0.9, alpha=0.6)
            title_extra = (f"  |  equil. theta = {det['theta_converged']:.2f} deg "
                           f"@ t = {det['t_converged']:.3e}s")
            print(f"[detect-equilibrium] {det['message']}")
        else:
            title_extra = "  |  equilibrium NOT detected (hard cap)"
            print(f"[detect-equilibrium] {det['message']}")
        # apples-to-apples reference: theta interpolated at the dissertation t_f
        if args.diss_tf is not None:
            th_diss = theta_at_time(times, thetas, args.diss_tf)
            print(f"[detect-equilibrium] theta at dissertation t_f={args.diss_tf:g}s "
                  f"= {th_diss:.3f} deg (interpolated)")
            if np.isfinite(th_diss):
                ax3.plot([args.diss_tf], [th_diss], "ms", ms=8, mfc="none", mew=1.5,
                         label=rf"diss. $t_f={args.diss_tf:g}$s: $\theta={th_diss:.2f}^\circ$")

    ax3.set_xlabel("time [s]")
    ax3.set_ylabel(r"$\theta$ [deg]")
    ax3.set_title(f"{label} (We=1000): final theta = {thetas[-1]:.2f} deg{title_extra}")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="lower right", fontsize=8)
    out_theta = os.path.join(args.out_dir, "theta_vs_time.png")
    fig3.savefig(out_theta, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"wrote {out_theta}")

    if not np.isnan(theta):
        print(f"theta = {theta:.3f} deg  (dissertation = 29.43 deg, "
              f"error = {theta-29.43:+.3f} deg, "
              f"relative {(theta-29.43)/29.43*100:+.2f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="run directory containing slab2dcoords/domain/")
    ap.add_argument("--x0", type=float, default=25.5e-3,
                    help="reference x location (matches lobe.ipynb cell @ line 884)")
    ap.add_argument("--mode", choices=["summary", "plot"], default="summary")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument(
        "--invert-display", action="store_true",
        help="Display 1-alpha (lobe shows up as 1, dissertation Fig. 4.10 convention). "
             "Use for slab72-fixed-style runs where the user-visible convention is "
             "vof=1 <-> paraffin even though the codebase stores alpha=alpha_air.",
    )
    ap.add_argument(
        "--label", default=None,
        help="dataset label used in plot titles (default: basename of run_dir)",
    )
    ap.add_argument(
        "--detect-equilibrium", action="store_true",
        help="annotate theta_vs_time.png with the equilibration verdict and the "
             "theta value interpolated at the dissertation t_f.",
    )
    ap.add_argument(
        "--tau-adv", type=float, default=1.41e-4,
        help="advective timescale L_lobe / U_shear (default: 0.01/70.71 = 1.41e-4 s "
             "for the slab72 We=1000 case).",
    )
    ap.add_argument("--equil-iqr-threshold", type=float, default=0.5,
                    help="IQR(theta) threshold in degrees (default: 0.5).")
    ap.add_argument("--equil-drift-threshold", type=float, default=0.5,
                    help="Inter-window |median delta| threshold in degrees "
                         "(default: 0.5).")
    ap.add_argument("--equil-min-floor", type=float, default=3.0,
                    help="min-time floor in units of tau_adv (default: 3).")
    ap.add_argument("--equil-window", type=float, default=1.0,
                    help="sliding-window width in units of tau_adv (default: 1).")
    ap.add_argument("--equil-persistence", type=int, default=2,
                    help="number of consecutive qualifying windows required (default: 2).")
    ap.add_argument("--diss-tf", type=float, default=5.0e-4,
                    help="dissertation t_f for apples-to-apples reference (default: 5e-4 s).")
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.join(args.run_dir, "postproc")
    if args.mode == "summary":
        cmd_summary(args)
    else:
        cmd_plot_final(args)


if __name__ == "__main__":
    main()
