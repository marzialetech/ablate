#!/usr/bin/env python3
"""
Per-snapshot animations of receding-angle development for the slab2d case.

Three deliverables (all from the same 51-snapshot run):

  1. lobe_animation.mp4 — close-up of the leading-edge subregion. Per-frame
     styling mirrors dissertation Fig. 4.10 / 4.9b (lobe.ipynb @ line 350):
        - gray fill of the lobe region (water phase, alpha < 0.5)
        - black alpha=0.5 contour line
        - red dashed horizontal baseline at (x_int, y_int) extending right
        - anchor + vertex (red ×, green △); names + parabola + θ (Eq. 4.8)
          in a **legend** (no on-plot text labels)
        - red dashed parabola f(x) = A*(x-x_v)^2 + y_v over [x_int, xhi]
        - gray fill_between (parabola - baseline) — area-under-parabola shading
        - red filled wedge at the anchor (close-up only: R = dx/3 per radial
          edge; full-domain view keeps R = dx like lobe.ipynb)
          showing the receding angle theta
        - red x-axis (tangent) arrow and y-axis (normal) arrow rooted at anchor
        - no axis ticks; window x_int +/- 2*dx, y_int -/+ 3*dy (dx = dy = 2e-3)

  2. context_animation.mp4 — full domain (x in [0,0.1], y in [0,0.0254]) with
     a red rectangle marking the (1) close-up window. Provides spatial
     perspective for where the lobe lives within the slab.

  3. combined_animation.mp4 — both panels stacked vertically: (2) on top
     showing the full domain with the close-up rectangle, (1) on bottom
     showing the dissertation-style close-up.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Wedge


def discover_snapshots(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "slab2dcoords" / "domain").glob("domain.*.hdf5"))


def load_snapshot(path: Path):
    with h5py.File(path, "r") as f:
        verts = f["geometry/vertices"][:]
        cells = f["viz/topology/cells"][:]
        cx = verts[cells, 0].mean(axis=1)
        cy = verts[cells, 1].mean(axis=1)
        vof = f["cell_fields/solution_volumeFraction"][0, :]
        t = float(f["time"][0, 0])
    return cx, cy, vof, t


# Reuse the canonical anchor + vertex finders so animate_slab.py and
# postprocess_slab.py can never drift apart.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from postprocess_slab import compute_anchors_thesis  # noqa: E402


def compute_theta_and_anchors(triang, vof, x0, vertex_method: str = "min_slope"):
    """Return dict with x_int,y_int,x_top,y_top,A,slope,theta or None.

    Thesis-faithful: x_int is pinned to x0 (constant in time); y_int is the
    interpolated alpha=0.5 crossing on the vertical line x = x0; (x_top, y_top)
    is the contour point where |dy/dx| is minimized within a window past the
    anchor (i.e. partial_alpha/partial_x = 0 on the contour), refined by a
    local parabolic fit. Pass vertex_method="argmin_xhi" to fall back to the
    lobe.ipynb heuristic.
    """
    x_int, y_int, x_top, y_top = compute_anchors_thesis(
        triang, x0, vof, vertex_method=vertex_method)
    if x_int is None:
        return None
    Xv = x_top - x_int
    Yv = y_top - y_int
    if abs(Xv) < 1e-12:
        return None
    A = -Yv / (Xv ** 2)
    slope = -2 * A * Xv
    theta = float(np.degrees(np.arctan(slope)))
    return dict(
        x_int=x_int, y_int=y_int, x_top=x_top, y_top=y_top,
        A=A, slope=slope, theta=theta,
    )


# Context view bounds (cropped from simulation domain [0, 0.1] x [0, 0.0254] m
# to focus on the leading-edge region; the far field is dropped).
DOMAIN_X = (0.01, 0.09)
DOMAIN_Y = (0.005, 0.025)

# Wedge radius = wedge_radius_frac * dx (each straight radial side to the arc).
# Close-up uses a compact fan (1/3 dx); full-domain context keeps dissertation R ~ dx.
WEDGE_RADIUS_FRAC_CLOSEUP = 1.0 / 3.0
WEDGE_RADIUS_FRAC_CONTEXT = 1.0


def closeup_window(x_int: float, y_int: float, dx: float = 2e-3,
                   dy: float = 2e-3) -> tuple[float, float, float, float]:
    """Fixed dissertation window (lobe.ipynb): independent of vertex location.

    The lobe vertex may lie to the right of x_int + 3.5*dx (thesis min_slope);
    it is still drawn on the full-domain view but can be clipped in the zoom.
    """
    return x_int - 2 * dx, x_int + 3.5 * dx, y_int - dy, y_int + 3 * dy


def draw_overlay_on_axes(ax, info: dict, dx: float = 2e-3,
                         dy: float = 2e-3,
                         wedge_radius_frac: float = WEDGE_RADIUS_FRAC_CLOSEUP,
                         ) -> None:
    """Draw the dissertation Fig. 4.9b overlays at the anchor: red dashed
    parabola + baseline, gray fill_between, red filled wedge, red local x/y
    arrows, anchor + vertex markers (legend gives definitions). Used by *both* the close-up
    panel and the full-domain context panel.

    ``wedge_radius_frac`` scales the wedge: use ``WEDGE_RADIUS_FRAC_CLOSEUP``
    (dx/3) for zoom panels and ``WEDGE_RADIUS_FRAC_CONTEXT`` (dx) for the
    full-domain view so the fan stays readable at map scale.

    A single legend (anchor / vertex / parabola / θ via Eq. 4.8) replaces
    on-plot text labels; placement differs slightly between close-up and context.

    Geometric overlays (baseline, parabola, gray fill) end at x_int + 3.5*dx,
    matching the fixed red-box / close-up width. Vertex marker uses true
    (x_top, y_top) and may fall outside the close-up xlim.
    """
    x_int, y_int = info["x_int"], info["y_int"]
    x_top, y_top = info["x_top"], info["y_top"]
    A = info["A"]
    theta = info["theta"]
    x_right = x_int + 3.5 * dx

    ax.hlines(y_int, x_int, x_right, color="r", lw=1.6,
              linestyle="--", zorder=4)

    xpl = np.linspace(x_int, x_right, 200)
    ypl = A * (xpl - x_top) ** 2 + y_top
    ax.plot(xpl, ypl, color="r", linestyle="--", lw=2.0, zorder=5)

    ax.fill_between(
        xpl, ypl, np.full_like(xpl, y_int),
        where=(ypl >= y_int),
        color="gray", alpha=0.30, linewidth=0, zorder=3,
    )

    R = wedge_radius_frac * dx
    ax.add_patch(Wedge(
        (x_int, y_int), R, 0, theta,
        facecolor="red", edgecolor=None, alpha=0.35, linewidth=0, zorder=6,
    ))

    L = 2 * dx
    H = 2 * dy
    body_w = 0.02 * dx
    hw = 0.2 * dx
    hl = 0.2 * dx
    ax.arrow(x_int, y_int, L, 0, head_width=hw, head_length=hl,
             width=body_w, fc="r", ec="r", linewidth=0.6, zorder=7,
             length_includes_head=True)
    ax.arrow(x_int, y_int, 0, H, head_width=hw, head_length=hl,
             width=body_w, fc="r", ec="r", linewidth=0.6, zorder=7,
             length_includes_head=True)

    # anchor + vertex markers only (names + equations live in legend below)
    ax.plot(x_int, y_int, "rx", markersize=10, mew=2.2, zorder=11)
    ax.plot(x_top, y_top, marker="^", ms=9, mfc="lime", mec="darkgreen",
            mew=1.2, linestyle="none", zorder=11)

    is_context = abs(wedge_radius_frac - WEDGE_RADIUS_FRAC_CONTEXT) < 1e-12
    fontsize = 6 if is_context else 7
    loc = "upper right" if is_context else "upper left"

    leg_handles = [
        Line2D(
            [0], [0], marker="x", color="red", markeredgewidth=2.2,
            markersize=10, linestyle="none",
            label=r"anchor $(x_0,y_0)$: $x_0=\mathrm{const}$, "
                  r"$\alpha(x_0,y)=\frac{1}{2}$",
        ),
        Line2D(
            [0], [0], marker="^", color="none", markerfacecolor="lime",
            markeredgecolor="darkgreen", markersize=9, linestyle="none",
            label=r"vertex $(x_v,y_v)$: $\partial\alpha/\partial x=0$ on "
                  r"$\alpha=\frac{1}{2}$",
        ),
        Line2D(
            [0], [0], color="red", linestyle="--", linewidth=2,
            label=r"parabola: $f(x)=A(x-x_v)^2+y_v$,  "
                  r"$A=\frac{y_0-y_v}{(x_0-x_v)^2}$",
        ),
        Line2D(
            [0], [0], color="none", linestyle="none", markersize=0,
            label=r"$\theta_{\mathrm{sim}}=\frac{180}{\pi}\,\arctan(f'(x_0))$,  "
                  r"$f'(x_0)=2A(x_0-x_v)$  (Eq. 4.8)",
        ),
    ]
    leg = ax.legend(
        handles=leg_handles,
        loc=loc,
        fontsize=fontsize,
        framealpha=0.93,
        edgecolor="0.45",
        fancybox=True,
        borderpad=0.55,
        labelspacing=0.35,
    )
    if leg is not None:
        leg.set_zorder(25)


def draw_closeup_on_axes(ax, triang, vof, info: dict, dx: float = 2e-3,
                         dy: float = 2e-3, framed: bool = False) -> None:
    """Render dissertation Fig. 4.10 / 4.9b styling onto an existing ax.

    If framed=True, the panel itself is wrapped in a red rectangle so the
    panel reads as 'the inside of the red box from the context view'.
    """
    x_int, y_int = info["x_int"], info["y_int"]

    ax.tricontourf(triang, vof, levels=[-0.5, 0.5], colors=["#7e848d"],
                   alpha=0.55)
    ax.tricontour(triang, vof, levels=[0.5], colors="k", linewidths=1.1)

    draw_overlay_on_axes(ax, info, dx, dy)

    xlo, xhi, ylo, yhi = closeup_window(x_int, y_int, dx, dy)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if framed:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("r")
            spine.set_linewidth(1.8)
    else:
        for spine in ax.spines.values():
            spine.set_visible(False)


def draw_context_on_axes(ax, triang, vof, info: dict, dx: float = 2e-3,
                         dy: float = 2e-3) -> None:
    """Full domain + same Fig. 4.9b overlays at the anchor + red rectangle
    marking the close-up window. The close-up panel is *literally* the
    interior of this rectangle magnified.
    """
    x_int, y_int = info["x_int"], info["y_int"]

    ax.tricontourf(triang, vof, levels=[-0.5, 0.5], colors=["#7e848d"],
                   alpha=0.55)
    ax.tricontour(triang, vof, levels=[0.5], colors="k", linewidths=0.9)

    # exact same overlays as in the close-up, drawn at world scale
    # (full-domain wedge uses R = dx so the angle fan stays visible at map scale)
    draw_overlay_on_axes(ax, info, dx, dy,
                         wedge_radius_frac=WEDGE_RADIUS_FRAC_CONTEXT)

    xlo, xhi, ylo, yhi = closeup_window(x_int, y_int, dx, dy)
    rect = Rectangle((xlo, ylo), xhi - xlo, yhi - ylo,
                     linewidth=1.8, edgecolor="r", facecolor="none",
                     linestyle="-", zorder=10)
    ax.add_patch(rect)

    ax.set_xlim(*DOMAIN_X)
    ax.set_ylim(*DOMAIN_Y)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]", fontsize=9)
    ax.set_ylabel("y [m]", fontsize=9)
    ax.tick_params(labelsize=8)


def render_closeup_frame(out_png: Path, triang, vof, t: float,
                         info: dict, dataset_label: str = "slab") -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    draw_closeup_on_axes(ax, triang, vof, info)
    ax.set_title(
        rf"{dataset_label} (We=1000)   $t={t:.4e}$ s   $\theta={info['theta']:.2f}^\circ$",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def render_context_frame(out_png: Path, triang, vof, t: float,
                         info: dict, dataset_label: str = "slab") -> None:
    Lx = DOMAIN_X[1] - DOMAIN_X[0]
    Ly = DOMAIN_Y[1] - DOMAIN_Y[0]
    W = 11.5
    H_ax = W * Ly / Lx
    fig, ax = plt.subplots(figsize=(W, H_ax + 0.9))
    draw_context_on_axes(ax, triang, vof, info)
    ax.set_title(
        rf"{dataset_label} (We=1000)   $t={t:.4e}$ s   "
        rf"$\theta={info['theta']:.2f}^\circ$   "
        r"domain (excluding far field; red box = closeup)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def render_combined_frame(out_png: Path, triang, vof, t: float,
                          info: dict, dataset_label: str = "slab") -> None:
    """Two-panel: cropped domain on top (with red rectangle), close-up below.

    Both panels use set_aspect("equal"), so the figure height has to match:
      ctx_h = W * Ly_ctx / Lx_ctx               (top panel axis height)
      cu_h  = W * (4*dy) / (6*dx) = W * (2/3)   (close-up axis height)
    Lx_ctx, Ly_ctx come from DOMAIN_X / DOMAIN_Y.
    """
    W = 11.5
    Lx_ctx = DOMAIN_X[1] - DOMAIN_X[0]
    Ly_ctx = DOMAIN_Y[1] - DOMAIN_Y[0]
    ctx_h = W * Ly_ctx / Lx_ctx
    cu_h = W * (4 * 2e-3) / (6 * 2e-3)
    H = ctx_h + cu_h + 1.5

    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(
        2, 1,
        height_ratios=[ctx_h, cu_h],
        hspace=0.18, top=0.93, bottom=0.05, left=0.07, right=0.97,
    )
    ax_ctx = fig.add_subplot(gs[0])
    ax_cu = fig.add_subplot(gs[1])

    draw_context_on_axes(ax_ctx, triang, vof, info)
    ax_ctx.set_title("domain (excluding far field; red box = closeup)",
                     fontsize=10)

    draw_closeup_on_axes(ax_cu, triang, vof, info, framed=True)
    ax_cu.set_title("close-up", fontsize=10)

    fig.suptitle(
        rf"{dataset_label} (We=1000)   $t={t:.4e}$ s   $\theta={info['theta']:.2f}^\circ$",
        fontsize=12, y=0.985,
    )
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


RENDERERS = {
    "lobe":     ("frames",          "lobe_animation.mp4",     render_closeup_frame),
    "context":  ("frames_context",  "context_animation.mp4",  render_context_frame),
    "combined": ("frames_combined", "combined_animation.mp4", render_combined_frame),
}


def stitch_mp4(frames_dir: Path, mp4: Path, fps: int) -> None:
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        str(mp4),
    ]
    print("[anim] running:", " ".join(cmd))
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    print(f"[anim] wrote {mp4}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir (default: <run_dir>/postproc/animation)")
    ap.add_argument("--x0", type=float, default=25.5e-3,
                    help="reference x location (matches lobe.ipynb @ line 884)")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--variant", choices=list(RENDERERS) + ["all"],
                    default="all",
                    help="which animation(s) to produce (default: all)")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument(
        "--label", type=str, default=None,
        help="dataset label printed in panel/figure titles (default: <run_dir basename>)",
    )
    args = ap.parse_args()
    label = args.label or args.run_dir.resolve().name

    out = args.out or (args.run_dir / "postproc" / "animation")
    out.mkdir(parents=True, exist_ok=True)

    snaps = discover_snapshots(args.run_dir)
    if not snaps:
        print(f"no snapshots in {args.run_dir}")
        return 1
    print(f"[anim] {len(snaps)} snapshots")

    variants = list(RENDERERS) if args.variant == "all" else [args.variant]
    frames_dirs = {v: out / RENDERERS[v][0] for v in variants}
    for fd in frames_dirs.values():
        fd.mkdir(exist_ok=True)

    for k, p in enumerate(snaps):
        x, y, vof, t = load_snapshot(p)
        triang = mtri.Triangulation(x, y)
        info = compute_theta_and_anchors(triang, vof, args.x0)
        if info is None:
            print(f"  frame {k:>4d}/{len(snaps)-1}  t={t:.4e}s  no contour, skip")
            continue
        for v in variants:
            png = frames_dirs[v] / f"frame_{k:04d}.png"
            RENDERERS[v][2](png, triang, vof, t, info, dataset_label=label)
        if k % 5 == 0 or k == len(snaps) - 1:
            print(f"  frame {k:>4d}/{len(snaps)-1}  t={t:.4e}s  theta={info['theta']:.2f} deg")

    if args.no_video or shutil.which("ffmpeg") is None:
        for v in variants:
            print(f"[anim] frames at {frames_dirs[v]}; ffmpeg skipped")
        return 0

    for v in variants:
        stitch_mp4(frames_dirs[v], out / RENDERERS[v][1], args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
