#!/usr/bin/env python3
"""
Postprocess `nphase-zalesak-5phase` results: render volume-fraction (alphak) and
interface-sharpening RHS (fsharpk) fields per snapshot as PNG frames suitable
for visual review or stitching into an animation.

Usage:
    python tools/postprocess_zalesak.py RUN_DIR [--out OUT_DIR] [--field FIELD]
                                                [--animate] [--cli-only]

RUN_DIR is the timestamped directory created by ablate's Hdf5MultiFileSerializer
(e.g. `nphase-zalesak-5phase_2026-04-30T19-44-20`). The script discovers all
`domain/domain.NNNNN.hdf5` files inside it, reads `solution_alphak` and
`aux_fsharpk`, and writes per-snapshot PNGs to `<RUN_DIR>/postprocess/` (or the
directory passed via `--out`).

The cell ordering inside each HDF5 is lexicographic (row-major: x fastest, then
y) which lets us reshape directly to a (ny, nx) image.

Outputs per snapshot frame_NNNNN.png:
  Top row    : alpha_k for k = 0..4 (volume fraction per phase, range [0,1])
  Bottom row : fsharp_k for k = 0..4 (interface-sharpening RHS, signed)
  Title      : t = <sim time>, step = <NNNNN>

Also writes:
  argmax_NNNNN.png  : composite view, each cell colored by its dominant phase
  summary.png       : 4-up time-evolution mosaic of the dominant-phase field

If `--animate` is passed and `ffmpeg` is on PATH, the per-frame PNGs are
stitched into `frames.mp4` and `argmax.mp4`.

Sanity check: fsharpk should be nonzero in interfacial cells (a few-cell halo
around each disk boundary). If every fsharpk field is identically zero, the
NPhaseIntSharp process is not contributing.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


SNAPSHOT_RE = re.compile(r"domain\.(\d+)\.hdf5$")


def discover_snapshots(run_dir: Path) -> List[Path]:
    domain_dir = run_dir / "domain"
    if not domain_dir.is_dir():
        raise SystemExit(f"no `domain/` subdirectory under {run_dir}")
    files = []
    for p in domain_dir.iterdir():
        m = SNAPSHOT_RE.search(p.name)
        if m:
            files.append((int(m.group(1)), p))
    if not files:
        raise SystemExit(f"no domain.NNNNN.hdf5 files under {domain_dir}")
    files.sort()
    return [p for _, p in files]


def grid_shape(verts: np.ndarray) -> Tuple[int, int]:
    """Recover (ny, nx) for a uniform structured 2D mesh from vertex coords."""
    xs = np.unique(verts[:, 0])
    ys = np.unique(verts[:, 1])
    return len(ys) - 1, len(xs) - 1


def read_snapshot(path: Path):
    with h5py.File(path, "r") as f:
        alphak = f["cell_fields/solution_alphak"][0]   # (ncell, nphase)
        fsharpk = (
            f["cell_fields/aux_fsharpk"][0] if "cell_fields/aux_fsharpk" in f else None
        )
        time = float(f["time"][0, 0])
        verts = f["geometry/vertices"][:]
    return alphak, fsharpk, time, verts


def to_image(field_1d: np.ndarray, ny: int, nx: int) -> np.ndarray:
    return field_1d.reshape(ny, nx)


def render_frame(
    out_path: Path,
    alphak: np.ndarray,
    fsharpk: np.ndarray | None,
    time: float,
    step: int,
    ny: int,
    nx: int,
):
    nphase = alphak.shape[1]
    nrows = 2 if fsharpk is not None else 1
    fig, axes = plt.subplots(nrows, nphase, figsize=(3.0 * nphase, 3.0 * nrows + 0.3))
    if nrows == 1:
        axes = np.array([axes])

    for k in range(nphase):
        img = to_image(alphak[:, k], ny, nx)
        ax = axes[0, k]
        im = ax.imshow(
            img,
            origin="lower",
            extent=(0, 1, 0, 1),
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
            aspect="equal",
        )
        ax.set_title(f"alpha_{k}")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if fsharpk is not None:
        # Symmetric color range around 0 so positive/negative contributions are visible.
        absmax = float(np.abs(fsharpk).max())
        if absmax == 0.0:
            absmax = 1e-30
        for k in range(nphase):
            img = to_image(fsharpk[:, k], ny, nx)
            ax = axes[1, k]
            im = ax.imshow(
                img,
                origin="lower",
                extent=(0, 1, 0, 1),
                vmin=-absmax,
                vmax=+absmax,
                cmap="seismic",
                aspect="equal",
            )
            ax.set_title(f"fsharp_{k}")
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"step {step:05d}    t = {time:.4f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_argmax(
    out_path: Path,
    alphak: np.ndarray,
    time: float,
    step: int,
    ny: int,
    nx: int,
):
    """Render dominant-phase per cell. Phase 4 (background) gets a neutral color so
    the four disks pop out by phase."""
    nphase = alphak.shape[1]
    dominant = alphak.argmax(axis=1)
    img = to_image(dominant.astype(np.int32), ny, nx)

    palette = [
        "#e74c3c",  # 0: red    (top disk)
        "#27ae60",  # 1: green  (left disk)
        "#3498db",  # 2: blue   (right disk)
        "#f1c40f",  # 3: yellow (bottom disk)
        "#ecf0f1",  # 4: light gray (background)
    ]
    cmap = ListedColormap(palette[:nphase])

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.imshow(
        img,
        origin="lower",
        extent=(0, 1, 0, 1),
        cmap=cmap,
        vmin=-0.5,
        vmax=nphase - 0.5,
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_title(f"dominant phase    step {step:05d}    t = {time:.4f}")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def render_summary(out_path: Path, frames_data, ny: int, nx: int):
    """Pick up to 4 evenly-spaced snapshots; render their dominant-phase fields
    side-by-side as a single PNG."""
    n = len(frames_data)
    if n == 0:
        return
    picks = np.linspace(0, n - 1, num=min(4, n), dtype=int)

    palette = [
        "#e74c3c",
        "#27ae60",
        "#3498db",
        "#f1c40f",
        "#ecf0f1",
    ]
    nphase = frames_data[0][1].shape[1]
    cmap = ListedColormap(palette[:nphase])

    fig, axes = plt.subplots(1, len(picks), figsize=(4.0 * len(picks), 4.2))
    if len(picks) == 1:
        axes = [axes]
    for ax, idx in zip(axes, picks):
        step, alphak, time = frames_data[idx]
        dominant = alphak.argmax(axis=1)
        ax.imshow(
            to_image(dominant.astype(np.int32), ny, nx),
            origin="lower",
            extent=(0, 1, 0, 1),
            cmap=cmap,
            vmin=-0.5,
            vmax=nphase - 0.5,
            aspect="equal",
            interpolation="nearest",
        )
        ax.set_title(f"step {step}    t={time:.3f}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("zalesak 5-phase: dominant phase over time")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def maybe_animate(out_dir: Path, prefix: str, output: Path):
    if not shutil.which("ffmpeg"):
        print(f"[postproc] ffmpeg not found on PATH; skipping {output.name}")
        return
    pattern = str(out_dir / f"{prefix}_%05d.png")
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", "10",
        "-i", pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        str(output),
    ]
    print("[postproc]", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"[postproc] wrote {output}")
    except subprocess.CalledProcessError as exc:
        print(f"[postproc] ffmpeg failed: {exc.stderr.decode(errors='replace')[-400:]}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="run directory containing domain/domain.NNNNN.hdf5")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: <run_dir>/postprocess)")
    ap.add_argument("--animate", action="store_true", help="stitch frames into mp4 with ffmpeg if available")
    ap.add_argument("--max-frames", type=int, default=None, help="cap on number of frames to render")
    args = ap.parse_args(argv)

    run_dir = args.run_dir.resolve()
    out_dir = (args.out or (run_dir / "postprocess")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    snaps = discover_snapshots(run_dir)
    if args.max_frames:
        snaps = snaps[: args.max_frames]
    print(f"[postproc] found {len(snaps)} snapshots under {run_dir}")

    summary_data = []
    ny = nx = None

    for path in snaps:
        m = SNAPSHOT_RE.search(path.name)
        step = int(m.group(1))
        alphak, fsharpk, time, verts = read_snapshot(path)
        if ny is None:
            ny, nx = grid_shape(verts)
            print(f"[postproc] inferred grid {ny}x{nx}")

        frame_path = out_dir / f"frame_{step:05d}.png"
        render_frame(frame_path, alphak, fsharpk, time, step, ny, nx)
        argmax_path = out_dir / f"argmax_{step:05d}.png"
        render_argmax(argmax_path, alphak, time, step, ny, nx)

        # Quick numerical sanity report: helps catch a silently-broken run.
        sums = alphak.sum(axis=1)
        if fsharpk is not None:
            fmax = float(np.abs(fsharpk).max())
        else:
            fmax = float("nan")
        print(
            f"  step {step:5d}  t={time:.4f}  "
            f"sum(alphak) in [{sums.min():.4f}, {sums.max():.4f}]  "
            f"max|fsharpk|={fmax:.3e}"
        )

        summary_data.append((step, alphak, time))

    render_summary(out_dir / "summary.png", summary_data, ny, nx)
    print(f"[postproc] wrote {len(snaps)*2 + 1} PNGs to {out_dir}")

    if args.animate:
        maybe_animate(out_dir, "frame", out_dir / "frames.mp4")
        maybe_animate(out_dir, "argmax", out_dir / "argmax.mp4")


if __name__ == "__main__":
    main()
