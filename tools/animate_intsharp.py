#!/usr/bin/env python3
"""
Render a per-snapshot side-by-side animation comparing intsharp vs control runs
of the 5-phase Zalesak test.

Layout per frame (one row, four panels):
    [intsharp argmax]  [intsharp alpha-heatmap (background phase)]
    [control  argmax]  [control  alpha-heatmap (background phase)]

The argmax view shows where each phase "wins" the cell; the alpha-heatmap on the
background phase (phase 4) is the most readable single field for showing the
sharpness difference because the four disks appear as dark holes punched through
a yellow background and the rim transition reveals diffusion vs sharpening.

After rendering frames, optionally invokes ffmpeg to assemble a video.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def discover_snapshots(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "domain").glob("domain.*.hdf5"))


def read_snapshot(path: Path):
    with h5py.File(path, "r") as f:
        alphak = f["cell_fields/solution_alphak"][0]
        time = float(f["time"][0, 0])
    return alphak, time


def infer_grid(ncell: int) -> tuple[int, int]:
    nx = int(round(math.sqrt(ncell)))
    if nx * nx != ncell:
        raise ValueError(f"{ncell} cells does not factor as a square mesh")
    return nx, nx


def render_frame(
    out_png: Path,
    i_alpha: np.ndarray,
    c_alpha: np.ndarray,
    t: float,
    bg_phase: int,
) -> None:
    nx, ny = infer_grid(i_alpha.shape[0])

    def argmax_image(a):
        s = a.sum(axis=1)
        finite = np.isfinite(a).all(axis=1) & (s > 1e-9)
        idx = np.full(a.shape[0], -1, dtype=int)
        idx[finite] = np.argmax(a[finite], axis=1)
        return idx.reshape(ny, nx)

    fig, axes = plt.subplots(2, 2, figsize=(11, 11))

    axes[0, 0].imshow(argmax_image(i_alpha), origin="lower", cmap="tab10",
                      vmin=-1, vmax=i_alpha.shape[1] - 1, aspect="equal")
    axes[0, 0].set_title("intsharp on  |  argmax phase", fontsize=11)
    axes[0, 1].imshow(i_alpha[:, bg_phase].reshape(ny, nx), origin="lower",
                      cmap="viridis", vmin=0, vmax=1, aspect="equal")
    axes[0, 1].set_title(f"intsharp on  |  alpha_{bg_phase} (background)", fontsize=11)

    axes[1, 0].imshow(argmax_image(c_alpha), origin="lower", cmap="tab10",
                      vmin=-1, vmax=c_alpha.shape[1] - 1, aspect="equal")
    axes[1, 0].set_title("control (off)  |  argmax phase", fontsize=11)
    axes[1, 1].imshow(c_alpha[:, bg_phase].reshape(ny, nx), origin="lower",
                      cmap="viridis", vmin=0, vmax=1, aspect="equal")
    axes[1, 1].set_title(f"control (off)  |  alpha_{bg_phase} (background)", fontsize=11)

    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])

    omega = 30.0
    deg = (omega * t * 180.0 / math.pi) % 360.0
    fig.suptitle(f"5-phase Zalesak rotation  |  t = {t:.4f}s  ({deg:6.1f}°)",
                 fontsize=13, y=0.995)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intsharp", required=True, type=Path)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path,
                        help="output dir for frames + mp4")
    parser.add_argument("--bg-phase", type=int, default=4,
                        help="phase index for the alpha-heatmap panel (default 4)")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    frames_dir = args.out / "frames"
    frames_dir.mkdir(exist_ok=True)

    i_snaps = discover_snapshots(args.intsharp)
    c_snaps = discover_snapshots(args.control)
    n = min(len(i_snaps), len(c_snaps))
    print(f"[animate] {n} matched frames (intsharp={len(i_snaps)}, control={len(c_snaps)})")

    for k in range(n):
        i_alpha, i_t = read_snapshot(i_snaps[k])
        c_alpha, c_t = read_snapshot(c_snaps[k])
        if abs(i_t - c_t) > 1e-9:
            print(f"[animate] WARN: time mismatch at frame {k}: i={i_t}, c={c_t}")
        out_png = frames_dir / f"frame_{k:04d}.png"
        render_frame(out_png, i_alpha, c_alpha, i_t, args.bg_phase)
        if k % 5 == 0 or k == n - 1:
            print(f"  frame {k:>4d}/{n - 1}  t={i_t:.4f}s")

    if args.no_video or shutil.which("ffmpeg") is None:
        print(f"[animate] frames at {frames_dir}; ffmpeg skipped")
        return 0

    mp4 = args.out / "comparison.mp4"
    cmd = [
        "ffmpeg", "-y", "-framerate", str(args.fps),
        "-i", str(frames_dir / "frame_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        str(mp4),
    ]
    print("[animate] running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[animate] wrote {mp4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
