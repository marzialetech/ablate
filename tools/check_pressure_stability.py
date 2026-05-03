#!/usr/bin/env python3
"""Walk every snapshot in a run and report pressure / density / fsharp ranges
plus mixture-density consistency. Used to validate the n-phase intsharp
projection fix: for n-identical-phase tests, p, rho, RHOE should be flat at
the IC values (eps=2.5, p = (gamma-1)*rho*eps = 1.0, rho=1.0)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


def latest_tagged_subdir(run_dir: Path) -> Path:
    if (run_dir / "domain").is_dir():
        return run_dir
    subs = [d for d in run_dir.iterdir() if d.is_dir() and (d / "domain").is_dir()]
    if not subs:
        raise FileNotFoundError(f"no domain/ under {run_dir}")
    subs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return subs[0]


def discover_snapshots(run_dir: Path) -> list[Path]:
    base = latest_tagged_subdir(run_dir)
    return sorted((base / "domain").glob("domain.*.hdf5"))


def get_array(h: h5py.File, candidates: list[str]) -> np.ndarray | None:
    for k in candidates:
        if k in h:
            arr = np.asarray(h[k][0])
            return arr
    return None


def diagnose(snap: Path) -> dict:
    with h5py.File(snap, "r") as h:
        t = float(h["time"][0, 0])
        a = np.asarray(h["cell_fields/solution_alphak"][0])
        ar = np.asarray(h["cell_fields/solution_alphakrhok"][0])
        e = np.asarray(h["cell_fields/solution_allaire"][0])  # [RHOE, RHOU, RHOV]

        p = get_array(h, ["cell_fields/aux_p"])
        rho_aux = get_array(h, ["cell_fields/aux_rho"])
        fsharpk = get_array(h, [
            "cell_fields/aux_fsharpk", "cell_fields/aux_fsharpk0", "cell_fields/fsharpk"
        ])
        # If aux_fsharpk is per-phase, take across all components
        if fsharpk is not None and fsharpk.ndim == 2:
            f = fsharpk
        elif fsharpk is not None:
            f = fsharpk[:, None]
        else:
            f = None

    finite_a = np.isfinite(a).all(axis=1)
    finite_e = np.isfinite(e).all(axis=1)
    finite = finite_a & finite_e
    n = a.shape[0]
    nf = int(finite.sum())

    # mixture density from alphakrhok
    rho_mix = ar.sum(axis=1)
    rhoe = e[:, 0]
    rhou = e[:, 1]
    rhov = e[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(rho_mix > 1e-12, rhou / rho_mix, 0.0)
        v = np.where(rho_mix > 1e-12, rhov / rho_mix, 0.0)
        ke = 0.5 * rho_mix * (u * u + v * v)
        e_int = np.where(rho_mix > 1e-12, (rhoe - ke) / rho_mix, 0.0)

    rec = {
        "snap": snap.name,
        "t": t,
        "finite_frac": nf / n,
    }
    if nf == 0:
        rec.update({"all_nan": True})
        return rec
    sl = finite

    rec.update({
        "rho_min": float(rho_mix[sl].min()),
        "rho_max": float(rho_mix[sl].max()),
        "rho_drift_pct": float(100.0 * np.abs(rho_mix[sl] - 1.0).max()),
        "e_int_min": float(e_int[sl].min()),
        "e_int_max": float(e_int[sl].max()),
        "e_int_drift_pct": float(100.0 * np.abs(e_int[sl] - 2.5).max() / 2.5),
        "alpha_sum_drift": float(np.abs(a[sl].sum(axis=1) - 1.0).max()),
    })

    if p is not None and np.isfinite(p[sl]).all():
        rec.update({
            "p_min": float(p[sl].min()),
            "p_max": float(p[sl].max()),
            "p_drift_pct": float(100.0 * np.abs(p[sl] - 1.0).max()),
        })
    if f is not None:
        ff = f[np.isfinite(f).all(axis=1) if f.ndim == 2 else np.isfinite(f).ravel()]
        if ff.size:
            rec["fsharp_max_abs"] = float(np.abs(ff).max())
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--label", default="run")
    args = ap.parse_args()

    snaps = discover_snapshots(args.run_dir)
    if not snaps:
        print(f"NO_SNAPSHOTS dir={args.run_dir}", file=sys.stderr)
        return 2

    print(f"# label={args.label}  dir={args.run_dir}")
    cols = (
        "snap", "t", "finite_frac",
        "rho_drift_pct", "e_int_drift_pct", "p_drift_pct", "alpha_sum_drift",
        "rho_min", "rho_max", "p_min", "p_max",
        "e_int_min", "e_int_max",
        "fsharp_max_abs",
    )
    fmts = {
        "snap": "{:>22}",
        "t": "{:>9.5f}",
        "finite_frac": "{:>6.3f}",
        "rho_drift_pct": "{:>8.2e}",
        "e_int_drift_pct": "{:>8.2e}",
        "p_drift_pct": "{:>8.2e}",
        "alpha_sum_drift": "{:>9.2e}",
        "rho_min": "{:>8.4f}",
        "rho_max": "{:>8.4f}",
        "p_min": "{:>8.4f}",
        "p_max": "{:>8.4f}",
        "e_int_min": "{:>8.4f}",
        "e_int_max": "{:>8.4f}",
        "fsharp_max_abs": "{:>8.2e}",
    }
    header = " | ".join(fmts[c].replace("d", "s").replace(".5f", "s").replace(".3f", "s")
                         .replace(".2e", "s").replace(".4f", "s").replace(".2e", "s")
                         .format(c) for c in cols)
    print(header)
    print("-" * len(header))
    for snap in snaps:
        rec = diagnose(snap)
        if rec.get("all_nan"):
            print(f"{snap.name:>22} | t={rec['t']:.5f} | ALL NaN")
            continue
        cells = []
        for c in cols:
            if c == "snap":
                cells.append(fmts[c].format(rec.get(c, snap.name)))
            else:
                v = rec.get(c, float("nan"))
                cells.append(fmts[c].format(v if v is not None else float("nan")))
        print(" | ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
