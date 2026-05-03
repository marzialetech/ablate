# zalesak / nosharp_donor

Three-phase 2D Zalesak rotational advection test (Ch. 3 of the dissertation),
**control baseline** with NPhaseIntSharp disabled and pure donor-cell
advection (no MUSCL slope limiter on the alphak/alphakrhok fields).

This case is the visual + quantitative reference against which sharpened
variants (e.g. `pm_muscl_g1e-3`) are compared. Without sharpening, the slot
opens up and the disks blur over a single revolution.

## Setup

| | |
|---|---|
| domain | 2D unit square, BoxMesh 200 × 200 |
| EOS | three identical `KthStiffenedGas` phases (passive multi-tracer) |
| IC | two slotted disks (k=0, k=1) + background (k=2) at fixed positions |
| velocity | rigid-body rotation ω = 30 rad/s, prescribed via `ZalesakTestSourceTerm` |
| ts_max_time | 2π/30 = 0.20944 s (one full revolution) |
| dt | 1 × 10⁻⁴ s |
| total steps | 2,094 |
| HDF5 snapshots | 21 (interval = 100 steps) |
| **NPhaseIntSharp** | **omitted** from the YAML (sharpening OFF) |
| **MUSCL** | **off** (donor-cell only; coupled to NPhaseIntSharp presence) |

## Run

```bash
ablate --input cases/zalesak/nosharp_donor/input.yaml
```

Expected wall time: ~10 min on a 10-core M-series Mac (single ablate process).

The output goes to a timestamped subdirectory
`zalesak-2disk-3phase-NOSHARP-donor-full_<DATE>/` (because `tagDirectory: true`
is set). The pre-recorded subtree shipped with the case is named
`zalesak-2disk-3phase-NOSHARP-donor-full_2026-05-02T19-40-37/`.

## Post-processing

The shipped figures are built by the `render_indicator_strip.py` and
`postprocess_zalesak.py` tools:

```bash
# multi-row indicator strip; pair this case with pm_muscl_g1e-3 for a side-by-side
python tools/render_indicator_strip.py \
  cases/zalesak/nosharp_donor/zalesak-2disk-3phase-NOSHARP-donor-full_*/domain \
  cases/zalesak/pm_muscl_g1e-3/zalesak-2disk-3phase-pm-muscl-full-g1e3_*/domain \
  --labels "no intsharp" "intsharp" \
  --truth-contour --omega 30 \
  --out figures/zalesak_2row.pdf

# per-snapshot phase + argmax PNGs + (optional) MP4 stitching
python tools/postprocess_zalesak.py \
  cases/zalesak/nosharp_donor/zalesak-2disk-3phase-NOSHARP-donor-full_*
```

## Coupling note

In this codebase, the MUSCL slope-limited reconstruction on the `alphak` and
`alphakrhok` fields is implicitly enabled iff the `NPhaseIntSharp` process is
present in the YAML. Omitting `NPhaseIntSharp` (as this case does) reverts the
advection to first-order donor-cell, which is the original codebase behaviour
and keeps this case useful as a control.
