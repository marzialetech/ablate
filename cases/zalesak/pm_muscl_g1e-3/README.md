# zalesak / pm_muscl_g1e-3

Three-phase 2D Zalesak rotational advection test, **sharpened variant**:
NPhaseIntSharp active in the Parameswaran–Mandal (PM) form with
γ = 1 × 10⁻³, plus second-order MUSCL reconstruction with a Barth–Jespersen
slope limiter on the `alphak` / `alphakrhok` fields.

Companion to `nosharp_donor`. The slot is preserved noticeably better through
a full revolution; the diffuse interface band is thinner.

## Setup

Identical to `nosharp_donor` for mesh, EOS, IC, velocity prescription,
ts_max_time, dt, and snapshot cadence. The differences are:

| | |
|---|---|
| **NPhaseIntSharp** | **enabled**, form = `parameswaran-mandal` |
| Γₖ (sharpening magnitude) | [γ, γ, γ] with γ = 1 × 10⁻³ |
| εₖ (interface width target) | [h, h, h] (one cell width) |
| **MUSCL on αₖ / αₖρₖ** | **on** (Barth–Jespersen 2D limiter), implicit-coupled to the presence of NPhaseIntSharp |

## Run

```bash
ablate --input cases/zalesak/pm_muscl_g1e-3/input.yaml
```

Expected wall time: ~10 min on a 10-core M-series Mac.

The pre-recorded subtree is
`zalesak-2disk-3phase-pm-muscl-full-g1e3_2026-05-02T21-24-56/`.

## Post-processing

See `../nosharp_donor/README.md` for the side-by-side `render_indicator_strip.py`
recipe (it pairs the two cases into the figure shipped at
`figures/zalesak_2row.pdf`).

Per-case quick-look:

```bash
python tools/postprocess_zalesak.py \
  cases/zalesak/pm_muscl_g1e-3/zalesak-2disk-3phase-pm-muscl-full-g1e3_*
```

## Why γ = 1 × 10⁻³ specifically

This is a deliberate mid-range choice from a γ ∈ {0, 1e-4, 1e-3, 1e-2, 2e-2,
4e-2} sweep that demonstrated monotonic slot-preservation improvement with
increasing γ, with full numerical stability across the entire range. γ = 4e-2
gives the sharpest interface but γ = 1e-3 is sharp enough to make the
difference visually obvious while sitting in the safest part of the stability
envelope. Choose a higher γ for production use if maximum interface
crispness is needed.
