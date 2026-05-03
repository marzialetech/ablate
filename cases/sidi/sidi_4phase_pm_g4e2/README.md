# sidi / sidi_4phase_pm_g4e2

Static Initially-Diffuse Interface (SIDI) test from dissertation
Sec. 2.1.2.3.1 — four phases, three disks at 120° angular spacing about the
domain center against a fourth background phase. The flow is quiescent
(velocity ≡ 0); only NPhaseIntSharp acts on the interface band, exercising
the sharpening operator in isolation from any advective dynamics.

The IC is intentionally diffuse: each disk's αₖ profile is a `tanh` ramp with
characteristic interface width ε_char/h ≈ 8.5. Under sharpening with
γ = 4 × 10⁻² (PM form), the band collapses to ε_char/h ≈ 0.14 in roughly 400
time steps and remains stable thereafter.

## Setup

| | |
|---|---|
| domain | 2D unit square, BoxMesh 200 × 200 |
| EOS | four identical `KthStiffenedGas` phases |
| IC | three `tanh` disks at 120° apart + diffuse background |
| velocity | identically zero (quiescent) |
| dt | 2.5 × 10⁻⁵ s |
| ts_max_steps | 2,000 |
| HDF5 snapshots | 13 (interval = 50 steps initially, sparser later) |
| **NPhaseIntSharp** | enabled, form = `parameswaran-mandal` |
| Γₖ | [γ, γ, γ, γ] with γ = 4 × 10⁻² |
| εₖ | [h, h, h, h] |
| BCs | `EssentialGhost` mirroring IC values on all walls |

## Run

```bash
ablate --input cases/sidi/sidi_4phase_pm_g4e2/input.yaml
```

Expected wall time: ~4 min on a 10-core M-series Mac.

The pre-recorded subtree is
`sidi-4phase-pm-g4e2_2026-05-02T21-37-23/`.

## Post-processing

```bash
# extract eps_char(t)/h trajectories per disk and plot convergence
python tools/sidi_thickness.py \
  cases/sidi/sidi_4phase_pm_g4e2/sidi-4phase-pm-g4e2_*/domain \
  --out figures/sidi_pm_g4e2_thickness.pdf

# multi-snapshot strip of the alpha indicator with analytical disk contours
python tools/render_sidi_strip.py \
  cases/sidi/sidi_4phase_pm_g4e2/sidi-4phase-pm-g4e2_*/domain \
  --out figures/sidi_pm_g4e2_alpha_strip.pdf
```

## Conservation checks observed during the recorded run

- ρ drift over 2,000 steps:  < 0.01 %
- p drift:                  < 0.01 %
- αₖ remained in [0, 1] at all sampled cells
- 0.04 % of cells classified as "diffuse" (0.05 ≤ αₖ ≤ 0.95) at the plateau

These confirm the sharpening operator is well-behaved on the quiescent IC and
provide the validation point cited in the dissertation section.
