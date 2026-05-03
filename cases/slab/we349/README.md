# we349

Mid-Weber slab variant: derived from `slab72-we1000` by setting
`u₀ = √(5·We) = 41.7732 m/s` (We = 349). Used together with `we126` and
`slab72-we1000` to reproduce three points on dissertation Fig. 4.11
(receding angle vs leading-edge Weber number).

## Numerics

Identical to `slab72-we1000` except for the inlet shear rate:

| | |
|---|---|
| u₀ | 41.7732 m/s |
| We | 349 |
| τ_adv | 2.39 × 10⁻⁴ s |
| ts_max_time / τ_adv | 5.85 |

All other settings (mesh, BCs, IC topology, σ, integrator, dt, ts_max_time,
EOS pair) are identical. See `../slab72-we1000/README.md` for the full setup
discussion.

## Run

```bash
ablate --input cases/slab/we349/input.yaml
```

Expected wall time: ~40 min solo on a 10-core M-series Mac.

## Post-processing

```bash
python tools/postprocess_slab.py --mode plot --invert-display \
  --label "We=349" --tau-adv 2.39e-4 --detect-equilibrium \
  cases/slab/we349/

python tools/animate_slab.py --label "We=349" cases/slab/we349/
```

## Note on equilibration

At ts_max_time = 1.4 × 10⁻³ s the case has completed ~5.9 advective periods —
closer to but still below the ~10 τ_adv window the canonical We=1000 case
sees. The lobe is approaching but has not fully reached its quasi-steady
receding angle.
