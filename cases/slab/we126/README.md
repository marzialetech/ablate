# we126

Lower-Weber slab variant: derived from `slab72-we1000` by setting
`u₀ = √(5·We) = 25.0998 m/s` (We = 126). Used together with `we349` and
`slab72-we1000` to reproduce three points on dissertation Fig. 4.11
(receding angle vs leading-edge Weber number).

## Numerics

Identical to `slab72-we1000` except for the inlet shear rate:

| | |
|---|---|
| u₀ | 25.0998 m/s |
| We | 126 |
| τ_adv | 3.98 × 10⁻⁴ s |
| ts_max_time / τ_adv | 3.51 |

All other settings (mesh, BCs, IC topology, σ, integrator, dt, ts_max_time,
EOS pair) are identical. See `../slab72-we1000/README.md` for the full setup
discussion.

## Run

```bash
ablate --input cases/slab/we126/input.yaml
```

Expected wall time: ~40 min solo on a 10-core M-series Mac.

## Post-processing

```bash
python tools/postprocess_slab.py --mode plot --invert-display \
  --label "We=126" --tau-adv 3.98e-4 --detect-equilibrium \
  cases/slab/we126/

python tools/animate_slab.py --label "We=126" cases/slab/we126/
```

## Caveat

At ts_max_time = 1.4 × 10⁻³ s the case has only completed ~3.5 advective
periods — short of the ~10 τ_adv window the canonical We=1000 case enjoys.
The lobe at the run endpoint has not yet equilibrated; this is expected and
discussed in the figure interpretation. To reach an equivalent number of
advective periods at We=126 the case would need to be extended to ~4 × 10⁻³ s
(roughly 4× the wall time).
