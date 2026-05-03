# slab72-we1000

Canonical We=1000 slab-burner lobe study from the dissertation Sec. 4.4
(Fig. 4.10). The lobe morphology is integrated for 10·tau_adv to give it time
to settle into a quasi-steady receding angle.

## Numerics


|                         |                                              |
| ----------------------- | -------------------------------------------- |
| domain                  | 2D box, Gmsh `slabBurner2DMesh.msh`          |
| EOS pair                | `(PerfectGas, StiffenedGas)` (air, paraffin) |
| surface tension         | `SurfaceForce` enabled, σ = 50 N/m           |
| u₀ (leading-edge shear) | 70.711 m/s (We = ρ_l u₀² L / σ = 1000)       |
| τ_adv = L_lobe / u₀     | 1.41 × 10⁻⁴ s                                |
| ts_max_time             | 1.4 × 10⁻³ s ≈ 10 τ_adv                      |
| dt                      | 5 × 10⁻⁸ s                                   |
| integrator              | RK2 (Heun)                                   |
| total steps             | 28,000                                       |
| HDF5 snapshots          | 141 (interval = 200 steps)                   |


## Run

From the repository root:

```bash
ablate --input cases/slab/slab72-we1000/input.yaml
```

Expected wall time: ~40 min on a 10-core M-series Mac (single ablate process).
The output goes to `cases/slab/slab72-we1000/slab2dcoords/` (mirrors the
shipped pre-recorded subtree).

## Post-processing

```bash
# theta(t) trajectory + final-frame lobe figures
python tools/postprocess_slab.py --mode plot --invert-display \
  --label "We=1000" --tau-adv 1.41e-4 --detect-equilibrium \
  cases/slab/slab72-we1000/

# three MP4 animations: lobe-zoom, full-domain context, combined panel
python tools/animate_slab.py --label "We=1000" cases/slab/slab72-we1000/
```

Post-processed artifacts under `postproc/`:

- `theta_vs_time.png`         — receding-angle trajectory with equilibrium detection
- `lobe_final.png`, `lobe_zoom.png` — final-frame lobe morphology
- `animation/lobe_animation.mp4`     — tight zoom on the lobe head
- `animation/context_animation.mp4`  — full domain with red-box overlay
- `animation/combined_animation.mp4` — side-by-side composite

## Notes on the codebase

- `TwoPhaseEulerAdvection` enforces (PerfectGas, StiffenedGas) ordering
internally (`twoPhaseEulerAdvection.cpp:836-845`); the deck keeps that order
and the post-processor inverts the alpha display so the lobe shows up at
vof = 1 in figures.
- The level-set sign is `phi = slabTop − y` (positive inside the slab) so that
`VOF=0` inside the slab (paraffin), per the convention in
`levelSet/Utilities/VOF.cpp:53-58`.
- `SurfaceForce` is active. The C++ neighbor cache used to be hardcoded at
3 layers; this codebase plumbs `C` and `N` through to the runtime stencil so
the smoother actually honors the YAML.

