# NPhaseIntSharp `chiu_lin` form — implementation notes

This file documents the current state of the `form: chiu_lin` branch in
`nPhaseIntSharp.cpp`. Read this before changing the CL implementation or
choosing between PM and CL for production runs.

## Analytical form (what we're discretizing)

For each phase k:

    d alpha_k / d tau = div( F_k )
    F_k = Gamma_k * [ epsilon_k * grad(alpha_k)
                       - alpha_k (1 - alpha_k) * grad(alpha_k) / |grad(alpha_k)| ]

The first term is diffusive (smoothing); the second is antidiffusive
(sharpening). The two balance when `alpha_k(x)` takes the steady tanh shape
of width `epsilon_k`. This is the same Chiu–Lin / Olsson–Kreiss form
implemented for the single-phase reference `intSharp-marziale.cpp`.

## Discretization (current)

The CL branch evaluates `F_k . n` at each mesh face by:

    dalpha_dn = (alpha_R - alpha_L) / |L->R|        # 2-point FD, monotone
    alpha_f   = 0.5 * (alpha_L + alpha_R)
    sign_g    = sign(alpha_R - alpha_L)
    F . n     = Gamma * eps * dalpha_dn  -  Gamma * af*(1-af) * sign_g

then accumulates the divergence into both adjacent cells:

    fsharp_L += F . n_LR * area_face / V_L
    fsharp_R -= F . n_LR * area_face / V_R

This is a standard Green-Gauss face-flux integration. It is
**conservative by construction** (every flux is shared between two cells
with opposite sign) and uses **monotone face-FD gradients**, so it does
not have the LSQ-cell-gradient aliasing problem of the previous
implementation.

## Why this branch is currently EXPERIMENTAL (do not ship for dissertation reproduction)

Visually we observe two failure modes for the current face-based CL on
the 3-phase / 2-disk test bench (`runs/sweeps/zalesak_2disk_3phase_chiulin_g*_face`):

1. **High Gamma (>= 5e-3)**: per-step alpha increment via
   `(F.n)*area/V ~ Gamma/h` exceeds 1, saturating cells at alpha=0/1
   in a single PreStage. The result is a rotated-square crystallization
   pattern that NaN's in 2-3 frames. This is a CFL violation of the
   sharpening operator itself, separable from any physical EOS coupling.

2. **Low Gamma (<= 1e-3)**: per-step magnitude is bounded but the
   antidiffusive flux still fires at every face that has a nonzero
   `(alpha_R - alpha_L)`, including faces where the gradient is just
   numerical noise. The result is **rough/jagged disk boundaries** -- the
   antidiffusive flux flips boundary cells across phase membership at
   sub-grid scale, producing edge speckle. The rms boundary roughness
   grows monotonically with step count.

These two modes bracket the operator: too strong -> NaN; weak enough not
to NaN -> insufficient gradient gating, edge noise.

PM (`form: parameswaran_mandal`) does NOT have either issue at the same
test setup. PM's pointwise scalar source is bounded by Gamma * 0.25 per
cell (no 1/h amplification), and it doesn't have an antidiffusive flux
that fires on noise gradients (it's a polynomial in alpha, vanishing
at alpha = 0 or 1).

## What is needed to make CL production-ready

1. **TVD flux limiter** (e.g. minmod or van Leer) on the antidiffusive
   contribution at each face. Skip the antidiffusive flux when both
   sides are deep in the same phase or when the gradient magnitude
   relative to a noise threshold is below 1.

2. **Gradient-magnitude gate**: skip face flux when `|alpha_R - alpha_L|
   < threshold` (e.g. 1e-4). This eliminates the noise-driven boundary
   speckling at low Gamma.

3. **Pseudo-time substepping**: add a CFL-respecting `dtau` to bound the
   per-PreStage alpha increment to <= 0.1 (similar to marziale's
   `pseudoTime = 1e-4` hardcode in `intSharp-marziale.cpp:591`). This
   would let users specify Gamma in PM-comparable units and have the
   code substep internally.

4. (alternative) **Vertex-centered gradients + face-loop divergence**,
   matching `intSharp-marziale.cpp` exactly. This is what works in the
   single-phase reference. It's more invasive than (1)-(3) and has
   compatible interface gating built in.

Any of these by itself probably wouldn't be enough; production CL
likely needs (1) + (2) + (3) together (or (4) which subsumes them).

## Test-bench evidence

See `runs/sweeps/zalesak_2disk_3phase_chiulin_g*_face/` for the
100-step gates that produced the diagnoses above, and
`tools/figs/cl_face_low_gamma_step100.png` for the side-by-side
visualization vs PM and NOSHARP.

The PM branch has independent visual + quantitative validation as a
quarter-rotation case in `runs/sweeps/zalesak_2disk_3phase_pm_qr/`
(slot_fraction goes from 0.0217 to 0.0179 over the rotation, mass_drift
stays at machine precision, max_alpha_disk stays >= 0.999, no NaN).

## Related fixes that landed alongside the CL refactor

The following changes were made to the n-phase chain to make the
*background* of the Zalesak test cleanly decoupled from the
phase-indicator dynamics. They are general-purpose and apply to BOTH
PM and CL:

1. **`nPhaseIntSharp.cpp` projection update** — after applying fsharp
   to alphak, the cell now: clips alpha to [0,1], renormalizes sum_k
   alphak = 1, rebuilds alphakrhok = alphak * rhokold (with rhoold
   fallback for newly-appearing phases), and rebuilds RHOU and RHOE
   using cached velocity and specific internal energy. This makes
   intsharp a pure phase-indicator regularization with zero EOS
   footprint for n-identical-phase test cases (verified: alpha_sum
   drift = 2.22e-16 / mass_drift = 6.5e-11 over 2094-step QR).

2. **`nPhaseAllaireAdvection.cpp` ZalesakTestSourceTerm** — extended
   the rotational forcing to also pin per-phase mass (`alphakrhok =
   alphak * rho_const`) and total energy (`RHOE = rho_const * (eps +
   0.5*v^2)`) to their IC-uniform target values per step. Without this,
   the underlying Allaire AUSM+up + nonconservative chain accumulates
   ~1%/100-step drift in rho/eps/p at sharp alpha interfaces (verified
   by NOSHARP control). After the fix the test is reduced to a pure
   phase-indicator advection on a uniform rotational background, which
   is the dissertation's intent.

These two fixes also reduced PM's `max_fsharp` and made it more
predictable / less sensitive to dt — quarter rotation at Gamma=4e-2
now runs cleanly with `max_fsharp ~= 4e-3` consistently.
