# Reproducing the cases

End-to-end recipe for building this fork of `ablate` and running the six cases
under `cases/`. All commands assume the repository root as the working directory.

## 1. Prerequisites

A modern C/C++/Fortran toolchain (Apple clang 17 + Homebrew GCC 15 work; recent
GCC on Linux works; Intel ICC has not been tested on this fork). On macOS:

```bash
brew install gcc cmake autoconf automake libtool libpng pkg-config
```

A working Python 3 with `numpy`, `matplotlib`, and `h5py` is needed for
post-processing. From the repository root:

```bash
python3 -m venv .venv-postproc
source .venv-postproc/bin/activate
pip install numpy matplotlib h5py
```

## 2. Build PETSc

A one-shot script is provided. Run it from the repository root:

```bash
./setup_petsc.sh
```

Defaults to cloning into `~/petsc`, building the optimized arch
`arch-ablate-opt`. Override via the env vars `PETSC_DIR`, `PETSC_ARCH`, or
`PETSC_BRANCH` if desired. The script downloads MPICH + a curated list of
PETSc external packages (HDF5, METIS, ParMETIS, SuperLU_DIST, FFTW, SLEPc,
SuiteSparse, MUMPS via SuperLU, and friends) and excludes
opencascade/egads/kokkos which are not exercised by any of the included
cases. See the comments in `setup_petsc.sh` for the rationale.

Wall time: configure ~10–20 min, `make` ~20–40 min on an M-series Mac.

After PETSc finishes, export the env vars it tells you to:

```bash
export PETSC_DIR=$HOME/petsc
export PETSC_ARCH=arch-ablate-opt
```

## 3. Build ablate

```bash
mkdir -p cmake-build-release
cmake -S . -B cmake-build-release -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release -j 6 --target ablate
```

The resulting binary lives at `cmake-build-release/ablate`.

## 4. Run a case

Each case in `cases/<group>/<name>/` ships with its own `input.yaml`, the
recorded HDF5 snapshot trajectory, and the recorded `run.log`. To re-run a
case from scratch:

```bash
# example: the canonical We=1000 slab case
cd cases/slab/slab72-we1000
$REPO_ROOT/cmake-build-release/ablate --input input.yaml
```

Wall times (single ablate process on a 10-core M-series Mac):

| case | wall time |
|---|---|
| `slab/slab72-we1000` | ~40 min |
| `slab/we349`         | ~40 min |
| `slab/we126`         | ~40 min |
| `zalesak/nosharp_donor`         | ~10 min |
| `zalesak/pm_muscl_g1e-3`        | ~10 min |
| `sidi/sidi_4phase_pm_g4e2`      | ~4 min  |

The slab cases are independent and fan-out cleanly across cores; running all
three in parallel finishes in roughly the same wall time as one serial run.

## 5. Post-process

Each case's `README.md` lists the post-processing recipes. The shipped
`postproc/` subtrees (slab cases) and the marquee figures under `figures/`
were produced by these tools and can be regenerated bit-for-bit.

Common entry points:

```bash
# slab — theta(t) trajectory + final-frame lobe + 3 MP4 animations
python3 tools/postprocess_slab.py --mode plot --invert-display \
  --label "We=1000" --tau-adv 1.41e-4 --detect-equilibrium \
  cases/slab/slab72-we1000/
python3 tools/animate_slab.py --label "We=1000" cases/slab/slab72-we1000/

# zalesak — 2-row indicator strip (intsharp vs no intsharp)
python3 tools/render_indicator_strip.py \
  cases/zalesak/nosharp_donor/zalesak-2disk-3phase-NOSHARP-donor-full_*  \
  cases/zalesak/pm_muscl_g1e-3/zalesak-2disk-3phase-pm-muscl-full-g1e3_* \
  --labels "no intsharp" "intsharp" \
  --no-suptitle --truth-contour --omega 30 \
  --contour-color white --contour-linestyle ':' \
  --frames 8 --out figures/zalesak_2row.pdf

# sidi — eps_char(t)/h convergence plot + alpha indicator strip
python3 tools/sidi_thickness.py \
  cases/sidi/sidi_4phase_pm_g4e2/sidi-4phase-pm-g4e2_*/domain \
  --out figures/sidi_pm_g4e2_thickness.pdf
python3 tools/render_sidi_strip.py \
  cases/sidi/sidi_4phase_pm_g4e2/sidi-4phase-pm-g4e2_*/domain \
  --out figures/sidi_pm_g4e2_alpha_strip.pdf

# slab Fig. 4.11 reproduction (consumes the per-case theta_vs_time data)
python3 tools/we_vs_theta.py \
  --runs cases/slab/we126 cases/slab/we349 cases/slab/slab72-we1000 \
  --out figures/we_vs_theta.pdf
```

## 6. Outputs (HDF5 trajectories)

The case directories under `cases/` ship with the recorded HDF5 trajectory
trees. If a fresh clone of this repo is too large for your network, the same
data is also published as zip archives on the GitHub Releases page of this
fork — one zip per case, ~400-600 MB each. Download only the cases you need
and extract them in place under `cases/<group>/<name>/`.

## 7. What's where

```
ablate/
├── REPRODUCE.md            ← this file
├── setup_petsc.sh          ← one-shot PETSc clone + configure + build
├── CMakeLists.txt          ← top-level ablate build
├── src/                    ← ablate source (C++)
├── tools/                  ← post-processing scripts (Python)
├── cases/                  ← reproduction cases (input.yaml + HDF5 + postproc)
│   ├── slab/{slab72-we1000, we349, we126}/
│   ├── zalesak/{nosharp_donor, pm_muscl_g1e-3}/
│   └── sidi/sidi_4phase_pm_g4e2/
└── figures/                ← marquee pre-rendered figures (PDFs and PNGs)
```
