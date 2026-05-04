# ablate — ip case set

fork of [UBCHREST/ablate](https://github.com/UBCHREST/ablate) with reproducer
inputs, postprocessing, and outputs for the ip case set:
zalesak (sharpened + control), slab72 weber sweep (we1000/we349/we126), and
sidi 4-phase. trajectories ship as zip assets on the
[ip-v1 release](https://github.com/marzialetech/ablate/releases/tag/ip-v1).

## prerequisites

```bash
# macos
brew install gcc cmake autoconf automake libtool libpng pkg-config
# debian/ubuntu
sudo apt install gcc g++ gfortran cmake autoconf automake libtool libpng-dev pkg-config python3 python3-pip
```

```bash
# python deps for postprocessing
python3 -m venv .venv && source .venv/bin/activate
pip install h5py matplotlib numpy scikit-image
```

`ffmpeg` on `$PATH` is required for the slab animation step.

## build

```bash
# build petsc (~20-40 min). installs to $HOME/petsc by default; override with
# PETSC_DIR. excludes opencascade/egads/kokkos (not used by the ip cases).
./setup_petsc.sh

# export petsc paths into your shell
export PETSC_DIR=$HOME/petsc
export PETSC_ARCH=arch-ablate-opt
export PKG_CONFIG_PATH="$PETSC_DIR/$PETSC_ARCH/lib/pkgconfig:$PKG_CONFIG_PATH"

# build ablate. CMAKE_POLICY_VERSION_MINIMUM is needed because bundled deps
# (yaml-cpp, mu-parser, json) declare cmake_minimum_required < 3.5; -j4 caps
# parallelism so the C++ frontend doesn't OOM 16 GB machines.
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build -j4 --target ablate
```

verify:

```bash
./build/ablate --help
```

## reproduce a case from scratch

each case lives in `cases/<group>/<name>/`. its `input.yaml` is the ablate deck;
`postproc/run.sh` regenerates the figure(s) once the run finishes.

```bash
# pick any case; this example uses sidi (the smallest standalone case)
cd cases/sidi/sidi_4phase_pm_g4e2

# run ablate; writes hdf5 trajectory into a timestamped subdir of this cwd
../../../build/ablate --input input.yaml

# regenerate the figures into ./postproc/
./postproc/run.sh
ls postproc/*.pdf
```

note: `cases/zalesak/pm_muscl_g1e-3/postproc/run.sh` requires the
`cases/zalesak/nosharp_donor/` trajectory (control side of the comparison
plot). run nosharp_donor first, then pm_muscl_g1e-3, then postproc.

## reproduce figures from released trajectories (no build required)

skip the ablate build entirely if you only want to regenerate the figures:

```bash
gh release download ip-v1 --repo marzialetech/ablate \
    --pattern 'sidi_sidi_4phase_pm_g4e2.zip' \
    --dir cases/sidi/sidi_4phase_pm_g4e2

cd cases/sidi/sidi_4phase_pm_g4e2
unzip -q sidi_sidi_4phase_pm_g4e2.zip
./postproc/run.sh
```

apply the same pattern for any other case. zip layout matches the live-run
output layout, so postproc scripts work either way.

## case index

| case | path | release zip | figure(s) |
|---|---|---|---|
| zalesak control | `cases/zalesak/nosharp_donor/` | `zalesak_nosharp_donor.zip` | (control for pm_muscl plot) |
| zalesak sharpened | `cases/zalesak/pm_muscl_g1e-3/` | `zalesak_pm_muscl_g1e-3.zip` | `zalesak_2row.pdf` |
| slab we=1000 | `cases/slab/slab72-we1000/` | `slab_slab72-we1000.zip` | `combined_animation.mp4` |
| slab we=349 | `cases/slab/we349/` | `slab_we349.zip` | `combined_animation.mp4` |
| slab we=126 | `cases/slab/we126/` | `slab_we126.zip` | `combined_animation.mp4` |
| sidi 4-phase | `cases/sidi/sidi_4phase_pm_g4e2/` | `sidi_sidi_4phase_pm_g4e2.zip` | `sidi_pm_g4e2_alpha_strip.pdf`, `sidi_pm_g4e2_thickness.pdf` |

## upstream

base fork: [UBCHREST/ablate](https://github.com/UBCHREST/ablate). this branch
is squashed onto upstream `main` so the diff is exactly the ip case set plus
the source changes that produced it.
