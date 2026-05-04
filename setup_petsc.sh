#!/bin/bash
# setup_petsc.sh — one-shot PETSc clone + configure for the ablate fork shipped
# in this repository.
#
# Tested on macOS 15 (Sequoia, arm64, Apple clang 17) with Homebrew GCC 15.
# Should work on any modern macOS / Linux box with a working C/C++/Fortran
# toolchain. On macOS, install the prerequisites first via Homebrew:
#
#     brew install gcc cmake autoconf automake libtool libpng pkg-config
#
# Adjust PETSC_DIR below to wherever you want PETSc to live.
set -euo pipefail

PETSC_DIR="${PETSC_DIR:-$HOME/petsc}"
PETSC_ARCH="${PETSC_ARCH:-arch-ablate-opt}"
# PETSc revision ablate is officially tested against (per ablate.dev/#status).
# Newer PETSc main has API drift the fork doesn't compile against
# (PetscErrorMessage 3rd arg became const char**, VLA strictness, etc.).
PETSC_REF="${PETSC_REF:-382a0339}"

if [ ! -d "$PETSC_DIR" ]; then
  echo "==> cloning PETSc into $PETSC_DIR"
  git clone https://gitlab.com/petsc/petsc.git "$PETSC_DIR"
  git -C "$PETSC_DIR" checkout "$PETSC_REF"
else
  echo "==> reusing existing PETSc at $PETSC_DIR"
fi

cd "$PETSC_DIR"

# Configure flags below intentionally exclude opencascade / egads / kokkos:
#   - opencascade / egads: CAD geometry kernels. Used by ablate only for
#     CAD-imported meshes (src/mathFunctions/geom/surface.cpp, currently
#     commented out). The bldenton/oce fork that PETSc downloads fails to
#     compile against the Apple clang 17 / libc++ stdlib on recent macOS SDKs
#     (deleted operator<<(char16_t)). Not needed for the FV-only test cases
#     shipped here.
#   - kokkos: PETSc's pinned commit (3.7.01 era) is incompatible with current
#     master, which requires >= 4.3.00. ablate's kokkos integration is in
#     src/utilities/kokkosUtilities.cpp and src/eos/tChem*; both are commented
#     out of their CMakeLists. Re-enable if you bring chemistry models or
#     GPU paths into scope.
#
# CMake policy fix-up: a few of the bundled deps trigger CMake 4.x compatibility
# warnings without CMAKE_POLICY_VERSION_MINIMUM=3.5.

CMAKE_POLICY_VERSION_MINIMUM=3.5 ./configure \
  PETSC_ARCH="$PETSC_ARCH" \
  --with-debugging=0 \
  --download-mpich \
  --download-mpich-configure-arguments=--disable-two-level-namespace \
  --download-ctetgen \
  --download-tetgen \
  --download-fftw \
  --download-hdf5 \
  --download-metis \
  --download-ml \
  --download-parmetis \
  --download-slepc \
  --download-suitesparse \
  --download-superlu_dist \
  --download-triangle \
  --download-zlib \
  --download-f2cblaslapack \
  --with-slepc

echo ""
echo "==> running make (this is the long step; ~20-40 min on M-series Mac)"
make PETSC_DIR="$PWD" PETSC_ARCH="$PETSC_ARCH" all

echo ""
echo "==> done. Add to your shell rc:"
echo "      export PETSC_DIR=$PWD"
echo "      export PETSC_ARCH=$PETSC_ARCH"
