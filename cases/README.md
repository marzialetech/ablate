# Reproduction cases

Six cases organized by physics group. Each subdirectory contains an `input.yaml`,
the necessary mesh/aux files, the recorded HDF5 snapshot trajectory, and the
pre-rendered post-processing artifacts.


| group   | case                  | dissertation reference         | snapshots | wall time (M-series Mac) |
| ------- | --------------------- | ------------------------------ | --------- | ------------------------ |
| slab    | `slab72-we1000`       | Sec. 4.4, Fig. 4.10 (We=1000)  | 141       | ~40 min                  |
| slab    | `we349`               | Fig. 4.11 (We=349 datum)       | 141       | ~40 min                  |
| slab    | `we126`               | Fig. 4.11 (We=126 datum)       | 141       | ~40 min                  |
| zalesak | `nosharp_donor`       | Ch. 3 — control baseline       | 21        | ~10 min                  |
| zalesak | `pm_muscl_g1e-3`      | Ch. 3 — sharpened (PM, γ=1e-3) | 21        | ~10 min                  |
| sidi    | `sidi_4phase_pm_g4e2` | Sec. 2.1.2.3.1 — 4-phase SIDI  | 13        | ~4 min                   |


For build instructions and the recommended end-to-end reproduction sequence, see
`../REPRODUCE.md` at the repository root.

## Per-case directory layout

Each case directory contains:

- `input.yaml`             — the deck consumed by `ablate --input`
- `run.log`                — captured stdout from the recorded run
- `slabBurner2DMesh.msh` *(slab cases only)* — Gmsh mesh used by `dm_plex_filename`
- `slab2dcoords/` *(slab cases)* or `<title>_<timestamp>/` *(zalesak/sidi)*
— the HDF5 snapshot subtree written by ablate's HDF5 monitor; this is the raw
data that all post-processing tools consume
- `postproc/` *(slab cases only)* — `theta_vs_time.png`, `lobe_final.png`,
`lobe_zoom.png`, plus an `animation/` subdir containing the three MP4s
(`lobe_animation.mp4`, `context_animation.mp4`, `combined_animation.mp4`) and
per-frame PNG dirs
- `README.md`              — case-specific notes (this file)

The pre-rendered `postproc/` artifacts let you inspect results without re-running.
To rebuild any plot, follow the post-processing recipe in the case's README.