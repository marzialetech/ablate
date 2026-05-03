#!/usr/bin/env bash
# regenerate sidi_pm_g4e2_alpha_strip.pdf and sidi_pm_g4e2_thickness.pdf
# from the hdf5 trajectory of this case
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CASE_DIR="$(cd "$HERE/.." && pwd)"

python3 "$HERE/render_sidi_strip.py" "$CASE_DIR" \
  --labels "g=4e-2" \
  --centers "0.5,0.77" "0.26617,0.365" "0.73383,0.365" \
  --R 0.125 \
  --frames 6 \
  --out "$HERE/sidi_pm_g4e2_alpha_strip.pdf"

python3 "$HERE/sidi_thickness.py" "$CASE_DIR" \
  --labels "g=4e-2" \
  --centers "0.5,0.77" "0.26617,0.365" "0.73383,0.365" \
  --h 0.005 \
  --eps0-over-h 8 \
  --out "$HERE/sidi_pm_g4e2_thickness.pdf"
