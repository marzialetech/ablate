#!/usr/bin/env bash
# regenerate zalesak_2row.pdf comparing nosharp_donor (control) vs pm_muscl_g1e-3 (sharpened)
# from the hdf5 trajectories of both zalesak cases
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CASE_DIR="$(cd "$HERE/.." && pwd)"
GROUP_DIR="$(cd "$CASE_DIR/.." && pwd)"

python3 "$HERE/render_indicator_strip.py" \
  "$GROUP_DIR/nosharp_donor" \
  "$CASE_DIR" \
  --labels "no intsharp" "intsharp" \
  --frames 8 \
  --out "$HERE/zalesak_2row.pdf" \
  --no-suptitle
