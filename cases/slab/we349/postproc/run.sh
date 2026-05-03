#!/usr/bin/env bash
# regenerate combined_animation.mp4 for this case from the hdf5 trajectory
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CASE_DIR="$(cd "$HERE/.." && pwd)"
python3 "$HERE/animate_slab.py" "$CASE_DIR" --variant combined --label we349
