#!/bin/bash
# Postproc orchestrator for the Weber-sweep slab runs.
# For each case: extract theta(t), render lobe_final.png + lobe_zoom.png,
# build the 3 animation MP4s, and produce a CSV row for the composite plot.
set -e
cd "$(dirname "$0")/.."

ROOT=$(pwd)
TOOLS=$ROOT/tools

# (We, u0) pairs covering the dissertation Fig 4.10 set + the existing We=1000.
declare -a CASES=(
  "10|7.07106781186548|runs/slab/we10"
  "28|11.83215956619923|runs/slab/we28"
  "46|15.16575089103551|runs/slab/we46"
  "76|19.49358868961793|runs/slab/we76"
  "126|25.09980079602226|runs/slab/we126"
  "210|32.40370349203930|runs/slab/we210"
  "349|41.77319714841085|runs/slab/we349"
  "580|53.85164807134504|runs/slab/we580"
  "1000|70.71067811865476|runs/slab/slab72-extended_repro"
)

postproc_one() {
  local We=$1 u0=$2 d=$3
  local L=0.01
  # tau_adv = L / u0
  local tau_adv=$(awk "BEGIN{printf \"%.6e\", $L/$u0}")
  local label="We=${We}"
  echo "[We=${We}] postproc_slab.py --mode plot ..."
  python3 "$TOOLS/postprocess_slab.py" \
    --mode plot --invert-display \
    --label "$label" --tau-adv "$tau_adv" \
    --detect-equilibrium \
    "$d" > "$d/postproc.log" 2>&1 || echo "  (postproc_slab returned non-zero; continuing)"
  echo "[We=${We}] animate_slab.py --variant all ..."
  python3 "$TOOLS/animate_slab.py" \
    --label "$label" \
    "$d" >> "$d/postproc.log" 2>&1 || echo "  (animate_slab returned non-zero; continuing)"
}

# Run all 9 in parallel (each is mostly single-threaded matplotlib + ffmpeg).
PIDS=()
for case in "${CASES[@]}"; do
  We=${case%%|*}
  rest=${case#*|}
  u0=${rest%%|*}
  d=${rest#*|}
  postproc_one "$We" "$u0" "$d" &
  PIDS+=($!)
done

for p in "${PIDS[@]}"; do
  wait "$p" || true
done
echo "ALL POSTPROC DONE"
