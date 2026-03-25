#!/bin/bash
set -e
shopt -s nullglob

BASE_DIR="${1:-data/final}"
OUT_BASE="${BASE_DIR}/public"

PATTERNS=(
  "divers*.pt"
  "subsets/yvi*.pt"
  "subsets/divers_small*.pt"
  "subsets/yvi_small*.pt"
)

for pattern in "${PATTERNS[@]}"; do
  for file in $BASE_DIR/$pattern; do

    # skip if no match (extra safety)
    [[ -e "$file" ]] || continue

    if [[ "$file" == *small* ]]; then
      out_dir="$OUT_BASE/small"
    else
      out_dir="$OUT_BASE/large"
    fi

    mkdir -p "$out_dir"

    python scripts/transform_to_nested.py "$file" "$out_dir"
  done
done