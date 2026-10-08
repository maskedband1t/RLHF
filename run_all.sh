#!/usr/bin/env sh
# Run every lab script in order. Needs nothing but python3.
cd "$(dirname "$0")/lab" || exit 1
for f in 0*.py; do
  printf '\n################  %s  ################\n\n' "$f"
  python3 "$f" || { echo "FAILED: $f"; exit 1; }
done
