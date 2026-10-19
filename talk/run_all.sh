#!/usr/bin/env bash
# Rebuild everything for the talk (~1 h on an RTX 5060): sweep -> final models -> figures/tables.
set -euo pipefail
cd "$(dirname "$0")"
python tune.py     | tee results/tune.log
python final.py    | tee results/final.log
python evaluate.py | tee results/evaluate.log
