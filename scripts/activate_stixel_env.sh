#!/usr/bin/env bash

# Source this file; do not execute it:
#   source scripts/activate_stixel_env.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${STIXEL_VENV:-$HOME/venvs/stixel}"

if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "Missing venv at $VENV. Run ./scripts/setup_stixel_env.sh first." >&2
  return 1 2>/dev/null || exit 1
fi

# Prevent packages from ~/.local/lib/pythonX.Y/site-packages from leaking in.
export PYTHONNOUSERSITE=1

# Remove stale Python paths inherited from another environment/shell.
unset PYTHONPATH

source "$VENV/bin/activate"
source /opt/ros/jazzy/setup.bash

# Ubuntu's system TBB is older than the interface required by this Numba build.
# Use Numba's OpenMP threading backend instead.
export NUMBA_THREADING_LAYER=omp

if [[ -f "$ROOT/install/setup.bash" ]]; then
  source "$ROOT/install/setup.bash"
fi

export STIXEL_WS="$ROOT"

echo "stixel environment active"
echo "  python: $(command -v python)"

python - <<'PY'
import numpy
print("  numpy:", numpy.__version__)
PY