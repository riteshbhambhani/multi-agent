#!/usr/bin/env bash
# Setup script to create a Python 3.11 virtualenv and install backend requirements.
# Run from repository root: ./scripts/setup_env.sh
set -euo pipefail

PYTHON=${PYTHON:-}

# detect python3.11
if [[ -n "$PYTHON" ]]; then
  PY=$PYTHON
elif command -v python3.11 >/dev/null 2>&1; then
  PY=$(command -v python3.11)
elif command -v pyenv >/dev/null 2>&1 && pyenv versions --bare | grep -q "^3.11"; then
  PY=$(pyenv which python)
else
  echo "Python 3.11 not found. Install with Homebrew: 'brew install python@3.11' or use pyenv." >&2
  exit 2
fi

echo "Using python: $PY"

rm -rf .venv
$PY -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r backend/requirements.txt

echo "Virtualenv .venv created and dependencies installed."