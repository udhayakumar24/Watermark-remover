#!/usr/bin/env bash
# One-time setup + launch. macOS / Linux.
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "Creating virtualenv…"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "Installing dependencies…"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo "Checking the neural model…"
python scripts/fetch_model.py || echo "(continuing without it — temporal/diffusion/blur still work)"

echo
exec python app.py "$@"
