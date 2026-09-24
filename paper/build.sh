#!/usr/bin/env bash
# Build the paper with tectonic (one binary, fetches packages on first use; bibtex runs by itself),
# then print the page count. Usage: bash paper/build.sh ; view with: open paper/main.pdf
set -euo pipefail
cd "$(dirname "$0")"
TECTONIC=${TECTONIC:-$HOME/.local/bin/tectonic}
"$TECTONIC" --keep-logs main.tex
PYTHON=../.venv/bin/python
if [ -x "$PYTHON" ]; then
  "$PYTHON" -c "import pymupdf; print('pages:', len(pymupdf.open('main.pdf')))"
fi
grep -E "Overfull|Underfull.*badness 10000" main.log | head -5 || true
