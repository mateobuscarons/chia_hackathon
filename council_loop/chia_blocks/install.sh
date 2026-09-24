#!/bin/bash
# Install the blocks into a Python environment that holds the released CHIA package (chialoops):
# each block and its tests are copied to the place they hold in CHIA's tree, then the patches
# are applied to CHIA's own files. Idempotent. Usage: bash install.sh <venv directory>
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
VENV=$(cd "${1:?usage: bash install.sh <venv directory>}" && pwd)   # absolute: the copy below runs from /tmp
SITE=$(cd /tmp && "$VENV/bin/python" -c "import chia; print(list(chia.__path__)[0])")
cp "$HERE/chia/trace/spend.py" "$HERE/chia/trace/ledger.py" "$SITE/trace/"
cp "$HERE/chia/models/decider.py" "$HERE/chia/models/call_gate.py" "$HERE/chia/models/context_gate.py" "$HERE/chia/models/vertex_json.py" "$SITE/models/"
cp "$HERE/chia/simulators/champsim_config.py" "$SITE/simulators/"
mkdir -p "$SITE/trace/tests" "$SITE/models/tests" "$SITE/simulators/tests"
cp "$HERE"/chia/trace/tests/test_*.py "$SITE/trace/tests/"
cp "$HERE"/chia/models/tests/test_*.py "$SITE/models/tests/"
cp "$HERE"/chia/simulators/tests/test_*.py "$SITE/simulators/tests/"
for PATCH in "$HERE"/patches/*.patch; do
    # Applied already? The first file the patch touches holds the first line the patch adds.
    # (Apple's patch has no --dry-run and GNU's no --check, so the file is asked, not patch.)
    TARGET=$(grep -m1 '^+++ b/' "$PATCH" | cut -c7-)
    MARKER=$(grep -m1 '^+[^+]' "$PATCH" | cut -c2-)
    if grep -qF -- "$MARKER" "$SITE/../$TARGET"; then
        echo "$(basename "$PATCH"): already applied"
    else
        (cd "$SITE/.." && patch -p1 -N -f -s < "$PATCH")
        echo "$(basename "$PATCH"): applied"
    fi
done
echo "blocks installed into $SITE"
