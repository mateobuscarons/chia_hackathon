#!/bin/bash
# The runtime check of the blocks against the released CHIA package, from a clean virtualenv:
# install chialoops, copy the blocks into it, apply the CLI patch, run the Tier-0 tests inside the
# package, and print the spend view over one of the profiler logs shipped in results/. No key, no
# cluster, no simulator; a few minutes, most of it pip. Usage: bash check.sh [venv directory]
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
VENV=${1:-/tmp/chia_blocks_check}
python3.10 -m venv "$VENV"
"$VENV/bin/pip" install -q chialoops pytest
SITE=$(cd /tmp && "$VENV/bin/python" -c "import chia; print(list(chia.__path__)[0])")
bash "$HERE/install.sh" "$VENV"
echo "== tests inside the installed package =="
OURS=""
for f in "$SITE"/trace/tests/test_ledger.py "$SITE"/trace/tests/test_ledger_node.py "$SITE"/trace/tests/test_spend.py \
         "$SITE"/models/tests/test_call_gate.py "$SITE"/models/tests/test_context_gate.py "$SITE"/models/tests/test_vertex_json.py "$SITE"/simulators/tests/test_champsim_config.py; do OURS="$OURS $f"; done
(cd /tmp && "$VENV/bin/python" -m pytest -q $OURS 2>&1 | tail -1)
echo "== chia viz-profile --format spend over the shipped CIRCT log (ungated pass) =="
(cd /tmp && "$VENV/bin/chia" viz-profile --format spend "$HERE/../../results/audit/circt_assess/plain")
echo "== and over the gated pass =="
(cd /tmp && "$VENV/bin/chia" viz-profile --format spend "$HERE/../../results/audit/circt_assess/gated")
