"""Puts this folder first on the path. CHIA's `chia` is a namespace package, so with this folder
first the blocks (`chia.trace.ledger`, `chia.models.vertex_json`, ...) resolve here and the rest
of `chia.*` to the installed package; the same tests run inside the package after `install.sh`."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

