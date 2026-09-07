#!/usr/bin/env bash
# One-VM ChampSim worker on GCP (Debian 12 image). Run once on a fresh VM, from
# the synced repo directory (~/hackathon):   bash cluster/vm_bootstrap.sh
#
# Installs build tools, builds ChampSim (vcpkg), makes TREES copies of the
# ChampSim tree so that many designs compile at once (one build holds a tree),
# sets up the Python env, and downloads the traces listed in TRACES.
set -euo pipefail

TREES="${TREES:-12}"
TRACES=(
  "605.mcf_s-665B"
  "619.lbm_s-2676B"
  "620.omnetpp_s-874B"
  "623.xalancbmk_s-700B"
)
TRACE_URL="https://dpc3.compas.cs.stonybrook.edu/champsim-traces/speccpu"

sudo apt-get update -y
sudo apt-get install -y build-essential git curl zip unzip tar pkg-config cmake ninja-build \
                        python3 python3-pip python3-venv xz-utils ca-certificates

# ChampSim + vcpkg (same recipe as CLAUDE.md, headless)
# Pinned to the commit the Mac results were produced with; the SPP patch (fixes
# two spp_dev crashes, see upstream/README.md) must apply or spp_dev designs crash.
CHAMPSIM_COMMIT="51588e1"
if [ ! -d champsim ]; then
  git clone https://github.com/ChampSim/ChampSim.git champsim
  (cd champsim && git checkout "$CHAMPSIM_COMMIT" && git submodule update --init)
  (cd champsim && git apply ../upstream/0002-champsim-spp-dev-ghr-victim.patch)
  (cd champsim && ./vcpkg/bootstrap-vcpkg.sh -disableMetrics && ./vcpkg/vcpkg install)
  (cd champsim && ./config.sh champsim_config.json && make -j"$(nproc)")
fi

# Extra trees: a build holds a whole tree, so TREES designs compile in parallel.
for index in $(seq 1 $((TREES - 1))); do
  if [ ! -d "champsim_${index}" ]; then
    cp -r champsim "champsim_${index}"
  fi
done

# Python env for the loop
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
if [ ! -d .venv ]; then
  uv venv --python 3.10 .venv
  uv pip install -p .venv/bin/python numpy scipy scikit-learn matplotlib google-genai
fi

# Traces: download in parallel, skip what is already there
mkdir -p traces
cd traces
for trace in "${TRACES[@]}"; do
  file="${trace}.champsimtrace.xz"
  if [ ! -s "$file" ]; then
    curl -sSL -o "$file" "${TRACE_URL}/${file}" &
  fi
done
wait
cd ..
ls -la traces
echo "BOOTSTRAP DONE"
