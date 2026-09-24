"""Build ChampSim from a full configuration, not from one prefetcher.

`ChampSimNode.build_champsim` writes one prefetcher module at one level and builds. A design
space over a hierarchy moves geometry, prefetchers, replacement policies and queue depths at
every level at once; ChampSim takes all of that as one configuration JSON. This module builds
from such a configuration: it writes the JSON, runs `config.sh` and `make` (no `make clean`,
which is what makes a build of the next design take seconds rather than minutes, because only
the generated environment changes), names the binary after the configuration, and returns the
same `ChampSimBuildResult` the existing node returns, so `run_champsim` runs it unchanged.

The pure helpers (`config_name`, `check_config`) need neither Ray nor ChampSim and are what
the Tier-0 tests cover. The node function is defined when CHIA is importable.
"""

import hashlib
import json
import os
import re
import shlex

CACHE_LEVELS = ["L1I", "L1D", "L2C", "LLC", "ITLB", "DTLB", "STLB"]

try:
    from chia.base.ChiaFunction import ChiaFunction
    from chia.simulators.champsim import ChampSimBuildResult, _filter_build_diagnostics, _git, _run_logged
except ImportError:
    ChiaFunction = None


def config_name(config):
    """A name for the binary that is a function of the configuration alone: the same
    configuration always builds to the same name, so a build cache can be keyed by it. The
    executable_name field itself is left out of the hash."""
    stripped = dict(config)
    stripped.pop("executable_name", None)
    digest = hashlib.sha256(json.dumps(stripped, sort_keys=True).encode()).hexdigest()[:12]
    return "champsim_" + digest


def check_config(config):
    """The mistakes a generated configuration makes most: a cache with sets that are not a
    power of two, non-positive ways or queues, an unknown cache section. Returns the list of
    problems, empty when the configuration is buildable as far as the JSON can tell."""
    problems = []
    for section in config:
        if section in CACHE_LEVELS and isinstance(config[section], dict):
            cache = config[section]
            sets = cache.get("sets")
            if sets is not None and (not isinstance(sets, int) or sets <= 0 or sets & (sets - 1)):
                problems.append("{}: sets must be a positive power of two, not {!r}".format(section, sets))
            for field in ["ways", "mshr_size", "rq_size", "wq_size"]:
                value = cache.get(field)
                if value is not None and (not isinstance(value, int) or value <= 0):
                    problems.append("{}: {} must be a positive integer, not {!r}".format(section, field, value))
            # A prefetch queue of size 0 is legal and is ChampSim's own default for the TLBs.
            value = cache.get("pq_size")
            if value is not None and (not isinstance(value, int) or value < 0):
                problems.append("{}: pq_size must be a non-negative integer, not {!r}".format(section, value))
            name = cache.get("prefetcher")
            if name is not None and not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", str(name)):
                problems.append("{}: prefetcher {!r} is not a module name".format(section, name))
            name = cache.get("replacement")
            if name is not None and not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", str(name)):
                problems.append("{}: replacement {!r} is not a module name".format(section, name))
    return problems


if ChiaFunction is not None:

    @ChiaFunction(resources={"champsim": 1.0})
    def build_champsim_config(champsim_root, config, *, timeout_s=600, clean=False):
        """Build ChampSim from a full configuration dict and return the binary as bytes.

        Args:
            champsim_root: the ChampSim checkout on the worker.
            config: the configuration, in ChampSim's own JSON schema (a partial one overrides
                the checkout's defaults, as `config.sh` merges it). `executable_name` is set
                from the configuration's hash unless given.
            timeout_s: wall-clock limit for `config.sh` and `make` together.
            clean: run `make clean` first. Off by default: `config.sh` regenerates the
                environment and `make` rebuilds only what changed, which is what a design
                space with thousands of builds needs.

        Returns:
            A `ChampSimBuildResult`; `module_name` carries the executable name.
        """
        problems = check_config(config)
        if problems:
            raise ValueError("configuration not buildable: " + "; ".join(problems))
        config = dict(config)
        executable = config.get("executable_name") or config_name(config)
        config["executable_name"] = executable
        base_rev = ""
        rev = _git(["rev-parse", "HEAD"], champsim_root, timeout=30)
        if rev.returncode == 0:
            base_rev = rev.stdout.strip()
        config_path = os.path.join(champsim_root, executable + ".json")
        with open(config_path, "w") as config_file:
            json.dump(config, config_file, indent=2)
        try:
            steps = []
            if clean:
                steps.append("make clean")
            steps.append("python3 ./config.sh {}".format(shlex.quote(config_path)))
            # The generated environment is what a configuration changes; make does not always
            # see that, so its object is removed before make runs.
            steps.append("rm -f .csconfig/generated_environment.o")
            steps.append("make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)")   # nproc is GNU; a Mac has sysctl
            returncode, stdout, stderr, timed_out, wall = _run_logged(" && ".join(steps), champsim_root, timeout_s)
        finally:
            try:
                os.unlink(config_path)
            except OSError:
                pass
        success = returncode == 0 and not timed_out
        binary = b""
        if success:
            binary_path = os.path.join(champsim_root, "bin", executable)
            try:
                with open(binary_path, "rb") as binary_file:
                    binary = binary_file.read()
            except OSError as error:
                success = False
                stderr += "\nFailed to read binary: {}".format(error)
        diagnostics = ""
        if not success:
            if timed_out:
                diagnostics = "TIMEOUT after {:.0f}s (limit {}s)".format(wall, timeout_s)
            else:
                diagnostics = _filter_build_diagnostics(stdout, stderr)
        return ChampSimBuildResult(binary=binary, module_name=executable, champsim_root=champsim_root,
                                   base_rev=base_rev, success=success, returncode=returncode,
                                   build_duration_s=wall, stdout_tail=stdout[-3000:], build_diagnostics=diagnostics)
