# Blocks proposed for CHIA

Five small additions to `chia/`, each with Tier-0 tests and a docs page, verified inside the
released package (`chialoops` 1.0.1, Ray 2.54). Nothing in a loop has to change to use the first
one; the others are opt-in. Each was written for the council loop in this repository and then used,
unchanged, on CHIA's own CIRCT issue loop. None of them knows what a cache, a design or an issue is.

| block | module | what it does | proof in this repository |
|---|---|---|---|
| **Spend view** | `chia/trace/spend.py` + `patches/0001-viz-profile-format-spend.patch` (23 lines in `chia/cli`) | `chia viz-profile --format spend <log or dir>`: one CSV row per node and model with calls, tokens by kind (input, output, cache read, cache creation) and USD, then a total. A node's own `cost_usd` is trusted; otherwise a price table prices it; an unpriced model is counted in tokens and named, never guessed. `LLMSpend` is a loop-wide USD/token cap that raises before the call that would pass it. | Run on a genuine log from CHIA's Vertex node under a local Ray (two prompts, local and remote); on CHIA's CIRCT assess stage re-run through the same node (16 issues plain, 13 gated); the council's cost table is summed by the same function from the loop's own per-call rows. |
| **Evaluation ledger** | `chia/trace/ledger.py` (`Ledger`, `LedgerNode`, `report`, `rank_agreement`) | One JSON-lines row per candidate any iterative loop evaluates: who proposed it, why, what it measured, which constraints it broke, what it cost. From the rows: best-so-far, evaluations to a target, plateau, wasted share, feasible rate, equal-quality and convergence speedups, rank agreement between two evaluators. A `candidate` profiler event per row. | Every arm of the paper (random rebuilt from its seed, BO from its log, eleven council iterations from transcripts and reports) is scored by this one module; the round-by-round rebuild of the caps-off runs found the ban rule that stalled the council. |
| **Decider and call gate** | `chia/models/decider.py` (`Decider`, `JevDecider`, `ConstantDecider`), `chia/models/call_gate.py` (`CallGate`, `gate_and_log`, `log_outcome`) | Before an optional agent call, a cheap calibrated decider (TypeSafe's Jev, 0.042 USD per million input tokens) answers the caller's yes/no question from the caller's state summary. Shadow mode makes every call and records the verdicts beside the outcomes; apply mode skips the calls under the threshold. One `call_gate` profiler event per round and one `call_gate_outcome` per call. The state and the question are the caller's; the block holds no domain. | Council: over 140 shadow-mode specialist calls the gate at 0.4 would skip 46%, none of which gained; applied, a climb round's model cost fell 18%. CIRCT assess stage (CHIA's own prompt on CHIA's Vertex node, the paper's 16 issues): the question "is this a defect report?" skipped 3 of the 5 non-bugs and none of the 11 bugs, 16 to 13 agent turns, 21% fewer input tokens, the same issues sent on to reproduction as the paper's; ungated, the replica matched the paper's verdict on 15 of 16 issues (`results/audit/circt_assess/`). |
| **Context gate** | `chia/models/context_gate.py` (`PromptSections`, `ContextGate`) | For a prompt written in `## ` sections, the decider judges per section whether the task needs it, from the caller's one-line description of the section, and a difficulty that maps to the caller's thinking tiers; shadow records, apply drops. Instruction sections are kept by name prefix. | Shown its sections' first characters (numbers) the decider rated the evidence unneeded: a negative result, kept. Shown descriptions it keeps the evidence and drops boilerplate: applied, 16 to 17% fewer briefing tokens per call and a climb round 31% cheaper together with the call gate. Quality equivalence needs more seeds than two. |
| **Gemini JSON node** | `chia/models/vertex_json.py` (`VertexGeminiJSON`) | A model node for one JSON answer from Gemini on Vertex at a chosen temperature, output limit and thinking budget, no tools. Its profiler event carries `input_tokens`, `output_tokens` (answer plus thinking, as billed), `thinking_tokens`, `model`, `cost_usd` at the spend block's list price, and whatever the caller adds (its label, round, run tag). Rate limits and server errors back off; a malformed answer is the caller's retry, each a billed call. CHIA's Vertex node fixes the generation config and leaves the thinking tokens out of the bill (finding 3 below). | Every model call of the council goes through it (`council_loop/analyst.py`); the Tier-0 test drives it with a stub client (the call it makes, the record it writes, the retry on a 429). |
| **ChampSim configuration build** | `chia/simulators/champsim_config.py` (`build_champsim_config`, `check_config`, `config_name`) + `patches/0002-run-champsim-raw-stats.patch` (7 lines in `chia/simulators/champsim.py`) | A `@ChiaFunction` that builds ChampSim from a complete configuration (every level's geometry, prefetcher, replacement policy, queue depths, core and memory) and returns CHIA's own `ChampSimBuildResult`, so `ChampSimNode.run_champsim` runs it unchanged. `ChampSimNode.build_champsim` builds one prefetcher at one level; any design space over a hierarchy needs this. The patch adds `raw_stats` to `ChampSimRunResult`: the `--json` record as ChampSim wrote it, beside the typed subset `run_champsim` parses (finding below). | Proven on the VM: the table's best design rendered to a configuration, built through the block in 11 s, run through CHIA's `run_champsim` on the llama2 trace at 1M/2M: IPC 1.5465, identical to the table's value from the loop's own build (`check_champsim_config.py`). The same run on a Mac gives 1.5129: ChampSim's results differ between clang/libc++ and gcc/libstdc++ builds, which is why every table in this repository comes from one machine. |

## Why CHIA

CHIA's paper (section 2) asks for "data collection and profiling of both results and of the loop
itself built in as first-class citizens". The profiler already records every model call's tokens
and model (eight of eleven model nodes write them); nothing summed, capped or attributed them, and
four of the five case studies in `examples/` keep a private iterations database (1090 lines). The
spend view and the ledger are those two things once. The gates are the third: a loop can now say
which of its calls were worth making, at a thousandth of the price, before it stops making them.

## Layout, tests, docs

CHIA's own layout: modules under `chia/<area>/`, tests in `chia/<area>/tests/` that run without a
cluster (`pytest -q chia`: 30 passed, 1 skipped without Ray; `bash check.sh` runs them inside the released package, where the Tier-1 Ray test runs too: 31 passed), docstrings on every public function,
`docs/user_guides/ledger_and_spend.rst` and `docs/api/trace_blocks.rst`. The CLI change is a patch
against `ucb-bar/chia` 16c35e9 (release 1.0.1). AI assistance was used in writing the code and is
disclosed here, as `CONTRIBUTING.md` asks.

## Runtime check (what we ran)

    python3.10 -m venv v && v/bin/pip install chialoops pytest
    SITE=$(v/bin/python -c "import chia; print(list(chia.__path__)[0])")
    cp chia/trace/spend.py chia/trace/ledger.py $SITE/trace/
    cp chia/models/decider.py chia/models/call_gate.py chia/models/context_gate.py $SITE/models/
    cp chia/simulators/champsim_config.py $SITE/simulators/
    (cd $SITE/.. && patch -p1 < patches/0001-viz-profile-format-spend.patch)
    v/bin/chia viz-profile --format spend <a ChiaProfileCollector.log or its directory>

With that in place (`check.sh` does all of it from a clean virtualenv): the 31 tests pass inside the package, `LedgerNode.record`/`report` and
`build_champsim_config` carry `chia_remote`, CHIA's Vertex node under `start_collector` produces a
log the spend view sums (`results/audit/circt_assess/*/ChiaProfileCollector.log` are two such logs),
and the gates' events sit beside CHIA's own `dispatch`/`complete` events in it.

## Three findings about CHIA's Vertex node, for an issue upstream

Running CHIA's CIRCT assess prompt through `VertexGeminiLLM` with `gemini-3.1-pro-preview`: (1) when the
prompt mentions a tool that is not declared, Gemini attempts the call, Vertex finishes with
`MALFORMED_FUNCTION_CALL`, and the node returns `success=True` with an empty result (14 of 16 issues);
the finish reason should be treated as a failure. (2) A 429 raises at once ("never retry"), so a batch of
sixteen calls dies on the first rate limit; an opt-in backoff would match the other backends' policy.
(3) The node's usage record adds `candidates_token_count` as the output and never reads
`thoughts_token_count`, so a thinking model's spend is undercounted (thinking was 37% of our loop's
bill and is billed as output); and its generation config is fixed (no temperature, JSON mode or
thinking budget), so a loop with a set recipe cannot use it. The Gemini JSON node above is the
smallest node that does both; the fix upstream is two lines in `_run_generate_async` and a config argument.

## Two findings about CHIA's ChampSim node, for an issue upstream

(1) `run_champsim` parses a subset of ChampSim's `--json` record: the first DRAM channel only
(`dram_list[0]`; a two-channel memory reports half its row-buffer traffic), no `miss_merge` counts
(the comment above the loop names them, the loop reads hits and misses), core 0 only. A loop that
models power from the counters or shows the memory system to an agent needs the whole record; the
patch above returns it beside the typed stats, and `council_loop/simulate.py` reads its counters from it.
(2) `build_champsim` and `build_champsim_config` (ours, before this fix) run `make -j$(nproc)`;
`nproc` is GNU, so on a Mac `make -j` runs unbounded. `$(nproc 2>/dev/null || sysctl -n hw.ncpu)`
serves both.
