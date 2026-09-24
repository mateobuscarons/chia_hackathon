# CHIA hackathon: hand-over and the paper's source (written Sep 24 ~01:00 UTC)

Everything the paper needs is in this file with the file that holds each number. Dates stay out of
shipped code, README and paper. The user's working rules: `CLAUDE.local.md`, the session memory and
section 12 here. Deadline: submissions close **Sep 24 AoE = Sep 25 11:59 UTC** on
`a3-chia-hackathon-26.hotcrp.com` (4-page ACM/IEEE 2-column PDF, AI writing help acknowledged at
the end; the loop open-sourced with its results). Judges: human PC on "author-identified
highlights, paper, artifact"; AI reviewers inform. Tracks: reusable CHIA blocks (best fit), agent-
driven power optimisation, cross-SoC cache hierarchies.

## 1. Spine, storyline, highlights

**The instrumented loop and its blocks are the contribution; the council is the case study.**
Everything told straight, negatives kept. No BO in the paper (user decision; earlier versions' BO data
stays in `results/vm2/` for our own judgement and is not cited).

Storyline (five lines): agentic loops are judged by their outcome and never measured themselves;
CHIA's paper (section 2) asks for "data collection and profiling of both results and of the loop
itself built in as first-class citizens". We add the blocks that measure a loop from the log CHIA
already writes (spend view and cap, ledger, decider-driven gates) and prove them inside the released
CHIA. A capped cache-hierarchy council is the case study: it reaches the quality random search has
after 1000 designs in 39 to 44 designs (mean over seeds, per regime), and the blocks explained
its failures and cut its model spend 22 to 30% at equal final quality. The same blocks transfer: CHIA's own CIRCT issue loop, gated, made 19% fewer agent
turns with the same issues going forward. Everything reproduces from the repo with one command and no
key.

Author-identified highlights (draft): (1) `chia viz-profile --format spend`: cost per node, model and
token kind from the log CHIA already writes, proven on CHIA's own node and case study; (2) the call
gate: shadow, threshold, apply; 30% less spend on our loop at equal quality, 19% fewer turns on CHIA's
CIRCT stage; (3) the ledger: one record format scoring any iterative loop the way the DSE literature
compares methods; (4) the lean council: 23 to 25x fewer simulations than random under hard caps, and what
the instruments found about it; (5) a Gemini JSON node and a ChampSim configuration node; five node
findings filed upstream as fixes (the blocks stay in the repository).

## 2. What ships: the blocks (`council_loop/chia_blocks/`, CHIA's layout)

`pytest -q chia`: 33 passed; `bash check.sh`: clean venv + `chialoops` 1.0.1 + `install.sh` (copies the
blocks and tests into the released package, applies the two patches, portable across Apple and GNU
patch) + the same 33 tests inside the package + the spend view over the shipped CIRCT logs. CHIA's
`chia` is a namespace package, which is how the loose blocks always imported; Ray workers see only the
installed package, so `install.sh .venv` is the documented path. `UPSTREAM.md` holds what/why/proof per
block and the node findings; `docs/user_guides/ledger_and_spend.rst`, `docs/api/trace_blocks.rst`.
License BSD-3 (`LICENSE`).

| block | module | what | proof |
|---|---|---|---|
| spend view + cap | `chia/trace/spend.py` + `patches/0001-viz-profile-format-spend.patch` (23 lines in `chia/cli`) | sums the token fields CHIA's model nodes already write (input, output, cache read, cache creation, model, cost) per node, model and node-and-model; a node's own `cost_usd` trusted, else a price table, unpriced models named; `--format spend` CSV beside `--format table`; `LLMSpend` cap raises before the call that would pass it | genuine logs: CHIA's Vertex node under a local Ray (`scratchpad/vertex_demo_log`), five CIRCT passes (`results/audit/circt_assess/*/ChiaProfileCollector.log`); the council's cost table is summed by the same function from `results/llm_calls.jsonl` |
| ledger | `chia/trace/ledger.py`: `Ledger`, `LedgerNode`, `report`, `markdown`, `rank_agreement` | one JSONL row per candidate (candidate dict, metrics dict, objective name, source, rationale, violations, cost); best-so-far, evaluations-to, plateau, wasted share, feasible rate, equal-quality and convergence speedups, rank agreement; a `candidate` profiler event per row | every arm of the paper scored from one format (random rebuilt from its seed and the table, the council from reports); the caps-off collapse of an earlier version traced to one rule by the round-by-round rebuild |
| decider + call gate | `chia/models/decider.py` (`Decider`, `JevDecider`, `ConstantDecider`), `chia/models/call_gate.py` (`CallGate`, `gate_and_log`, `log_outcome`) | the caller supplies a state summary and one yes/no question per optional call; shadow makes every call and records verdicts; apply skips under the threshold; `call_gate` and `call_gate_outcome` events | council: 304 shadow calls of the frozen recipe, 40% skippable at 0.4 with 2 marginal gains lost; applied: -30% spend caps off, -22% caps on, equal quality (section 4.3). CIRCT: 16 -> 13 turns, same issues forward |
| context gate | `chia/models/context_gate.py` (`PromptSections`, `ContextGate`) | per `## ` section the decider judges the caller's one-line description; the caller's thinking tiers and always-kept prefixes; shadow records, apply drops | shown section heads (numbers) the decider rated the evidence unneeded (negative, kept); shown descriptions, applied: 16-17% fewer briefing tokens per call |
| Gemini JSON node | `chia/models/vertex_json.py` (`VertexGeminiJSON`, `prompt` a `@ChiaFunction`) | one JSON call at a set temperature, output limit and thinking budget; the event carries input, output (answer + thinking, as billed), thinking, model, USD from the spend block's prices, and the caller's label/round/tag; retries 429/5xx; CHIA's Vertex node fixes its config and omits thinking tokens | every model call of the loop; Tier-0 test with a stub client |
| ChampSim configuration build | `chia/simulators/champsim_config.py` (`build_champsim_config`, `check_config`, `config_name`) + `patches/0002-run-champsim-raw-stats.patch` (7 lines: `raw_stats` on `ChampSimRunResult`, since CHIA's parse keeps the first DRAM channel only and no merges) | `@ChiaFunction` building ChampSim from a complete configuration (every level's geometry, prefetcher, replacement, queues, core, memory), returning CHIA's `ChampSimBuildResult` so `run_champsim` is unchanged (CHIA's node builds one prefetcher at one level) | the table's best design built in 11 s and run through CHIA's `run_champsim` on the VM: IPC 1.5465, identical to the table (`check_champsim_config.py`, `_spec.json`); on a Mac 1.5129: results differ by compiler/stdlib, so every table comes from one machine |

Upstream work order (`upstream/README.md`): the blocks PR; Vertex node fix 1 (a `MALFORMED_FUNCTION_CALL`
finish returns `success=True` with an empty result when the prompt mentions an undeclared tool; 14 of
16 CIRCT issues blank); Vertex node fix 2 (a 429 raises at once, no retry); a profiler docs note
(events are fire-and-forget: read after a blocking `get_events`; `reset_profiler()` when a driver
restarts the collector); the `gs://` trace patch; the ChampSim `spp_dev` fix (to ChampSim).

## 3. Methods

**Chip and caps** (`council_loop/socs.py`, `next`): ChampSim's stock core widened to 6-issue, ROB 384, 8-wide
fetch, LQ 160 / SQ 96, 22 nm at 3.8 GHz, DDR5-4800 (2 channels); caps 4.0 mm2 of cache silicon (CACTI 7
at 22 nm), 1.0 W of cache + off-chip power (CACTI energies x ChampSim counters, DRAM 15 pJ/bit), no
workload slower than the stock. The stock's LLC (2048x16) draws 1.095 W on llama2, so the reference
is the stock with the LLC halved (1024 sets): 1.96 mm2, 0.805 W, IPC 0.935. Caps-off SoC `nocap`: same
chip, no caps, shares the tables; its stock is the unshrunk one, IPC 0.9427. Silicon, leakage and an
ordered hierarchy are checked before simulation (never counted); power and the floor after (counted,
recorded with the reason, never stood on).

**Trace and fidelity**: llama2.c-llama2_7b.1 (DPC4 ai-ml, 200 MB prefix); development and every
comparison at 1M warm-up / 2M simulated; CHIA's default 5M/25M for the fidelity check. The other five
gate traces: section 4.9.

**Design space** (`council_loop/space.py`, `SEARCH_SPACE`): 13 knobs: L1D sets/ways/prefetcher; L2 sets/ways/
prefetcher/replacement/MSHR (16, 32, 64); LLC sets/ways/prefetcher/replacement/MSHR (32, 64, 128). Sets
and ways on per-level ladders (ways 4, 8, 12, 16); prefetchers no, next_line, ip_stride, va_ampm_lite
(L1D, L2), spp_dev (L2, LLC); replacement lru, srrip, drrip, ship. Stock: L1D 64x8 next_line, L2 1024x8
spp_dev srrip MSHR 32, LLC 2048x16 (1024 under caps) no prefetcher srrip MSHR 128. Table best 1.5465.

**Protocol**: the budget unit is a design (one simulation); every arm opens with a wave and then measures
5 designs a round in parallel. Random: 1000 designs (10 then 5 a round), uniform over the feasible set,
seeded, rebuilt from its seed and the table by `council_loop/ledgers.py random:<cell>:<seed>:<budget>`. Council:
60 designs or 20 without a new incumbent or two rounds that measured nothing. Seeds change random's
draws, the council's random openings and its temperature-0.7 answers. Targets: the mean over random's
three seeds of its best after 1000 designs (caps on 1.4329, caps off 1.5001; the strongest seed's value
is also reported in 4.1). Speedup = 1000 / the council's mean designs to that value; a seed that spends
its 60-design budget under the value counts at the budget, 60 designs (a tie at the budget), with its
final IPC beside it. The paper shows the speedup for the council as run (the shadow arm) and the
applied gates as final IPC and spend.

**The frozen council** (`council_loop/council.py`, ~1990 lines, pyflakes clean): round 1 = the analyst's sheet
+ its opening design + 19 random feasible designs (20 openings). Every later round, one wave of 5:
the **statistician** (a GP with Matern-5/2 over every design this run measured, refused designs at
stock - 0.05, scoring a 20,000-design seeded pool plus the incumbent's one-knob neighbours by expected
improvement; its five best predictions in every prompt) proposes its top pick; the **analyst** writes
the sheet (verdict per level, bottleneck with evidence, findings with numbers, unattributable designs,
failing knobs filtered mechanically once measured alone, cap headroom) and one whole design (the
statistician's runner-up takes the slot when it writes none); four **specialists** (prefetch, geometry,
replacement, concurrency) propose one move each on their own knobs or hold, taken as proposed; a gain
the power cap refused last round comes back **repaired to fit** (the level holding the most power
shrunk rung by rung until the design's own counters estimate it under 0.95 x cap, level re-chosen each
step); **untried single-knob moves** at the level the sheet blames top the wave up. Incumbent = best
feasible design measured, whoever proposed it. A move already measured against the incumbent is a
hold. Removed from earlier versions (never paid, or harmed): the value-ban rule, the jump round with
its counterfactual and restructure prompts, the analyst's re-ask, the specialists' one-level re-ask,
the level rotation, the one-rung repair, the analyst-turn path. Switches for the ablation only:
`STATISTICIAN`, `FILL`, `REPAIR` (fit|none); `OPENINGS=20`.

**Model and gates**: Gemini 3.1 Pro preview on Vertex (`global`), JSON mode, temperature 0.7, thinking
budget 4096 (specialists 1024 when the applied context gate rates the task below hard); list prices
2.00 / 12.00 USD per M tokens, thinking billed as output. Decider: TypeSafe Jev (`jev-latest`, answered
`jev-1.13.0`), 0.042 USD per M input tokens. Call gate threshold 0.4 ("does the sheet give evidence for
changing this concern's knobs this round?"); context gate 0.5 for specialists, 0.3 for the analyst,
sections judged by a one-line description, instruction sections always kept (`Role`, `Task`, `Once
more`, `The search is stalled`). Shadow gating = every call made, verdicts logged; apply gating = calls
under the threshold not made, briefings trimmed, specialist thinking capped.

## 4. Results (every number with its file)

### 4.1 Council vs random, caps off (lean, 20 openings, 4 seeds per arm)

Target: the mean of random's best after 1000 designs over three seeds, 1.5001 (finals 1.4982 / 1.4910 /
1.5111; `results/ledgers/lean/nocap_llama2_nocap_random1000_random_s*`). Random's mean best-so-far:
1.167 (20), 1.206 (40), 1.331 (60), 1.359 (100), 1.395 (200), 1.437 (500), 1.500 (1000).

| arm | finals | designs to 1.5001 | mean (60 = budget) | USD per run | mean | model calls | tokens per run (all kinds) |
|---|---|---|---|---|---|---|---|
| shadow gating | 1.522 / 1.522 / 1.511 / 1.529 | 37 / 41 / 47 / 32 | 39 (25x) | 1.91 / 1.55 / 1.60 / 1.79 | 1.71 | 49.8 | 563,551 |
| apply gating | 1.511 / 1.526 / 1.517 / 1.497 | 53 / 53 / 42 / 60 (budget) | 52 (19x) | 1.13 / 1.20 / 1.22 / 1.24 | 1.20 (-30%) | 31.0 (-38%) | 352,164 (-38%) |

Council best-so-far, mean over seeds: shadow 1.251 (20), 1.482 (40), 1.521 (60); apply 1.223 (20),
1.391 (40), 1.510 (60). Council finals 1.497 to 1.529 against random's mean 1.500. Against the strongest
random seed (1.5111): shadow 43 / 48 / 60 / 32, apply 60 / 58 / 62 / 60 (60 = budget;
`results/report/lean20/caps_off_vs_random1000.md`). Ledgers `results/ledgers/lean/nocap_llama2_lean20_nocap*`; reports
`results/vm2/results/runs/nocap_llama2_lean20_nocap*.json`; table and money figures `results/report/lean20/`.

### 4.2 Council vs random, caps on (lean, 20 openings, 4 seeds per arm)

Target: the mean of random's best after 1000 designs over three seeds, 1.4329 (finals 1.4144 / 1.4448 /
1.4396; `results/ledgers/lean/llama2.c-llama2_7b.1_llama2_random_seed*`). Random's mean best-so-far:
1.165 (20), 1.267 (40), 1.278 (60), 1.346 (200), 1.433 (1000).

| arm | finals | designs to 1.4329 | mean (60 = budget) | USD per run | mean |
|---|---|---|---|---|---|
| shadow gating | 1.524 / 1.541 / 1.441 / 1.542 | 52 / 37 / 60 / 28 | 44 (23x) | 1.70 / 1.54 / 1.55 / 1.74 | 1.63 |
| apply gating | 1.524 / 1.544 / 1.544 / 1.536 | 28 / 61 / 38 / 29 | 39 (26x) | 1.30 / 1.29 / 1.21 / 1.29 | 1.27 (-22%) |

Council best-so-far, mean over seeds: shadow 1.164 (20), 1.396 (40), 1.511 (60); apply 1.189 (20),
1.463 (40), 1.496 (60). Tokens per run (all kinds): shadow 551,192, apply 385,026 (-30%); model calls
47.2 v 34.0 (-28%). Designs to 1.50 (council, 8 seeds): 33, 36, 38, 38, 42, 59, 61, 60 (budget); to 1.54: 45, 52, 61, 62 (4 of 8
within 60). With 10 openings one seed in two stalled at 1.17 (2 of 4 seeds of the same recipe: 1.168
and 1.543/1.508/1.522); with 20 openings none of 16. Ledgers `results/ledgers/lean/next_llama2_lean20_cap*`.

### 4.3 Gating: what it saves, at what quality

- **Shadow counterfactual, frozen recipe** (304 specialist calls, 8 seeds, `results/vm2/results/
  skip_gate.jsonl` tags `lean20_cap-s*`, `lean20_nocap-s*`): 57% of specialist calls ended in a hold.
  At threshold 0.4 the gate would skip 40% (121): 92 holds, 29 proposals, of which 2 gained more than
  0.005 (+0.006, +0.005). Of the 20 gains above 0.005 among all 304 calls, 18 carried verdicts of 0.4 or
  more. Per concern (76 calls each): concurrency mean verdict 0.30, held 82%, 82% skippable; replacement
  0.38, held 61%, 63% skippable; prefetch 0.67, held 56%, 10%; geometry 0.73, held 25%, 2%. The gate
  reads from the sheet which concerns have evidence this round; it is not told the history.
  (Earlier versions, 140 calls: 46% skippable, 0 gains lost.)
- **Apply vs shadow, caps off** (4 v 4, section 4.1): -38% tokens, -38% calls, -30% spend; finals
  1.50-1.53 v 1.51-1.53 (means 1.513 v 1.521).
- **Apply vs shadow, caps on** (4 v 4, section 4.2): -30% tokens, -28% calls, -22% spend (1.27 v 1.63
  USD); finals 1.52-1.54 v 1.44-1.54 (means 1.537 v 1.512). Specialists consulted 2.4-2.8 times a
  round instead of 4 (caps off 2.1-2.4); 11-14 calls skipped per run (caps off 14-18); 16-17% of
  briefing tokens dropped; a climb round 0.13-0.14 USD instead of 0.17-0.18 (`results/report/lean20/
  gating.md` caps off, `results/report/lean20_cap/gating.md` caps on).
- **The sentence**: with the CHIA gates applied, the lean council reached the same final quality (mean
  IPC 1.537 v 1.512 under caps, 1.513 v 1.521 without) for 22 to 30% less model spend; the decider cost
  about 0.1 cents a round.
- **CHIA's CIRCT assess stage** (`results/audit/circt_assess/README.md`; their `assess.md` prompt, the
  16 issues of their Table 5, CHIA's Vertex node, the call gate asking "is this a defect report?" at
  0.65): ungated x3: 15/16, 15/16, 13/16 verdicts equal to the paper's, the paper's 7 issues proceed;
  gated x2: 13 turns, 14/16 and 15/16, the same 3 non-bugs skipped (#2669, #4396, #7127), 0 of 11 bugs
  skipped, the paper's 7 proceed (one pass 6: #6740 moved on an ungated call, and moved again in an
  ungated pass). Tokens per pass 37,732 -> 30,409 (-19%), USD 0.095 -> 0.081 (-15%, answer lengths
  vary), turns 16 -> 13 (-19%); at the paper's 0.40 USD per turn, 1.20 of the stage's 6.40 USD (19%).
  Decider: 0.001 USD for 16 decisions. Deviations forced: tool paragraph removed, one-line system
  prompt, 429 retry (node bugs, section 2). Tool: `council_loop/circt_assess.py`.

### 4.4 Who produced the new best designs (caps on vs caps off, 8 seeds each, shadow and apply)

A new best = a feasible design above every earlier feasible design of its run, after round 1
(`source` field of the ledgers; repairs are `council:repair:<member>`). Rule and counts reproducible
from `results/ledgers/lean/*lean20*`.

| regime | new bests | LLM specialists | LLM analyst | repairs of a refused gain | surrogate (GP) | single-knob fill | designs refused by the power cap |
|---|---|---|---|---|---|---|---|
| caps on | 53 | 13 (25%) | 12 (23%) | 8 (15%) | 13 (25%) | 7 (13%) | 179 of 493 (104 random openings, 27 surrogate, 19 specialist, 15 analyst, 12 fill, 2 repair) |
| caps off | 58 | 15 (26%) | 8 (14%) | 0 | 21 (36%) | 14 (24%) | 0 |

Caps on: LLM proposals 47% + repairs of refused gains 15% = 62%, surrogate 25%: the cap turns the search
into reading where the power goes, and the model's whole designs and repairs find the escapes. Caps
off: the surrogate's share rises to 36% and fills to 24%, LLM 40%: a smoother numeric landscape where
the GP's model of knob interactions leads. The caps decided the path, not the outcome (finals 1.44 to
1.54 v 1.50 to 1.53).

### 4.5 Ablation: the LLM-only council (`bare`: no statistician, fill, repair; 10 openings, 2 seeds per regime)

Caps on 1.1631 at 28 and 1.1707 at 25; caps off 1.1518 at 26 and 1.1640 at 26: the specialists run
out of proposals and the run ends. The model diagnoses and writes coupled designs; the surrogate, the
fill and the repair find the escapes. `results/ledgers/lean/*bare*`, `results/report/lean/`.

### 4.6 Fidelity

40 llama2 designs (the best feasible plus a spread over the IPC range) re-measured at 5M/25M: Spearman
0.986, Kendall 0.938, top-5 overlap 80%. `results/vm2/results/fidelity_llama2.json`, `.log`, table
`results/vm2/results/tables/llama2.c-llama2_7b.1_w5M_s25M.json`; `council_loop/fidelity.py`.

### 4.7 Cost anatomy

Frozen recipe, 648 calls over 16 runs (`results/vm2/results/llm_calls.jsonl`): input 49%, thinking 37%,
answers 14% of the bill; the analyst 44% of it (0.062 USD a call, 10.9k tokens in, 2.0k thinking) and a
specialist 0.027 (8.1k in). CHIA's own published studies (their Tables 1, 3, 4, 6, token classes at
Anthropic's price ratios 1 : 1.25 : 0.1 : 5): cache reads 49-59% of spend, output 13-28%, cache creation
14-35%, fresh input ~0%; critical path: half the iterations after the plateau ($101 of $202); CIRCT
stage costs assess 0.40, reproduce 0.68, fix 3.01, writeup 0.09 USD per issue. Different loops, different
levers; the profile tells you which. Provider prompt caching: Vertex returned no cached tokens on a
repeated 5.6k prefix (3.1 Pro): dead end.

### 4.8 ChampSim configuration node proof

Section 2: IPC 1.5465 = table on the VM; 1.5129 on the Mac (same commit 51588e1d, same `spp_dev` patch,
same 200 MB trace). Every table in the repo comes from the VM.

### 4.9 The brief gate (six traces, Mac, 1M/2M, `next` caps, no model call; `council_loop/search.py gate`)

| trace | headroom | knobs differ | best single move | lose alone | refused after measurement |
|---|---|---|---|---|---|
| llama2 (table) | +64% | 11 | 8% | 3 | |
| whisper_1 | +24.5% | 10 | 80% | 4 | 25 of 66 |
| clip_1 | +1.0% | 11 | 58% | 7 | 52 of 67 |
| stable-diffusion v1-5 | +0.5% | 1 | 100% | 0 | 43 of 67 |
| vit-large.1 | +0.4% | 10 | 100% | 4 | 50 of 67 |
| biogpt.1 | no buildable stock: 2.03 W on the shipping design, 1.60 W with the LLC cut to a quarter | | | | |

Only llama2 passes every bar (headroom, depth, a single move buys little, single moves mislead). ~70
simulations a trace decide whether an agentic loop is worth running at all.

### 4.10 Grounding audit (JEV reads each proposal's evidence and reasoning: "does the sheet support this move?")

Earlier versions (56 proposals, `results/audit/grounding_iterations7_8.jsonl`): verdict >= 0.7 -> 0
gains, 56% refused by the cap, mean -0.21; < 0.3 -> 14% gained: agreement with the analyst's diagnosis
predicted failure; the diagnosis step was the weak link. Frozen recipe (16 runs, 377 proposals,
`results/audit/grounding.jsonl`, `council_loop/audit.py`): flat. Gained: < 0.3 -> 29%, 0.3-0.5 -> 16%, 0.5-0.7 ->
31%, >= 0.7 -> 20%; mean gain negative in every band. Per member: prefetch 41% of proposals gained
(n=66), concurrency 47% (n=19), replacement 32% (n=38), analyst 18% (n=139), geometry 17% (n=115).
Reading: in the frozen recipe the analyst's diagnosis is no longer anti-predictive; whether a move
gains is not readable from its stated evidence. Grounding is a gauge of prompt discipline, not a
filter, and it is not in the paper's savings.

### 4.11 Negatives kept

The LLM-only council does not search (4.5). The context gate's first form (section heads shown to the
decider) rated the evidence unneeded. JEV cannot judge "one clear fix" (CIRCT: 10 of 11 bugs' fixes
rated unclear). The grounding sign means it is a gauge, not a filter. Prompt caching on Vertex: none.
With 10 openings one seed in two stalled under caps. Wall time at 1M/2M shows no advantage either way
(the model's minute a round offsets fewer rounds); the fair unit is designs.

## 5. Against the competition (`competiton.md`; figures as of the last survey)

CIRCT bug loop (IIIT-H): 15 new bugs through a human gate, 0 confirmed by maintainers; 241 ChiaFunction
uses, campaigns pre-registered by annotated git tag, two paper PDFs in the repo, a provenance comment per
number; reproduction needs a database outside the repo; no license. YARROW (NYU): 82 bugs, three Gemini
models, MIT, a credential-free one-minute demo (`quickstart.sh demo`). agcws: matched proposal counts,
bootstrap CIs, classical baselines, negatives kept; 7 GB repo, draft paper with TODOs. flux: 41 nodes,
no baseline, seeds, paper or license. repo-audit: evaluation validity thought through, stub numbers.
Our edge: generic blocks proven inside released CHIA and on CHIA's own case study, answering the need
CHIA's authors wrote down; every number from a file; negatives kept. Our gaps: tests (31 v hundreds),
no one-command offline demo yet, no pre-registration tag yet, the loop not yet fully CHIA-driven, paper
unwritten. With the artifact finished: a real shot at top 3; top 10 very likely.

## 6. The loop is CHIA-driven (done)

Model calls: tasks of `VertexGeminiJSON.prompt` (`council_loop/analyst.py`), each event with label, round and
tag; the loop's per-call rows (`results/llm_calls.jsonl`) are still written beside them. Builds:
`build_champsim_config` in a free tree, binary cached in `champsim_bin/` (`council_loop/simulate.py`). Runs:
`ChampSimNode.run_champsim` with the raw-stats patch, the loop's own parse on the raw record (identical
on 103 metrics of a re-measured design). Candidates: `Ledger` live at `results/ledgers/<tag>.jsonl`
(same rows as the rebuilt ledgers) with a `candidate` event each. Gates: `gate_and_log` and
`log_outcome` events. Runs: `run_cell` starts one Ray (`start_ray`: `champsim` and `vertex_creds`
resources), a spawn pool; `run_one` connects with `ray.init(address="auto", namespace=tag)`, its own
collector, a fresh log at `results/profiles/<tag>/ChiaProfileCollector.log`, `reset_profiler()` after
`start_collector`, flush + stop at the end. Proof log: `results/profiles/next-council-chiasmoke-s777`
(llama2 1M/2M, seed 777, 25 designs: 6 calls 0.2316 USD, 5 builds, 5 runs, 26 candidates, 1 `call_gate`,
4 `call_gate_outcome`, 6 `context_gate`); the spend view sums it. `preview` prompts identical before
and after the wiring. Environment: `.venv` holds `chialoops` 1.0.1 + Ray 2.54 (`requirements.txt`);
`bash council_loop/chia_blocks/install.sh .venv` after any block edit. Multi-core mixes cannot run
(CHIA's run node takes one trace).

## 7. Tooling and provenance

`council_loop/ledgers.py <out> <inputs>` (reports, `random:<cell>:<seed>:<budget>`, transcripts, logs; env
`TABLE_DIR=results/vm2/results/tables`; Mac and VM CACTI files identical); `council_loop/report.py <ledgers>
<vm_results> <out> arm=tag,...` (gating table, money figures; prices via the spend block); `council_loop/audit.py
<reports>`; `council_loop/circt_assess.py <out> plain|gated [threshold]` (in a venv with chialoops);
`council_loop/fidelity.py`; `council_loop/chia_blocks/check.sh`, `install.sh`, `check_champsim_config.py`;
`python3 council_loop/demo.py` (the offline readout: CIRCT gate, council gate counterfactual + savings,
ledger table, the loop's own logs; 0.6 s, plain python3, no install); `paper/` (IEEEtran, `bash
paper/build.sh` with tectonic in `~/.local/bin`, `open paper/main.pdf`; `fig_curves.py` draws Fig. 2 from
the ledgers and call rows; a provenance comment per number). Every VM measurement is under
`results/vm2/results/`; `results/ledgers/lean/` holds the scored ledgers.

## 8. VMs: both gone

`champsim-1` (paid account) is TERMINATED; the second account is deleted, so `champsim-2` is unreachable;
nothing runs on the Mac. Every VM measurement is pulled (`results/vm2/results/`: 41 run reports, the three
random1000 seeds complete). Chains in `results/vm2/chains/`; launch pattern `screen -dmS <tag> bash chain_<x>.sh`;
kill by process group.

## 9. PENDING before submission (deadline Sep 25 11:59 UTC)

Decisions today: targets = the mean of random's three 1000-design finals; a seed under the target counts
at the budget (60), no "never"/"miss" wording, no median; the apply arm is shown as the same IPC for less
spend, speedups only for the council as run; the paper is a cache design paper whose loop measures
itself (title chosen); only fixes go upstream, the blocks stay in the repository. Done: the CHIA-driven
loop (section 6), the demo, the paper draft (4 pages).

- [ ] Paper: author names/affiliation (placeholders); the AI-assistance line (the call for papers
      requires one; the user asked to remove it: pending their word); the BO arm question; final read.
- [x] Prune done: `loop/` is `council_loop/`, forest arms and unused chips and cells gone, results cut to
      the cited files (58 MB); the raw state is commit e76f80f on branch `experiments`. Prompts identical.
- [ ] README with the protocol, the demo, install/check, the run commands; HotCRP highlights (section 1).
- [ ] Side branch `experiments` (raw state, made without touching the tree) -> commit main -> release tag
      `frozen-council` -> push; each on the user's word. The repository is public.
- [ ] PRs: fixes only, eight items in `upstream/README.md` (the blocks are not proposed upstream).
- [ ] Regenerate `results/report/*/gating.md` with the mean targets (`council_loop/report.py` takes the target).
- [ ] Optional: 5M/25M council, caps off, 3 seeds (paid VM, ~4 h, ~15 USD).

## 10. Open problems

- Quality equivalence of the gates rests on 4 v 4 seeds per regime; the shadow counterfactual (304
  calls) is the larger sample.
- One caps-off apply seed ended its 60-design budget at 1.497, under the 1.5001 target; counted at the budget.

## 11. How to run (frozen recipe)

```bash
SOC=next OPENINGS=20 SEEDS=4 BUDGET=60 WAVE=5 PLATEAU=20 CONTEXT_GATE=shadow CALL_GATE=shadow \
  LOOP_WARMUP=1000000 LOOP_SIM=2000000 SIM_THREADS=5 PARALLEL_RUNS=4 python -m council_loop.search llama2 <tag> council
# apply gating: CONTEXT_GATE=apply CALL_GATE=apply CALL_GATE_THRESHOLD=0.4 GATE_THRESHOLD=0.5 ANALYST_GATE_THRESHOLD=0.3
# caps off: SOC=nocap ; ablation: STATISTICIAN=0 FILL=0 REPAIR=none
SOC=next SEEDS=3 BUDGET=1000 LOOP_WARMUP=1000000 LOOP_SIM=2000000 SIM_THREADS=10 PARALLEL_RUNS=3 python -m council_loop.search llama2 <tag> random
SOC=next LOOP_WARMUP=1000000 LOOP_SIM=2000000 python -m council_loop.search preview llama2      # every prompt, no model call
SOC=next LOOP_WARMUP=1000000 LOOP_SIM=2000000 python -m council_loop.search gate <cell> 30 0     # the brief gate
TABLE_DIR=results/vm2/results/tables SOC=next LOOP_WARMUP=1000000 LOOP_SIM=2000000 python -m council_loop.ledgers results/ledgers/lean <report.json>... random:llama2:0:1000
python -m council_loop.report results/ledgers/lean results/vm2/results results/report/lean20 shadow=lean20_cap apply=lean20_cap_apply
cd council_loop/chia_blocks && ../../.venv/bin/python -m pytest -q chia && bash check.sh
python3 council_loop/demo.py                                      # the offline readout, no key, no install
bash council_loop/chia_blocks/install.sh .venv                    # blocks + patches into the installed CHIA
bash paper/build.sh && open paper/main.pdf                        # tectonic in ~/.local/bin
```
Mac model calls need `CLOUDSDK_CONFIG=$HOME/.config/gcloud-chia`; the VM uses `GCP_PROJECT=a3-chia-hack26ath-7716 ANALYST_LOCATION=global`; the JEV key is `jev_api` in `.env`.

## 12. Rules of engagement (the user's)

- Every launch needs the user's explicit OK with cost and time; no headline launch until confirmed.
- Domain decisions (which arms appear, which trace, what counts as success, what the paper claims)
  are the user's: lay out options and trade-offs, ask one question at a time; report facts, never
  swap arms or traces alone.
- Iterate fast, read the transcripts, cut a run that is not performing. One council at a time.
- Never commit or push without the user's word; no AI authorship lines in commits.
- Shipped code and docs state plain facts, no dates; deleted material lives in git history.
- Small code chunks, explained in plain words, approval before each new chunk; plain style in
  the shipped folder (loops, full names, no comprehensions).
