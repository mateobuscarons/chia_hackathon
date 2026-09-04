# CHIA Hackathon — Hypothesis-Driven Cache Optimization

A3 workshop hackathon (agentic-arch.org). Deadline **Sep 20, 2026**: 4-page paper + open-sourced CHIA loop + results. Judged by the CHIA team (UC Berkeley SLICE lab) — an academic program committee that rewards novel loop mechanisms and reusable CHIA blocks, not raw IPC numbers.

## Core design (frozen): "rules that bet"

An LLM agent optimizes cache hierarchies (ChampSim) and distills an **architect's playbook**: explicit rules with the form *(condition, claim, worked example)*, each carrying a win/loss record and Brier score.

- Every hypothesis/rule logs a quantitative prediction **before** each simulation; the simulator settles the bet. The LLM never grades itself.
- Experiments are chosen where forecasters (rules, hypotheses, surrogate) **disagree most**; agreed-upon outcomes are skipped.
- Rules that lose bets get **re-scoped** (condition sharpened), not deleted.
- Rules act mechanically: they prune the search space / warm-start the surrogate, weighted by track record.
- **Headline claim:** rules learned on SoCs A/B cut simulations-to-target on unseen SoC C vs surrogate transfer and cold start.
- Key baseline arm: agent seeded with textbook-only knowledge, no loop — the delta measures the loop's value (and defends against "the LLM already knew this").

## Repo layout

See `README.md` for the file table. Key points: `loop/loop.py` is simulator-agnostic; `loop/champsim_problem.py` is the only ChampSim glue; `loop/chia_nodes.py` + `loop/run_chia.py` run it as a CHIA loop; `loop/socs.py`/`configs.py` hold **placeholder** SoC profiles, budgets and search space pending team review; `proposal.tex` is the accepted proposal.

## Setup (not in repo)

```bash
git clone --depth 1 https://github.com/ChampSim/ChampSim.git champsim
cd champsim && git submodule update --init && ./vcpkg/bootstrap-vcpkg.sh && ./vcpkg/vcpkg install
./config.sh champsim_config.json && make -j8 && cd ..
git clone --depth 1 https://github.com/ucb-bar/chia.git chia
uv venv --python 3.10 .venv && uv pip install -p .venv/bin/python -e ./chia google-genai
mkdir traces && curl -o traces/605.mcf_s-665B.champsimtrace.xz \
  https://dpc3.compas.cs.stonybrook.edu/champsim-traces/speccpu/605.mcf_s-665B.champsimtrace.xz
```

GCP: project `project-c23a6080-f5d0-4871-9cb`, ADC auth (`gcloud auth application-default login`), `aiplatform.googleapis.com` enabled. Gemini runs on the project's credits.

## Status (Sep 2, 2026, evening)

End-to-end pipeline built and tested on partial data; **dense sweeps running** (`run_sweeps.sh`, 9 SoC x trace tables into `results/`). Done today: generic simulator-agnostic loop (`loop/loop.py` takes a `problem` dict), SoC profiles + hard area budget (`socs.py`, placeholders), additive + GP surrogates, disagreement-based selection, bets by every forecaster settled at the dispute midpoint, rule re-scoping, distill + textbook rules, transfer experiment with 7 arms (`experiment.py`), figures (`plots.py`), CHIA nodes (`chia_nodes.py`: config-space build, CHIA `ChampSimNode.run_champsim`, Vertex analyst) validated on local Ray, cluster yamls, gs:// patch for upstream (`upstream/`), README, paper skeleton (`paper/`).

**Next:** (1) when sweeps finish: `python -m loop.experiment traces/...` on all 3 traces, then `python -m loop.plots`; read the playbook and the losing/re-scoped rules — that is the paper's story. (2) Team review of placeholders: SoC profiles, area budgets, search space. (3) GCP cluster bring-up (`cluster/README.md`, needs Tailscale key from user). (4) Tier B wide space on GCP. (5) Fork CHIA and open the gs:// PR. (6) Optional: humans-bet arm.

## Problem scope (Sep 4, 2026): real problems are hard problems

**What a real memory-hierarchy design problem looks like.** Per cache level: size, associativity, line size, replacement, prefetcher and its parameters, MSHR/queue depths; plus coherence, interconnect and DRAM settings. Millions of combinations under area and power budgets, so knobs interact (area spent on L2 is area not spent on LLC). One detailed simulation (gem5, RTL) costs hours to a day per design per workload; teams evaluate tens to low hundreds of designs per cycle across 20-50 workloads. Every new SoC generation starts from the previous one: the transfer setting (chips A, B -> unseen C) *is* the industrial loop.

**Hardness, measured.** A problem is hard when four things hold together: a wide space (tens of thousands of designs or more), a sparse near-optimal band (a few percent or less of feasible designs within 2% of the best), large headroom (best vs untouched chip differs by tens of percent, so "good" and "best" are far apart), and knobs that interact through a constraint. Measure it with a small random probe: if random search reaches 90% of the gain in ~10 draws, the problem is easy and statistics alone suffice. Evidence: on Tier A (54 feasible designs, 8-30% of them near-optimal) every method converges in 2-7 simulations and nothing beats a GP; on the hard Tier B problems (46k designs, 1-3% near-optimal, +54-70% headroom) no statistical method reached within 2% of the best in 24 simulations and the full loop did.

**Scope going forward.** Tier A is a control and a unit test, not a target. The target is the hard regime: Tier B and harder, evaluated at explicit hardness levels, with the main claim made on the hard level. Baselines must scale with the problem (a fair Bayesian-optimization baseline, not a weakened one), and hardness must come from realism (more levels, more interacting knobs, more workloads, real budgets), never from hobbling the comparison.

## Stretch arm: Merlin compiler node (HW/SW co-design)

A PhD teammate (Berkeley, close to the CHIA project) is integrating a **Merlin compiler node** into CHIA so an agent can change the compiler/compilation agentically. For us that is a second knob axis: the `problem` dict in `loop/loop.py` is search-space agnostic, so compiler choices become knobs next to cache knobs, and `evaluate` becomes compile -> re-trace -> simulate. Rules can then say "when the compiler does X, cache knob Y stops mattering", which is the co-design story the CHIA team wants. **Blocker to verify before committing:** ChampSim traces are recorded from a fixed binary, so every compiler change needs trace regeneration (tracer availability, time per trace). Do not start this before the cross-SoC result is in hand.

## Working rules

- Budget: **~257 EUR total** GCP credits. Track LLM cost per call; expensive runs need the user's approval with a cost estimate first.
- Wrap long-running shell commands in `caffeinate -i`.
- Code style: explicit, procedural, junior-readable; no clever one-liners; small chunks with user approval between them.
- Domain decisions (search space, SoC profiles, claims) belong to the user and their PhD teammates — surface options, don't decide.
