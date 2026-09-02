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

- `loop/configs.py` — Tier-A search space (81 configs; **placeholder values pending team review**) + ChampSim config generation
- `loop/simulate.py` — the only file touching the simulator: `build_binary()`, `run_simulation()` (JSON stats → flat metrics dict)
- `loop/playbook.py` — rules + bet ledger as one auditable JSON file; Brier scoring
- `loop/analyst.py` — all Gemini calls (Vertex AI, `gemini-2.5-flash`); logs tokens/cost per call to `loop/llm_usage.json`
- `loop/loop.py` — wires it together: seed → hypotheses bet → simulate → settle → repeat
- `proposal.tex` — the accepted proposal

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

## Status (Sep 2, 2026)

MVP loop **validated end to end** locally: 2 autonomous rounds on mcf, 6 settled bets, correctly falsified the L2-capacity hypothesis and confirmed prefetching (+31% IPC), $0.004 LLM spend.

**Next steps, in order:** (1) surrogate model + rules-as-priors + disagreement-based selection, (2) multi-trace (lbm, gcc — workloads that break rules and force re-scoping), (3) distill step last (earlier would produce static textbook rules). Then **full CHIA-ification** (currently ~zero, judges are the CHIA team): `@ChiaFunction` nodes, analyst via `chia/models/vertex.py`, `chia up` GCP cluster (Tailscale), upstream PRs (gs:// trace-resolver fix — open early; config-space ChampSim node; playbook/ledger blocks). Possible stretch arm: teammate's Merlin compiler node for true HW/SW co-design (blocker to verify: trace regeneration after compiler changes).

## Working rules

- Budget: **~257 EUR total** GCP credits. Track LLM cost per call; expensive runs need the user's approval with a cost estimate first.
- Wrap long-running shell commands in `caffeinate -i`.
- Code style: explicit, procedural, junior-readable; no clever one-liners; small chunks with user approval between them.
- Domain decisions (search space, SoC profiles, claims) belong to the user and their PhD teammates — surface options, don't decide.
