Evaluation ledger and model spend
=================================

Two small blocks for measuring a loop itself: what it evaluated and what its model calls cost.
Both are pure Python, run with or without a cluster, and read the records CHIA already keeps.

The evaluation ledger
---------------------

A loop that searches, repairs or evolves a design evaluates candidates one after another. The
ledger is one record of them, kept as JSON lines so a run that dies keeps everything it
measured. Every row says who proposed the candidate, why, what it measured, which constraints
it broke and what the evaluation cost.

.. code-block:: python

   from chia.trace.ledger import Ledger, report, markdown

   book = Ledger("results/ledgers/council_s0.jsonl", objective="ipc")
   book.record(candidate, metrics, source="council:geometry", round_index=3,
               rationale="halving the L2 frees 0.3 W for the prefetcher", violations=(),
               cost={"seconds": 42.0, "usd": 0.02})

   book.best_so_far()          # best feasible value after each evaluation
   book.evaluations_to(1.50)   # evaluations until the best reached 1.50, or None
   book.plateau(20)            # evaluations until no gain over the next 20, or None
   book.wasted_share()         # share of evaluations that did not raise the best
   book.feasible_rate()        # share that broke no constraint

A refused candidate (a non-empty ``violations``) is recorded and shown but never counted as the
best. Two ledgers compare the way the sample-efficiency literature compares methods:

.. code-block:: python

   table = report({"council": council, "random": random_search}, reference="random", window=20)
   print(markdown(table))

The report gives, per ledger, the final value, the evaluations spent, the evaluations to the
reference's final value, the plateau, the wasted share, the feasible rate, the equal-quality
speedup (the reference's evaluations over the evaluations this ledger needed to reach the
reference's final value) and the convergence speedup (evaluations to plateau, reference over
this ledger), plus USD and seconds summed from each row's ``cost``.

``rank_agreement(low, high)`` compares two evaluators of the same candidates, for instance a
short simulation against a long one: Spearman's rho, Kendall's tau and the top-five overlap.

With CHIA present, every ``record`` also logs a ``candidate`` event to the profiler, so
``chia viz-profile`` shows the candidates beside the tasks, and ``LedgerNode`` exposes
``record`` and ``report`` as ``@ChiaFunction`` members so a loop on a cluster records from any
worker.

Model spend
-----------

CHIA's model nodes write what each call used into the profiler: ``input_tokens``,
``output_tokens``, cache tokens, ``cost_usd`` and ``model``. ``spend_summary`` adds them up:

.. code-block:: python

   from chia.trace.spend import read_profile, spend_summary

   summary = spend_summary(read_profile("/tmp/ray/<job>/ChiaProfileCollector.log"))
   summary["total"]["usd"], summary["by_function"], summary["by_model"], summary["unpriced_models"]

A call's own ``cost_usd`` is trusted when a node wrote one; otherwise the price table in
``PRICES`` prices it, and a model the table does not know is summed in tokens and listed under
``unpriced_models``, never guessed.

``LLMSpend`` is a loop-wide cap. Every model call charges it after the call; the charge that
would pass the cap raises ``SpendExhausted``, so a loop stops before it overspends:

.. code-block:: python

   from chia.trace.spend import LLMSpend, SpendExhausted

   cap = LLMSpend("results/spend.json", usd_cap=20.0)
   cap.charge("gemini-3.1-pro-preview", input_tokens=9000, output_tokens=1500)
   cap.remaining()             # {"usd": 19.964, "tokens": None}

The state is a small JSON file under a lock, so the processes of one run share the cap. On a
cluster the same class runs as one detached actor every worker charges::

   spend = ray.remote(LLMSpend).options(name="llm_spend", lifetime="detached").remote(path, 20.0)
   ray.get(spend.charge.remote(model, tokens_in, tokens_out))

Tests
-----

``pytest chia/trace/tests/test_ledger.py chia/trace/tests/test_spend.py`` runs without Ray, a
simulator or an API key.

The spend view on the command line
----------------------------------

With the CLI patch applied, ``chia viz-profile`` gains a format that sums the model calls of one
or more profiler logs, one row per node and model::

   $ chia viz-profile --format spend /data/chia_profiles/my_run
   func,model,calls,input_tokens,output_tokens,cache_read_input_tokens,cache_creation_input_tokens,cost_usd,priced
   prompt,gemini-3.1-pro-preview,16,56897,1371,0,0,0.1302,yes
   TOTAL,,16,56897,1371,0,0,0.1302,yes

Nothing in the loop changes: the rows come from the ``input_tokens``, ``output_tokens``, cache
tokens, ``cost_usd`` and ``model`` fields CHIA's model nodes already write into each call's event.
``--output`` writes the CSV to a file, as for ``--format table``.

Deciders and gates
==================

A decider (``chia.models.decider``) answers typed questions about a small state: a probability
for a yes/no question, a level with its confidence for a score question. ``JevDecider`` is
TypeSafe's Jev, a decision model priced so that asking it about a model call costs about a
thousandth of the call; ``ConstantDecider`` answers fixed values for tests. Any object with the
same ``decide(state, questions)`` method serves. The decider reads text, never numbers to
compute with: the caller describes what a section holds or what a round found, and asks.

The call gate
-------------

``CallGate`` sits in front of optional agent calls. The caller supplies a state summary and one
question per call; the gate returns a probability and a ``consult`` flag per call. In ``shadow``
mode every call is still made and the verdicts are recorded; in ``apply`` mode the calls rated
under the threshold are not made:

.. code-block:: python

   from chia.models.call_gate import CallGate, gate_and_log, log_outcome
   from chia.models.decider import JevDecider

   gate = CallGate(JevDecider(), threshold=0.4, mode="shadow")
   verdicts = gate_and_log(gate, {"issue": issue_text},
                           {"assess": {"instructions": "Does this issue report a defect?",
                                       "criteria": {"true": "existing behaviour is wrong",
                                                    "false": "a request, a question or a usage problem"}}},
                           label="issue 4396")
   if verdicts["assess"]["consult"]:
       answer = get(llm.prompt.chia_remote(llm, prompt))
       log_outcome("issue 4396", "assess", verdicts["assess"], made=True, outcome=answer.result[-40:])

``gate_and_log`` writes one ``call_gate`` profiler event per round (mode, threshold, every verdict,
the decider's own tokens and USD) and ``log_outcome`` one ``call_gate_outcome`` per call, so the
shadow evidence (what would have been skipped, and what those calls then produced) is read from
the same log as the spend. Run in shadow first; the threshold comes from that log.

The context gate
----------------

``ContextGate`` gates what a call reads. For a prompt written in ``## `` sections it asks the
decider, per section, whether the task needs it, from the caller's one-line description of the
section (``describe(name)``), plus a difficulty that maps to the caller's thinking tiers.
Sections whose names start with an ``always_kept`` entry are never asked about. ``shadow``
records what would be dropped and the token estimate; ``apply`` sends the shorter prompt:

.. code-block:: python

   from chia.models.context_gate import ContextGate, gate_and_log

   def describe(name):
       return {"Chip": "the chip card, the same every round",
               "Recent designs": "the last six designs anyone proposed, with knobs and gains"}.get(name)

   gate = ContextGate(decider, threshold=0.5, mode="apply", always_kept=("Role", "Task", "Once more"),
                      describe=describe, tiers={"easy": 1024, "medium": 1024, "hard": 4096})
   gated = gate_and_log(gate, prompt_text, label="geometry")
   answer = llm.prompt(gated["text"])          # gated["thinking_budget"] is the tier chosen

Shown a section's first characters instead of a description, a decider rates a table of numbers
as unneeded; the description is what makes the judgment sensible. The measured caveats of both
gates are in ``UPSTREAM.md``.

Two notes on the profiler
-------------------------

Profiler events are recorded fire-and-forget; call ``ray.get(get_collector().get_events.remote())``
before ``stop_collector()`` or the last events of a run can be lost. The ``ChiaProfiler`` singleton
keeps its collector handle: a driver that stops one collector and starts another in the same
process must call ``reset_profiler()`` in between, or run each job as its own process.
