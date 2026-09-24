# CHIA's CIRCT assess stage, tracked and gated

CHIA's paper (section 5.5) triages GitHub issues for the CIRCT compiler with an agent loop whose first
stage, *assess*, decides per issue whether it is a bug with one clear fix. Table 5 of the paper gives the
loop's verdict on 16 randomly chosen issues; Table 6 gives the stage's cost, 0.40 USD an issue.

This folder re-runs that stage under the blocks of this repository: the paper's own `assess.md` prompt
(from `chia/examples/circt_issue_solver/prompts`), the 16 issues as public GitHub text with the comments
written before the study, CHIA's Vertex model node (`gemini-3.1-pro-preview`) under CHIA's profiler, and
the CHIA call gate in front of the agent turn asking the decider one question: *does this issue report a
defect?* Everything a judge needs is here: `issues.json` (the rendered issues), one profiler log per pass,
`verdicts_<pass>.jsonl` and `answers_<pass>.json`.

Two deviations from the paper's stage, both forced: the replica has no CIRCT checkout, so the paragraph
offering the `circt_bash` tool is removed from the prompt and the system prompt is one line (offered the
tool, Gemini attempts the call, Vertex answers `MALFORMED_FUNCTION_CALL`, and CHIA's node returns an
empty success: 14 of 16 issues came back blank before the change); and the replica retries Vertex's 429s,
which the node does not.

## Passes

| pass | gate | agent turns | verdicts equal to the paper's | issues sent on to reproduction |
|---|---|---|---|---|
| plain | none | 16 | 15 of 16 | the paper's seven |
| plain2 | none | 16 | 15 of 16 | the paper's seven |
| plain3 | none | 16 | 13 of 16 | six (#6740 came back "fix unclear") |
| gated | call gate at 0.65 | 13 | 14 of 16 | six (#6740 came back "fix unclear") |
| gated2 | call gate at 0.65 | 13 | 15 of 16 | the paper's seven |

The three issues the gate skipped (#2669, #4396, #7127) are non-bugs in the paper and in every ungated
pass, and both gated passes skipped exactly those three. The one verdict that moved, #6740, moved on a
call the gate did not touch and moved again in an ungated pass: the model's own variance, not the gate's. The gate caught 3 of the 5 non-bugs and none of
the 11 bugs; the two non-bugs it let through are the two the paper's own agent also had to read (#2266
and #10571 both argue about intended behaviour).

## What the spend view printed

`chia viz-profile --format spend <pass directory>` over the profiler log CHIA wrote:

| pass | calls | input tokens | output tokens | USD (Gemini list price) |
|---|---|---|---|---|
| plain | 16 | 35,761 | 2,126 | 0.097 |
| plain2 | 16 | 35,761 | 1,813 | 0.093 |
| plain3 | 16 | 35,761 | 1,975 | 0.095 |
| gated | 13 | 28,419 | 2,207 | 0.083 |
| gated2 | 13 | 28,419 | 1,773 | 0.078 |

Calls −19%, input tokens −21%, spend −14% to −19% (the answers' length varies by pass; calls and input
tokens are the stable measure). The decider's 16 decisions cost 0.001 USD (`call_gate` events in the gated log carry
its tokens). At the paper's own price per turn, 0.40 USD, the gate removes 1.20 of the stage's 6.40 USD.

## The sentence

With CHIA's own assess prompt on CHIA's own Vertex node, applying the call gate, the stage sent the same
issues on to reproduction with 19% fewer agent turns and 21% fewer input tokens, for a tenth of a cent of
decider time.
