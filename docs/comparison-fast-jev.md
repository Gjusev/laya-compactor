# laya-compactor vs fast-jev-compaction

A comparison with [fast-jev-compaction](https://github.com/tamaratran/fast-jev-compaction),
the leading context-compaction project in the Jev ecosystem (listed at 2.7k
stars by the Jev Radar this project's plan cites). We read its source
(`src/compact.ts`) before building laya-compactor and adopted its best idea —
delete, don't rewrite — so this is a comparison between two projects that
share a philosophy but solve different problems.

## The short version

They do not compete head-to-head. fast-jev-compaction compacts **agent
transcripts** (Claude Code session history: tool calls and their results).
laya-compactor compacts **RAG retrieval batches** (documents scored against a
query). Each is purpose-built for its own unit of work; using either for the
other's job would fight its design.

## Architecture, side by side

| Dimension | fast-jev-compaction | laya-compactor |
|---|---|---|
| Domain | Agent transcripts (tool calls + results, paired by id) | Retrieved documents vs a query |
| Decision question | Two binary (`noul`) questions per call: should the call stay / should the result stay verbatim | One ordinal (0–3 relevance) `score` question per document |
| Context per decision | The whole conversation, fitted into 25k tokens through staged degradation (tool inputs truncated 1000→200→60 chars, old messages collapsed), resent with every request | Query + document only |
| Decision cost | Cloud API calls; the full state is repeated for every batch of questions | One shared forward pass (`predict_batch`) over the whole batch, local checkpoint |
| Decision tiers | 3: keep / keep call with a 300-char truncated result head / drop both | 2: keep verbatim / cut with a reason line (plus `min_score` threshold and token budget) |
| Domain safeguards | Pins the first message and the newest 6; never leaves a result without its call | Relevance-ordered keeps, measured gold-document retention |
| Fallback | If reduction < 25%, keeps the original transcript | Not applicable (the caller sets the budget) |
| Distribution | Claude Code plugin + npm library | pip + LangChain compressor + LlamaIndex postprocessor + CLI |
| Privacy / dependency | Paid cloud API per compaction (key, network) | Local checkpoint; offline after first download; no per-call cost |

## Where each one wins

**Transcript compaction: fast-jev-compaction.** The signal that decides
whether an old tool result matters is *what the assistant will do next*, and
that requires seeing the whole conversation — which is exactly what their
design pays for by resending the fitted state with every request. Their
recency pinning, call/result pairing, and the three-tier "keep a 300-char
head" option are excellent transcript-specific engineering. A truncated
300-char head of a tool result is often exactly enough; a truncated RAG
document usually loses the very fact it was retrieved for.

**Retrieval compaction: laya-compactor.** For RAG there is no dialogue state
to consult — topical relevance to the query is the right signal, and it can
be scored for the entire batch in one local forward pass. It is also the only
one of the two with published benchmark evidence in this domain (below).

**Decision economics.** Structurally, one local forward pass for N documents
beats N API requests that each carry a 25k-token state. The honest
counterpoint: on a desktop CPU our measured compaction latency is 6–10 s per
question batch, which a single cloud call can beat today; on the GPU laya
documents (~35 ms per decision on a T4) the economics invert decisively.

**Evidence culture.** Both projects are honest about limits — their README
publishes a limitations section, ours publishes losses. The difference:
laya-compactor ships a reproducible benchmark with baselines and its
sensitivity study; fast-jev-compaction ships unit tests against a fake Jev
and a demo, but no published quality table. We consider their product
polish (plugin install, fallback UX) better than ours, and our measured
evidence better than theirs.

## The measured numbers (ours)

Generator and judge: glm-5.3-flash via the Z.ai OpenAI-compatible API, 200
questions per dataset, shared 1,000-token budget for the three budgeted
policies (full context is the unbudgeted reference):

| | full | head | tail | laya-compactor |
|---|---|---|---|---|
| SQuAD exact match | 0.345 | 0.320 | 0.265 | **0.345** |
| SQuAD avg input tokens | 3214 | 1050 | 1040 | **973** |
| HotpotQA exact match | **0.230** | 0.175 | 0.110 | 0.200 |
| HotpotQA avg input tokens | 1440 | 1042 | 1041 | **979** |
| HotpotQA gold docs kept | **100%** | 87.5% | 58.8% | 94.5% |

Single-hop retrieval: identical quality to full context at 30% of the tokens.
Multi-hop: 87% of full's exact match at 68% of the tokens, beating both
truncation baselines at the same budget — with the 3-point loss to full
published rather than hidden. The rubric sensitivity study (three phrasings,
gold agreement 0.68 / 0.53 / 0.68 on the published 100-row labeled set) is
the kind of self-measurement that, as far as we know, has no equivalent in
the Jev compaction ecosystem.

## A possible v2 experiment

Their transcript problem could be attacked with our machinery: laya's
`predict_batch` scoring every tool call/result against the conversation
summary as state, with a "does this still matter?" rubric. Structurally
feasible; nobody has measured it. If you try it, tell us what you find.
