# weather-mcp RAG Tools: Methodology & Findings

Companion to [`FINDINGS.md`](FINDINGS.md), which covers the eval harness
itself (shared architecture, grading model, LLM-judge variance, cost
tracking) and the findings specific to the original six, non-RAG tools.
This document covers everything specific to the two retrieval-facing tools,
`search_forecast_history` and `explain_forecast_reasoning`: raw
retrieval-layer behavior against the corpus directly, harness bugs the RAG
questions surfaced, and eval-harness-level findings about how a model
actually uses these two tools. Status: covers the 16-question
`questions_rag.py` set and the retrieval-layer testing in
`scripts/test_retrieval.py` that preceded it.

## Raw retrieval-layer behavior (`scripts/test_retrieval.py` against the corpus directly)

This section is one layer below the eval harness -- no model, no tool
calls, just SQL against `weather_discussion_chunks` to characterize what
`embedding <=> $1` cosine search actually returns before ever asking a model
to use it through a tool. Corpus: `scripts/test_data.json`, ~2026-08-29
through 2026-08-31.

### Case: heat ridge tracking (KFWD -> KRAH)

**Ground truth** (found via `chunk_text ILIKE '%heat%'` scan, not
retrieval): a subtropical ridge builds over KFWD (Dallas/Fort Worth)
starting ~Aug 29 and shifts east; KRAH (Raleigh) independently narrates "the
~595 dam mid-level anticyclone will build eastward" the same day, with heat
peaking Aug 31 ("Peak of the heat will likely be Tuesday through Thursday").
Physically consistent west-to-east ridge propagation over the 3-day window.

**Query:** `"a ridge of high pressure aloft building and shifting eastward
bringing hot temperatures"`

**Result:** Sorting the top-20 nearest-neighbor hits by `issued_at`
reproduced the correct chronology and geography for the real signal (KFWD ->
KRAH, Aug 29 -> Aug 31, distances 0.50-0.56) -- proof that timestamp + vector
search can reconstruct a pattern's movement without any manual filtering.

**Caveat:** the same query also pulled in KSEW satellite/cloud-cover
boilerplate and KMPX "low pressure system moving" chunks at nearly identical
distance (0.53-0.56). `all-MiniLM-L6-v2` (384-dim, general purpose) doesn't
cleanly separate "ridge" from "trough/low" -- it's matching on
"meteorologist describing an upper-air feature's location," not the sign of
the feature. The *sorted, office-labeled* KFWD/KRAH subset is the real
result; the raw top-20 is not clean on its own.

**Implication:** distance-only retrieval is convincing for a demo but not
precise enough on its own for anything claiming correctness (e.g. "only
ridge-related chunks"). Would need one of: a bigger/domain-tuned embedding
model, a keyword pre-filter (`chunk_text ILIKE`) narrowing before vector
rerank, or a reranking pass.

### Case: office-scoped topical search

**Query:** `"chance of severe thunderstorms and heat advisories"`

**Result:** clean -- top 5 all landed on KLIX (New Orleans) `KEY_MESSAGES`
chunks, tight distance cluster (0.336-0.365), clear separation from the rest
of the corpus, and surfaced 5 distinct issuance times for the same
office/topic. Retrieval works as expected when the query maps to one
dominant office/topic rather than a cross-country pattern.

### Case: multi-hop causality question ("where did it start")

**Query:** `"Where did the heat wave that is hitting Greensboro start?"`

**Ground truth check first:** "Greensboro" never appears anywhere in the
corpus as a word; the station code `KGSO` appears exactly once, buried in a
KRAH record-highs list. No keyword shortcut available -- any correct answer
has to come from semantics, not string matching.

**Result (2-office corpus, before backfill):** all top 6 nearest-neighbor
hits landed on KRAH, correctly, purely on semantic grounds
(heat/humidity/central-NC language) -- a genuine win, since it means
retrieval can identify "which office actually covers this place/topic"
without a city-to-office lookup table. **But it did not answer the
question.** None of the KFWD ridge-origin chunks (the actual "where it
started" answer) showed up in the top 6 -- the query's similarity to "heat
is happening in central NC" dominated and the search never reached back to
the upstream cause. A single nearest-neighbor query against one embedding
cannot chain "RAH is hot because a ridge built over KFWD" -- that fact only
exists by combining two offices' text, and vector search alone doesn't do
that combination step.

**Corpus backfill:** with only KFWD (origin) and KRAH (destination) in the
corpus, there was no intermediate evidence a causality-tracing answer could
walk through. RAH's own discussion text names the missing middle -- the
ridge becomes "nearly stationary over the TN Valley and southern
Mid-Atlantic" -- so 5 offices were added along the stated path: KSHV
(Shreveport, immediate hop out of north Texas), KMEG/KOHX (Memphis/Nashville,
the named "TN Valley"), KFFC (Atlanta, Deep South leg), KCAE (Columbia,
named "southern Mid-Atlantic," last hop before NC). 161 chunks total match
`ridge` or `heat` across these 5 offices -- a real, traceable west-to-east
chain: KSHV ("upper ridge moves into the MidSouth" Aug 30 -> "migrates east
across the Mid-Miss[issippi]" Aug 31) -> KMEG ("centered over the Southern
Plains" Aug 30 AM -> "... and Lower [MS Valley]" Aug 31 AM -> "remains
centered over the Ozarks" Aug 31 PM) -> KOHX ("ridge continues to build" ->
"ridge will start to break down" Aug 31 PM) -> KFFC ("ridge will continue to
build eastward" -> "expansive upper ridge over the south-central U.S.") ->
KCAE ("anomalously strong upper ridge" as early as Aug 29).

**Re-ran the same query against the fuller corpus.** Result: still
dominated by KRAH -- 12 of the top 15 hits are KRAH chunks, and only one
intermediate hop (KMEG's "ridge remains centered over the Ozarks") cracks
the top 15, at rank 13. None of KSHV/KOHX/KFFC/KCAE appear in the top 15
despite being genuinely on the path and now present in the corpus.

**Working conclusion:** having the intermediate evidence in the corpus was
necessary but not sufficient. A single global top-k query still won't
assemble the chain on its own -- it's dominated by whichever office's
language most directly matches the query's surface wording (here, KRAH's
"heat" vocabulary beats KMEG/KFFC's "ridge" vocabulary even though the ridge
chunks are the actual causal answer). For a causality question like this to
work reliably, retrieval needs to either (a) diversify by office (top-N per
office instead of one global top-k) so every office on the path gets a
chance to surface, or (b) run explicitly as multiple queries/hops -- first
identify the relevant office(s), then separately query for "upstream
cause/origin" language -- and hand the LLM the union of all of it. Retrieval
narrows candidates; it does not chain causality on a single pass, even when
the chain is fully present in the data. This dataset (16 offices, a real
traceable west-to-east chain, and a documented single-pass failure mode) is
the fixed benchmark for the "adaptive/multi-hop retrieval (v3)" idea below.

### Open questions never followed up at the raw-SQL layer

- Does retrieval correctly distinguish two offices with genuinely similar
  weather (e.g. two Gulf Coast offices both discussing tropical moisture)
  rather than conflating them?
- What does a *negative* test look like -- a query with no good match in the
  corpus (e.g. "snow accumulation") -- does distance blow out appropriately,
  or does it still confidently return top-5 garbage?
- Chunk-type filtering (`WHERE chunk_type = 'DISCUSSION'`) combined with
  vector search hasn't been tried -- likely reduces the ridge/trough
  confusion above since `OTHER`/aviation-style chunks carry a lot of the
  noise.

## Harness bugs found while running the RAG-tool eval questions

Not `weather_mcp` bugs, not RAG-quality findings -- bugs in `run_eval.py`
itself, all only surfaced once the RAG questions started producing larger
inputs/outputs than the original 16-question set ever did:

1. **Headless auth conflict**: `claude -p` refused to run at all
   (`API Error: 401` or a connectors-disabled warning) whenever
   `ANTHROPIC_API_KEY` was set in the environment -- it took precedence over
   the CLI's own subscription login. Not a code bug (the `api` backend needs
   that variable), but worth documenting: run the headless backend with it
   explicitly unset for that invocation (`env -u ANTHROPIC_API_KEY ...`),
   don't unset it globally.
2. **Windows `cp1252` decode crash**: `subprocess.run(cmd, text=True, ...)`
   with no explicit `encoding=` defaults to the OS locale's codepage on
   Windows, not UTF-8. The moment any subprocess output (weather alert text,
   AFD text, or the model's own answer) contained a multi-byte UTF-8
   character `cp1252` can't represent, the internal reader thread crashed
   silently, leaving `result.stdout` as `None` and the real error hidden
   behind an unrelated `AttributeError: 'NoneType' object has no attribute
   'splitlines'`. This was luck-of-the-content, not content-specific --
   fixed by passing `encoding="utf-8"` explicitly.
3. **Windows command-line length limit (`WinError 206`)**: the judge prompt
   embeds the *entire* raw tool response text (`build_judge_prompt`'s
   `tool_data`), and `search_forecast_history` can return several KB across
   `top_k` full chunks -- multiple KB more than any of the original 16
   questions' tool responses ever produced. Passing that as a positional CLI
   argument to `claude -p` exceeded Windows' (much tighter than Linux's)
   command-line length limit. Fixed by piping the prompt via stdin instead
   (`subprocess.run(cmd, input=prompt, ...)`, with `-p` given no positional
   argument) -- confirmed working directly with a 20KB test prompt before
   rolling it into the harness. This scales to arbitrarily large prompts
   regardless of platform, so it's a strict improvement even for the
   original questions, not just the new ones.
4. **Anonymous HuggingFace Hub rate-limit risk**: each fresh subprocess
   (every `claude -p` call spins up its own MCP server process, and the
   embeddings model is a per-process lazy singleton) re-authenticates
   anonymously against HF Hub, logged explicitly:
   `Warning: You are sending unauthenticated requests to the HF Hub`. Not an
   actual 429 yet, but real, avoidable load across dozens of subprocess
   spawns per run -- mitigated two ways: `--skip-seed` to avoid re-seeding
   (and re-embedding all 160 fixture records) on every retry, and
   `HF_HUB_OFFLINE=1` once the model is already cached locally, which
   eliminates the network round-trip entirely.
5. **A flat, unscoped `ALLOWED_TOOLS` burned real usage on the wrong tool.**
   Before the fix, every question -- RAG or not -- got all 8 tools in its
   `--allowedTools` list, including `get_weather_discussion` (a full live AFD
   product, 2-5KB of raw text). No `questions_rag.py` `expected_calls` ever
   call for `get_weather_discussion`/`compare_forecasts` (checked directly:
   zero matches), but the model under test called it anyway on several
   RAG-only questions when it wasn't sure what else to try. That raw text then
   got embedded verbatim into `build_judge_prompt`'s `tool_data` and re-sent
   once per `--judge-samples` (3x) -- one stray call effectively quadrupled in
   spend, and showed up as unexplained full-discussion dumps in the run log.
   A 15-question RAG-only run burned roughly 50% of a usage budget this way.
   Fixed by scoping the allow-list per question via `select_questions()`
   pairing each question with `MCP_TOOLS` or `RAG_TOOLS`, applied to both
   backends (the headless `--allowedTools` flag, and a per-question filter on
   the API backend's `anthropic_tools` list). Confirmed working: a full
   16-question RAG rerun after the fix cost $1.41 total (model + judge x3),
   down from burning half a budget on 15 questions before it.
6. **Headless `claude -p` can read its own answer key off disk.** Once (5)
   correctly blocked `get_weather_discussion`, that same rerun still had
   `search_forecast_history` failing (Postgres was down -- see below), and
   with no working tool the model didn't just say so: on the two ridge-chain
   questions it **read `tests/eval/questions_rag.py` directly** (headless mode
   runs as a full coding agent in `REPO_ROOT`, and `--allowedTools`/
   `--strict-mcp-config` only govern MCP tool access, not the CLI's own
   built-in Read/Grep/Glob/Bash) and recited the `expected_calls`/
   `expected_facts` ground truth verbatim, complete with citing the line
   numbers. Both got `fact_score: 10` -- a fabricated signal, not evidence of
   anything about retrieval quality. Fixed by adding `--disallowedTools` with
   the built-in tool set (`Read,Grep,Glob,Bash,Write,Edit,NotebookEdit,
   WebFetch,WebSearch,Task`) to every headless model-under-test invocation.
   The `api` backend was never at risk -- it's a hand-rolled loop that only
   ever hands the model the MCP tool schemas, no built-in tools exist to leak
   through. **Any fact_score from a headless run before this fix should be
   treated as untrustworthy if the corresponding tool call could plausibly
   have failed.**
7. **Postgres wasn't running for a full run and nothing surfaced it clearly.**
   Same rerun as (6): `weather-mcp-db-1` had exited hours earlier, so roughly
   half the questions got `[WinError 1225] The remote computer refused the
   network connection` from `search_forecast_history`/`explain_forecast_reasoning`.
   `tool_score` still graded these as high (it only checks tool name/params,
   not success), so a skim of the summary numbers alone would not have shown
   the DB was down -- only reading individual `tool_data` fields caught it.
   Not a code bug, but worth the harness lesson: `docker ps` the DB container
   before trusting a run's results, especially after any gap between runs.

## `quality_score` is not reliable enough to stand alone on checkable questions

Confirmed with a concrete example, not just the judge-variance noise
documented in `FINDINGS.md`. Two of the RAG-tool questions have an
objectively correct answer (a specific NWS office code; an ordered chain of
office codes). On the same run: a question where the model named the
*wrong* office as a ridge's origin (KMEG instead of the true origin, KFWD)
was judged **6-8/10** -- plausible and well-formatted, and wrong. A question
where the model got a 7-office chain **6-of-7 correct, in the right order**
was judged **2-5/10**. This isn't the same phenomenon as judge-variance
(same input, different scores across runs) -- these are two different
questions, each judged once per sample, landing on opposite sides of
correct. Added `grade_facts()` to `grading.py` and `expected_facts` to
`questions_rag.py` for exactly these questions -- a deterministic text check
(does the answer mention the right office code(s), in the right relative
order) that scores the KMEG answer `0` and the 6-of-7 chain answer `9`,
catching both misses `quality_score` alone did not. This is a supplement,
not a replacement -- most questions don't have an answer this checkable, so
`quality_score` still carries them.

## RAG retrieval-recall finding (clean rerun, DB up, no leakage)

With harness-bug items 5-7 above fixed and Postgres confirmed running, a
full 16-question `--question-set rag` run produced honest, non-leaked
results for the first time: avg `tool_score` 3.7, avg `quality_score` 4.2,
$2.25 total cost (model + judge x3) -- the real baseline going forward, not
the earlier bug-inflated figures.

The three `expected_facts` questions scored **0/10, 8/10, 4/10** this time
(previously 10/10/0 under the leaked run -- see item 6 above for why those
numbers were fake). The 0/10 is itself a genuine, reproducible retrieval
finding, not a harness bug: asked "which office first reported the ridge
that eventually brought heat to Raleigh," the model called
`search_forecast_history(query="ridge heat Raleigh")` at the default
`top_k=5` and got back only `KRAH` and `KOHX` chunks -- never `KFWD`, the
true origin office and the one geographically farthest from Raleigh. The
model then confidently answered "KOHX first reported it," which is wrong.
This is a live instance of exactly the failure mode the multi-hop-causality
case above predicted from raw SQL testing: a query whose surface wording
matches the *destination* office's vocabulary out-competes the *origin*
office's differently-worded but causally correct chunks, and a narrow
`top_k` never gives the origin a chance to surface.

## Eval-harness-level findings: how a model actually uses these two tools

The findings above are either at the raw-SQL layer or about harness bugs.
This section is `tests/eval/run_eval.py` actually asking a model to choose
and use `search_forecast_history`/`explain_forecast_reasoning` on its own.

**The model can already approximate "multi-query expansion" on its own --
this changes what a v3 adaptive-retrieval feature needs to be.** The
full-chain question ("list every office that mentioned this ridge") is the
exact scenario the multi-hop-causality case above identifies as needing an
adaptive retrieval step. Given the freedom to make several tool calls, the
model made 3 `search_forecast_history` calls with differently-worded queries
and self-selected `top_k` of 20-30 (well above the default 5) -- and got
**KFWD -> KSHV -> KMEG -> KOHX -> KFFC -> KRAH**, 6 of the 7 documented
offices, in the correct order (only missing KCAE). That's the model
spontaneously doing the diversify-by-query strategy that raw-SQL testing
suggested would require *new adaptive-retrieval code* -- it did it
unprompted, with the tools as they exist today. This doesn't mean adaptive
retrieval is unnecessary (a single default-`top_k` call still fails, as the
origin-office finding above shows), but it reframes the fix: possibly a
docstring nudge toward "try several phrasings with a wider `top_k` for a
tracing/pattern question" gets most of the way there, rather than requiring
new retrieval architecture. Worth testing directly before building anything
more elaborate. Notably, this spontaneous widening did *not* happen on the
narrower origin-office question above (which failed at default `top_k=5`
without a retry) -- worth checking whether that's phrasing-dependent or just
run-to-run inconsistency.

**The model consistently avoids the `office_id` parameter, even for
easy-to-resolve city names.** Across the Dallas/Greensboro, sharper-origin,
and full-chain questions, the model repeatedly called `search_forecast_history`
with the city name folded into the free-text `query` string (e.g.
`"end of August weather conditions Dallas Texas"`) rather than setting
`office_id="KFWD"` -- including for "Raleigh," the one case where the
office code is directly derivable from the city name (KRAH). This held even
though `search_forecast_history`'s docstring explicitly recommends office
scoping. Unscoped search still worked reasonably well in these cases (the
corpus is small enough that relevance alone often lands on the right
office), but this is exactly the behavior that would degrade first as the
corpus grows -- worth a docstring pass emphasizing office scoping more
forcefully, and/or a follow-up eval run to see if it's consistent or noise.

**A "declined honestly" strategy was worth adding to a real question, for
the same reason already documented for Atlantis/Springfield/London in
`FINDINGS.md`.** Asked to compare Dallas and Greensboro's end-of-August
weather, the model made zero tool calls and explained that neither RAG tool
provides actual historical *observations* (only forecast discussion text)
-- arguably the most epistemically honest response available, not a failure
to attempt something. `questions_rag.py`'s `expected_calls` for that
question originally had no zero-call strategy, so this correct behavior
would have scored a flat 0. Fixed by adding a `declined` strategy -- but it
had to be ranked *between* the two real-attempt strategies (not above them,
unlike Atlantis), since `grading.py`'s exclusive-cluster logic picks
whichever strategy in a tied/highest-scoring cluster "wins," and a
`declined` strategy tied with or above a genuine multi-call attempt would
incorrectly cannibalize credit for that attempt. Confirmed empirically
against `grade_question` before trusting it.

**Two known-fragile questions found and fixed while running this for
real.** The original "mountains affect KRAH" pairing test worked as
designed (see below), but two *other* questions turned out to have real
authoring bugs, not model or retrieval failures: the full-chain question
originally said "this ridge" with no antecedent -- each eval question fires
as an isolated, context-free prompt, so the model correctly asked for
clarification instead of guessing. Reworded to name the ridge and timeframe
directly in the same prompt (see `questions_rag.py` comments for the fix).

**The mountains/KRAH fabrication test worked exactly as designed, and more
precisely than expected.** The model retrieved real KRAH content (Western
Piedmont fog patterns, Eastern Piedmont heat differences) and then
fabricated the *causal* mechanism on top of real citations -- "this is
exactly how orographic effects work" -- even though the retrieved text never
actually attributes those differences to mountains or terrain. This is a
subtler failure than inventing content outright: real citations, invented
causation. The judge scored it correctly (2/10). The KFFC positive-case
pairing was less conclusive -- the model skipped retrieval entirely on that
one (zero tool calls, answered from outside knowledge, plus an odd aside
asking whether this was "for the evaluation harness") -- worth a follow-up
run to see if that's consistent.

## Future work: adaptive/multi-hop retrieval (v3)

The multi-hop-causality case above is the concrete evidence for this:
single-pass top-k over one query embedding is structurally the wrong tool
for a causality question, no matter how good the corpus is (161 ground-truth
ridge/heat chunks vs. only 1 surfacing in the top 15 at raw-SQL layer). The
fix isn't more data, it's an adaptive retrieval step -- something that,
given a question shaped like "why/where did X start," runs a first pass to
identify the relevant office(s) and time window, then issues a second,
differently-worded query (e.g. swap "heat" for "ridge"/"upstream cause") or
diversifies by office before handing chunks to the LLM.

**Update after the eval-harness runs above**: given the freedom to make
several tool calls, the model already approximated this diversify-by-query
strategy on its own for the full-chain question -- 3 differently-worded
searches with a wide self-selected `top_k` got 6 of 7 offices in the correct
order, but the narrower origin-office question still failed at default
`top_k`. Before building new adaptive retrieval code, worth testing whether
a docstring nudge toward multi-query search for tracing/pattern questions
gets most of the benefit for far less engineering cost -- this dataset (16
offices, a real traceable west-to-east chain, and both a documented
single-pass failure and a documented spontaneous-success) is a good fixed
benchmark for that work either way.
