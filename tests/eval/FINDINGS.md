# weather-mcp Eval Harness: Methodology & Findings

Status: covers work through the `run_eval.py` unification (headless + API
backends, both now verified live), the 32-question dataset split across
`questions_mcp.py` (16 original) and `questions_rag.py` (16 exercising the
RAG tools), `grade_facts` for objectively-checkable questions, and per-set
tool scoping (`--question-set mcp|rag|both`) to keep RAG-only runs from
spending usage on the non-RAG tools. Update as more runs happen.

## What this harness measures

Two independent axes, scored separately on purpose:

- **`tool_score`** -- did the model call the right tool(s), with the right
  params, for a given natural-language question? Deterministic, rubric-based,
  computed by `grading.py`. No LLM involved in this score at all.
- **`quality_score`** -- is the model's final answer to the user actually good
  (accurate, grounded in the tool data, well-formed)? Judged by a second LLM
  call, averaged over `JUDGE_SAMPLES` (default 3) samples per question.

Keeping these separate was itself a finding: a run can score a perfect 10 on
tool selection while containing a real factual error a user could act on badly
(see "The Boone trip question" below), and single-tool-call correctness alone
would never have caught it.

A third, narrower check exists for the handful of questions with an
objectively checkable answer: **`fact_score`**, from `grade_facts()`,
deterministic like `tool_score` but checking the *answer text* against a
known ground truth (a specific office code, an ordered chain of offices)
rather than the tool calls. See "quality_score is not reliable enough to
stand alone on checkable questions" below for why this exists -- it isn't a
general replacement for `quality_score`, most questions don't have an answer
this checkable.

## Architecture

- `questions_mcp.py` / `questions_rag.py` -- the question set (32 total: 16
  original non-RAG questions, 16 exercising `search_forecast_history`/
  `explain_forecast_reasoning`), each with an `expected_calls` list describing
  acceptable tool-call strategies, and optionally an `expected_facts` dict for
  questions with a checkable answer. Split into two files (was one
  `questions.py`) so a retry scoped to one tool family doesn't re-spend usage
  re-testing the other; `select_questions()` in `run_eval.py` picks one, the
  other, or both via `--question-set`. Shared by both backends so results are
  comparable regardless of how the model was invoked.
- `grading.py` -- pure, deterministic grading logic (`grade_question` for
  tool calls, `grade_facts` for answer-text ground truth), fully unit-tested
  (`test_grading.py`, 18 tests) with no network calls.
- `run_eval.py` -- the harness itself, with two backends:
  - `--backend headless` (default): shells out to `claude -p`, billed against
    the operator's Claude subscription. Cost figures here are the CLI's
    *informational* estimate of API-equivalent price -- nothing is actually
    charged per call.
  - `--backend api`: direct `anthropic.messages.create()` calls plus a real
    MCP stdio session, driving its own multi-turn tool-use loop. Billed for
    real against `ANTHROPIC_API_KEY`. Cost here is computed from actual token
    usage x a pricing table in `run_eval.py` (verify against
    anthropic.com/pricing before treating as authoritative).

`run.py` (an early, single-hardcoded-question harness using mocked
`SimpleNamespace` responses) and the original `run_cli.py` (headless-only) have
both been retired in favor of this unified script.

## The `expected_calls` grading model

Evolved through several iterations once real runs showed the early designs
didn't hold up. Final shape, per question:

```python
"expected_calls": [
    {"name": "conditions", "calls": [{"tool": "get_current_conditions", "input": {...}}], "score": 10},
    {"name": "forecast", "calls": [{"tool": "get_daily_forecast", "input": {...}, "optional_params": {"days"}}], "score": 8},
]
```

- A **strategy** (`name`) can require one call or several (e.g. comparing two
  cities by calling `get_daily_forecast` twice, once per city).
- `optional_params` lets a model add a reasonable extra parameter (e.g.
  `hours=2` for "in an hour") without losing credit -- an earlier version did
  exact dict equality and incorrectly failed answers for using sensible
  optional params.
- `exclusive` marks strategies as redundant alternatives to each other (e.g.
  `compare_forecasts` vs. calling `get_daily_forecast` twice). If a model does
  both, only the higher-scoring one counts; every call belonging to the loser
  is treated as an extra, unexpected call and penalized the same as any other
  wasted call -- rather than a special-cased "forbidden combination" list.
- Any tool call that matches nothing in `expected_calls` is dinged a flat
  `UNEXPECTED_CALL_PENALTY` (2 points), and the total is clamped at 0.

This replaced an earlier "exact multiset of all calls must match one strategy"
design, which turned out to be matching on **call count** rather than call
*identity* -- coincidentally fragile, since unrelated combinations can happen
to total the same number of calls.

## LLM judge variance

Demonstrated directly, not just asserted: the exact same judge prompt (same
question, same tool data, same AI answer, same judge model, run standalone
outside the harness) scored **2** once and **9** on a second run. Averaging
`JUDGE_SAMPLES=3` (configurable via `--judge-samples`) smooths this somewhat
but does not eliminate it -- one batch run still produced `[2, 8, 2]` for an
answer that was clearly accurate. Single-sample LLM-judge scores should not be
trusted as precise; they're directional at best without much larger sample
sizes.

**Judges don't reliably follow "answer with only an integer," either.** A
16-question comparison run crashed on question 11 with
`ValueError: invalid literal for int() with base 10: '**7**'` -- the judge
wrapped its score in markdown bold despite the prompt explicitly saying
"**ONLY** answer with an integer between 0 and 10," and the harness's
`int(text.strip())` had zero tolerance for that. The rest of that 16-question
run (and its cost) was lost to one stray formatting choice. Fixed with
`parse_judge_score()`, which extracts the first integer via regex instead of
requiring an exact match -- covered by `test_run_eval.py`. Any harness that
parses a "just give me a number" LLM response needs to assume it won't always
get exactly that.

**`quality_score` is not reliable enough to stand alone on checkable
questions -- confirmed with a concrete example, not just judge-variance
noise.** Two of the new RAG-tool questions have an objectively correct
answer (a specific NWS office code; an ordered chain of office codes). On the
same run: a question where the model named the *wrong* office as a ridge's
origin (KMEG instead of the true origin, KFWD) was judged **6-8/10** --
plausible and well-formatted, and wrong. A question where the model got a
7-office chain **6-of-7 correct, in the right order** was judged **2-5/10**.
This isn't the same phenomenon as the judge-variance finding above (same
input, different scores across runs) -- these are two different questions,
each judged once per sample, landing on opposite sides of correct. Added
`grade_facts()` to `grading.py` and `expected_facts` to `questions.py` for
exactly these two questions -- a deterministic text check (does the answer
mention the right office code(s), in the right relative order) that scores
the KMEG answer `0` and the 6-of-7 chain answer `9`, catching both misses
`quality_score` alone did not. See `scripts/rag_test_notes.md`'s "Eval
harness run" section for the full RAG-specific writeup; this is the
methodology/harness side of the same finding.

## Harness bugs found while running the new RAG-tool questions

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

## RAG retrieval-recall finding (clean rerun, DB up, no leakage)

With items 5-7 above fixed and Postgres confirmed running, a full 16-question
`--question-set rag` run produced honest, non-leaked results for the first
time: avg `tool_score` 3.7, avg `quality_score` 4.2, $2.25 total cost (model +
judge x3) -- the real baseline going forward, not the earlier bug-inflated
figures.

The three `expected_facts` questions scored **0/10, 8/10, 4/10** this time
(previously 10/10/0 under the leaked run -- see item 6 above for why those
numbers were fake). The 0/10 is itself a genuine, reproducible retrieval
finding, not a harness bug: asked "which office first reported the ridge that
eventually brought heat to Raleigh," the model called
`search_forecast_history(query="ridge heat Raleigh")` at the default `top_k=5`
and got back only `KRAH` and `KOHX` chunks -- never `KFWD`, the true origin
office and the one geographically farthest from Raleigh. The model then
confidently answered "KOHX first reported it," which is wrong. This is a live
instance of exactly the failure mode `scripts/rag_test_notes.md`'s "Future
work: adaptive/multi-hop retrieval (v3)" section already predicted from raw
SQL testing: a query whose surface wording matches the *destination* office's
vocabulary out-competes the *origin* office's differently-worded but causally
correct chunks, and a narrow `top_k` never gives the origin a chance to
surface. See that file for the fuller writeup and the multi-query-expansion
angle already explored there.

## Real server bugs the eval process surfaced

Not eval-harness bugs -- actual defects in `weather_mcp` found while building
and running this harness against the live server:

1. **`get_point_data` regression**: pointed at `/active/{lat,lon}` instead of
   `/points/{lat,lon}`, silently breaking `get_daily_forecast`,
   `get_hourly_forecast`, `get_current_conditions`, and `get_weather_discussion`
   (everything that resolves a point via `get_point_data`) with a 404.
2. **`get_active_alerts` return-value bug**: built a formatted `alerts` list
   but returned the raw `features` list instead, crashing
   `"\n".join(...)` in `server.py` whenever an alert was actually active. The
   no-alerts case had the opposite problem -- returned a bare string, which
   `"\n".join()` then iterated character-by-character into garbage output.
3. **`get_weather_discussion` param bug**: `server.py` passed the raw
   `Coordinates` object where `nws.py` expected a resolved `grid_id` string,
   so the AFD lookup URL was always malformed and silently returned "no
   discussions found" for every request.
4. **`count` missing a type hint** on `get_weather_discussion(address, count=1)`
   caused the MCP-generated schema to advertise `count` as a `string`, which
   crashed `min(count, len(discussions))` the moment a model sent a numeric
   value as a string.
5. **Misleading/copy-pasted tool docstrings** -- `get_weather_discussion`'s
   description referenced "alerts" (leftover from `get_active_alerts`), and
   none of the six tool descriptions actually disambiguated from each other,
   directly contributing to inconsistent tool selection on ambiguous
   questions (see below).

## Tool-selection consistency findings

- **Ambiguous phrasing genuinely splits model behavior**, and this is a
  feature of the eval, not noise to suppress: "What is the weather in Kansas
  City, MO?" was answered with `get_current_conditions` in some runs and
  `get_daily_forecast` in others, both reasonable readings of an underspecified
  question. `expected_calls` models this directly by listing both as valid,
  differently-scored strategies rather than picking one "correct" answer.
- **The same question does not reliably get the same tool strategy across
  runs.** "Should I visit Asheville or Boone, NC next week for hiking?" picked
  the redundant two-call alternative (`tool_score: 4`) on one run, then picked
  the ideal `compare_forecasts` strategy (`tool_score: 10`) on a later run --
  identical question text, identical code, different outcome. "Greensboro vs.
  Charlotte, which has better weather" has so far picked `compare_forecasts`
  every time it's been run. This is exactly the kind of run-to-run variance a
  single spot-check would never catch, and it's the strongest argument in this
  whole project for running an eval suite repeatedly rather than trusting one
  pass/fail run.
- **Improving tool docstrings to explicitly disambiguate use cases** (e.g.
  telling `compare_forecasts` to say it already covers both locations, so a
  model shouldn't also call `get_daily_forecast`) was done in direct response
  to these findings, though a full before/after comparison run hasn't been
  done yet to quantify the effect.

## Quality findings independent of tool selection

The Boone, NC trip-planning question ("heading there in 5 hours, what should I
expect on arrival and over the weekend") scored a perfect `tool_score` --
correct multi-tool strategy, including bumping `days` above its default to
cover the whole weekend -- but the final answer had two real inaccuracies when
checked against the raw tool data:

- Claimed "mostly sunny" at the actual arrival time, when the hourly data
  still showed "mostly cloudy" for that hour (sunny conditions didn't arrive
  until several hours later).
- Mislabeled every day in the multi-day outlook, shifted about two days early
  (presented Sunday's data under a "Friday to Saturday" heading, etc.) --
  correct temperatures and conditions, wrong day-of-week arithmetic.

Both are model reasoning slips on top of entirely correct tool calls and
correct raw data -- exactly why `tool_score` and `quality_score` need to stay
separate metrics.

## Forecast horizon and a fabrication finding

`get_daily_forecast` was assumed to respect its `days` parameter arbitrarily
high, but checked directly: even requesting `days=14`, the real NWS API caps
at **14 periods (7 days)** regardless -- this is a hard limit of the
underlying `/gridpoints/.../forecast` product, not a bug in `nws.py`
(`forecast_count = min(days * 2, len(forecast_data))` correctly caps at
whatever NWS actually returns). From a Sunday, that reaches exactly through
"this weekend" and not one day further.

The first version of this test asked "Would it be better to go to Boone, NC
this weekend or next weekend?" -- added specifically because a human
immediately notices "next weekend" is 13 days out and the forecast probably
doesn't reach that far. It found real fabrication: the model split the one
weekend it had data for (Saturday, 6 days out) in half, relabeling
`Saturday`/`Saturday Night` as "Next Weekend, Aug 30-31" and `Tonight` (today)
as "This Weekend, Aug 24," manufacturing two distinct-looking weekends from a
single real data point. But it was run on a Sunday morning, when "this
weekend" is itself ambiguous even to a human (already happening? about to
happen?) -- so the finding was real but had a built-in confound.

The refined version removes that ambiguity: "Is it better to go to Boone, NC
this week or next week?" From a Sunday, "this week" maps exactly onto the
7-day window NWS actually returns; "next week" maps onto data that flatly does
not exist -- no interpretation required either way. Result: the model split
the single 7-day forecast into "This week (Aug 23-26)" and "Next week (Aug
27-30)" -- but Aug 27-29 are Thursday/Friday/Saturday of that *same* week, not
next week at all. There is zero data anywhere for the actual next week (Aug 30
onward), and the model never said so. This is a cleaner result than the first
version: not a date mislabeled by one day, but days 5-7 of a single forecast
window relabeled wholesale as a different week.

Both versions were re-fed to the judge standalone (same tool_data, same
answer, 3 samples each): **[2, 4, 2]** for the weekend version, **[3, 2, 4]**
for the week version -- consistently low in both cases, unlike the earlier
demonstrated judge-variance example. That's a meaningful data point in the
judge's favor: for a clear-cut fabrication, the judge's noise band narrows and
it converges on "bad" reliably, even though it's unreliable on more borderline
quality calls (see "LLM judge variance" above). `questions.py` keeps the
"this week or next week" phrasing as the canonical test since it isolates the
fabrication without the day-of-week confound.

## Edge-case coverage, and a grading-schema lesson

Three questions were added specifically to exercise error paths that had zero
eval coverage despite being fully implemented in `geocode.py`/`errors.py`:
an unrecognizable location ("Atlantis" -- `LocationNotFoundError`), an
ambiguous one ("Springfield," no state -- `LocationAmbiguousError`), and a
non-USA one ("London, UK" -- tests whether the "(USA locations only)"
docstring wording actually changes behavior).

The `expected_calls` written for these assumed a tool call was necessary in
all three cases, on the theory that the model can't know in advance that
Atlantis is fictional or that Springfield is ambiguous without trying. That
assumption was wrong two times out of three: for both Atlantis and
Springfield, the model used its own world knowledge and declined / asked for
clarification **without calling any tool at all** -- the same zero-call
pattern London was expected to produce. Since `expected_calls` for Atlantis
and Springfield initially had no zero-call option, this reasonable, arguably
*better* behavior (no wasted API call) would have scored a flat 0. Fixed by
adding a `"calls": []` "declined" strategy to all three, scored highest and
marked exclusive with the tool-call alternatives -- mirroring the pattern
already used for `compare_forecasts` vs. two separate calls, just at zero
calls instead of one. Verified directly against `grade_question` (not just by
inspection) that all three now score full credit for the observed no-call
behavior.

The methodology lesson generalizes: **don't assume a tool call is required
just because you can't picture the model succeeding without one.** Models
carry substantial world knowledge, and where that overlaps with what a tool
would tell you, the *better* answer is often not to call the tool at all. A
grading schema needs an explicit zero-call strategy anywhere that's plausible,
checked against real model behavior rather than assumed from the question
text alone -- exactly the same "run it first, then fix the schema" discipline
that shaped the `expected_calls` design throughout this project.

## Cost tracking

Every question and every judge sample records a cost figure, backend-appropriate:

- Headless: `total_cost_usd` from the CLI's own `stream-json` output --
  informational only, not an actual charge.
- API: computed from real `usage.input_tokens` / `usage.output_tokens` against
  a small pricing table in `run_eval.py` -- this backend spends real money.

The end-of-run summary reports total and per-question cost for the
model-under-test and the judge *separately* (so judge cost, held constant
across a model comparison, doesn't distort the number you actually care about),
plus a quality-points-per-dollar figure.

## Sample run (headless backend, Haiku under test, Sonnet judge, 1 judge sample)

A full run across all 12 questions, post-refactor, to confirm `run_eval.py`
works end to end:

```
Avg tool_score 10.2 / avg quality_score 8.2 at $0.0204/question
(33.6 quality points per $ spent on claude-haiku-4-5)
Model-under-test: $0.2453 total, $0.0204/question
Judge:            $0.5102 total, $0.0425/question
```

(Judge cost is higher than model-under-test cost here mainly because the judge
runs on Sonnet while the model-under-test runs on Haiku -- expected, and
exactly why the two costs are tracked separately.) This run also produced a
Boone answer with *four* tool calls (`get_current_conditions`,
`get_hourly_forecast`, `get_daily_forecast`, `get_active_alerts`) where only
the hourly and daily calls are in `expected_calls` -- scored `12 - 2*2 = 8`,
correctly crediting the two calls that mattered while docking the two that
weren't asked for.

## Model comparison: Haiku vs. Sonnet (full 16-question suite, 3 judge samples)

Two full runs, each covering all 16 questions:

| | Haiku under test, Sonnet judge | Sonnet under test, Opus judge |
|---|---|---|
| Avg `tool_score` | 9.5 | 10.1 |
| Avg `quality_score` | 7.0 | 8.6 |
| Model-under-test cost | $0.3356 total ($0.0210/question) | $0.7108 total ($0.0444/question) |
| Judge cost | $1.0782 total ($0.0674/question) | $1.9053 total ($0.1191/question) |
| Quality points per $ | **20.9** | 12.0 |

**Caveat that matters more than the numbers**: this design varies the
model-under-test *and* the judge simultaneously, so an observed
`quality_score` gap can't be cleanly attributed to "Sonnet answers better" --
some of it could be "Opus judges more generously than Sonnet," or some mix of
both. A rigorous follow-up would hold the judge constant (e.g., Sonnet judging
both Haiku's and Sonnet's answers) to isolate the model-under-test effect. Not
done here; noted so the numbers above aren't overclaimed.

With that caveat in place, the headline finding directly answers the
cost-vs-quality framing question this project started with: Sonnet costs
~2.1x more per question than Haiku here, but the Haiku config is still ~1.7x
more cost-efficient per quality point (20.9 vs. 12.0). More expensive did not
mean proportionally better.

Both models correctly declined all three no-call edge cases (Atlantis,
Springfield, London) in both runs -- consistent behavior on that dimension
regardless of model size. Haiku's judge scores were noticeably noisier in this
particular run than in earlier single-config runs (e.g. `[3, 9, 10]` for the
Kansas City question, `[4, 7, 3]` for the zip-code question) -- more evidence
for the already-documented judge-variance finding, not a new phenomenon, but a
wider spread than usual.

This run also caught a real harness bug along the way: the first attempt
crashed partway through on a judge response of `**7**` (markdown-wrapped)
instead of a bare integer -- see "LLM judge variance" above for the fix.

## Headless vs. direct API: a real comparison

The `api` backend was run live for the first time against all 16 questions
(Haiku under test, Sonnet judging, 3 samples each) -- same models as an
earlier headless run, letting the two be compared directly:

| | Headless (informational) | Direct API (real, billed) |
|---|---|---|
| Model-under-test cost/question | $0.0210 | **$0.0043** |
| Avg `tool_score` | 9.5 | 10.4 |
| Avg `quality_score` | 7.0 | 7.9 |

The real API cost came in at roughly **1/5th** of the headless backend's own
cost estimate for equivalent work. This isn't the headless backend being
"wrong" about its own cost -- it's honestly pricing what it actually does,
which includes a full Claude Code agent harness (system prompt, the
`ToolSearch` deferred-tool-loading round-trip) that a minimal, purpose-built
API integration doesn't carry. The practical lesson: headless cost figures
describe the cost of running this eval *through Claude Code*, not the cost of
running this tool-selection task through the API in general -- don't treat
one as a proxy for the other.

The zero-call "declined" strategy (Atlantis/Springfield/London) produced
identical behavior on both backends -- `tool_calls: []`, `tool_score: 10` in
both cases -- good confirmation that the grading design isn't an artifact of
one particular way of invoking the model. One transient failure occurred on
the first live API attempt (a judge response with no parseable score,
`ValueError: Judge response contained no integer: ''`) that didn't reproduce
on retry; `judge_api` now wraps parse failures with `stop_reason` and content
block types for diagnosis if it recurs (see "LLM judge variance" above for
the similar markdown-wrapping fix this mirrors).

## Open items

- The `PRICING_PER_MTOK_USD` table in `run_eval.py` is a manually maintained
  snapshot and should be checked against current published pricing before any
  cost comparison is treated as final.
- No test question yet specifically re-verifies the tool-docstring
  disambiguation fixes (e.g., re-running Kansas City / Asheville-Boone after
  the docstring changes to see if consistency measurably improved).
- **Idea, not yet scoped or built**: does in-context feedback change
  tool-selection behavior within a session (i.e. does telling the model its
  score, or better, *why* it scored that way, change what it does on a later,
  similar question)? This is not the same thing as this harness measures --
  it's in-context adaptation, not the model's underlying capability, and
  testing it would require a deliberately separate, controlled experiment
  (repeated near-identical questions in one long session with rich feedback
  injected, a no-feedback control session of equal length, and enough
  repetitions to clear the judge-noise floor already documented above) rather
  than a mode added to the existing 16-question suite. Noted for later, not
  currently planned.
