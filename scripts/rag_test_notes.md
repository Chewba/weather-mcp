# RAG test case notes

Working notes on retrieval behavior against the `scripts/test_data.json` corpus
(110 AFDs, 11 offices, ~2026-08-29 through 2026-08-31). Each case: what was
queried, what ground truth exists in the raw text, and what `test_retrieval.py`
actually returned. Add new cases below as they're found.

## Case: heat ridge tracking (KFWD -> KRAH)

**Ground truth** (found via `chunk_text ILIKE '%heat%'` scan, not retrieval):
a subtropical ridge builds over KFWD (Dallas/Fort Worth) starting ~Aug 29 and
shifts east; KRAH (Raleigh) independently narrates "the ~595 dam mid-level
anticyclone will build eastward" the same day, with heat peaking Aug 31
("Peak of the heat will likely be Tuesday through Thursday"). Physically
consistent west-to-east ridge propagation over the 3-day window.

**Query:** `"a ridge of high pressure aloft building and shifting eastward
bringing hot temperatures"`

**Result:** Sorting the top-20 nearest-neighbor hits by `issued_at` reproduced
the correct chronology and geography for the real signal (KFWD -> KRAH,
Aug 29 -> Aug 31, distances 0.50-0.56). This is the good example for a writeup:
proof that timestamp + vector search can reconstruct a pattern's movement
without any manual filtering.

**Caveat:** same query also pulled in KSEW satellite/cloud-cover boilerplate
and KMPX "low pressure system moving" chunks at nearly identical distance
(0.53-0.56). `all-MiniLM-L6-v2` (384-dim, general purpose) doesn't cleanly
separate "ridge" from "trough/low" -- it's matching on "meteorologist
describing an upper-air feature's location," not the sign of the feature.
Don't present this query's raw top-20 as clean; the *sorted, office-labeled*
KFWD/KRAH subset is the real result. For a writeup, worth explicitly showing
the noise as a limitation, not hiding it.

**Implication:** distance-only retrieval is convincing for a demo but not
precise enough on its own for anything claiming correctness (e.g. "only
ridge-related chunks"). Would need one of: a bigger/domain-tuned embedding
model, a keyword pre-filter (e.g. `chunk_text ILIKE`) narrowing before vector
rerank, or a reranking pass.

## Case: office-scoped topical search

**Query:** `"chance of severe thunderstorms and heat advisories"`

**Result:** clean -- top 5 all landed on KLIX (New Orleans) `KEY_MESSAGES`
chunks, tight distance cluster (0.336-0.365), clear separation from the rest
of the corpus, and surfaced 5 distinct issuance times for the same office/
topic. Good example of retrieval working as expected when the query maps to
one dominant office/topic rather than a cross-country pattern.

## Case: multi-hop causality question ("where did it start")

**Status:** complete. Corpus now includes 5 intermediate offices along the
stated path (KSHV, KMEG, KOHX, KFFC, KCAE), 16 offices / 160 products total.

**Query:** `"Where did the heat wave that is hitting Greensboro start?"`

**Ground truth check first:** "Greensboro" never appears anywhere in the
corpus as a word; the station code `KGSO` appears exactly once, buried in a
KRAH record-highs list. So there's no keyword shortcut available -- any
correct answer has to come from semantics, not string matching.

**Result:** all top 6 nearest-neighbor hits landed on KRAH, correctly, purely
on semantic grounds (heat/humidity/central-NC language) -- a genuine win,
since it means the retrieval can identify "which office actually covers this
place/topic" without any city-to-office lookup table.

**But it did not answer the question.** None of the KFWD ridge-origin chunks
(the actual "where it started" answer) showed up in the top 6 -- the query's
similarity to "heat is happening in central NC" dominated and the search
never reached back to the upstream cause. A single nearest-neighbor query
against one embedding cannot chain "RAH is hot because a ridge built over
KFWD" -- that fact only exists by combining two offices' text, and vector
search alone doesn't do that combination step.

**Why this only has 2 data points right now:** the corpus currently has the
two endpoints of the ridge (KFWD origin, KRAH destination) and nothing
between them, so there's no intermediate evidence a causality-tracing answer
could walk through -- an LLM given both endpoints can only assert a straight
line from Texas to North Carolina, not show the ridge actually crossing the
country. RAH's own discussion text names the missing middle: the ridge
becomes "nearly stationary over the TN Valley and southern Mid-Atlantic."
Based on that, backfilling 5 offices along the stated path:

- Shreveport, LA (SHV) -- immediate hop out of north Texas
- Memphis, TN (MEG) and Nashville, TN (OHX) -- the named "TN Valley"
- Atlanta, GA (FFC) -- Deep South leg
- Columbia, SC (CAE) -- named "southern Mid-Atlantic," last hop before NC

**Ground truth after backfill:** excellent -- this is the best test case in
the corpus. All 5 new offices independently narrate the *same* ridge as a
physically moving feature, with no cross-reference to each other:
KSHV ("upper ridge moves into the MidSouth" Aug 30 -> "migrates east across
the Mid-Miss[issippi]" Aug 31) -> KMEG ("ridge is centered over the Southern
Plains" Aug 30 AM -> "centered over the Southern Plains and Lower [MS Valley]"
Aug 31 AM -> "remains centered over the Ozarks" Aug 31 PM, i.e. the stated
ridge-center shifts within one office's own updates) -> KOHX ("ridge
continues to build" -> "ridge will start to break down" Aug 31 PM) -> KFFC
("ridge will continue to build eastward" -> "expansive upper ridge over the
south-central U.S.") -> KCAE ("anomalously strong upper ridge" as early as
Aug 29). 161 chunks total match `ridge` or `heat` across these 5 offices --
a real, traceable west-to-east chain now exists in the data, not just two
disconnected endpoints.

**Re-ran the same query against the fuller corpus.** Result: still dominated
by KRAH -- 12 of the top 15 hits are KRAH chunks, and only one intermediate
hop (KMEG's "ridge remains centered over the Ozarks") cracks the top 15, at
rank 13. None of KSHV/KOHX/KFFC/KCAE appear in the top 15 despite being
genuinely on the path and now present in the corpus.

**Working conclusion:** having the intermediate evidence in the corpus was
necessary but not sufficient. A single global top-k query still won't
assemble the chain on its own -- it's dominated by whichever office's
language most directly matches the query's surface wording (here, KRAH's
"heat" vocabulary beats KMEG/KFFC's "ridge" vocabulary even though the ridge
chunks are the actual causal answer). For a causality question like this to
work reliably, retrieval needs to either (a) diversify by office (e.g. top-N
per office instead of one global top-k) so every office on the path gets a
chance to surface, or (b) run explicitly as multiple queries/hops -- first
identify the relevant office(s), then separately query for "upstream
cause/origin" language -- and hand the LLM the union of all of it. Retrieval
narrows candidates; it does not chain causality on a single pass, even when
the chain is fully present in the data.

## Future work: adaptive/multi-hop retrieval (v3)

The heat-ridge case above is the concrete evidence for this: single-pass
top-k over one query embedding is structurally the wrong tool for a
causality question, no matter how good the corpus is (see 161 ground-truth
ridge/heat chunks vs. only 1 surfacing in the top 15). The fix isn't more
data, it's an adaptive retrieval step -- something that, given a question
shaped like "why/where did X start," runs a first pass to identify the
relevant office(s) and time window, then issues a second, differently-worded
query (e.g. swap "heat" for "ridge"/"upstream cause") or diversifies by
office before handing chunks to the LLM. This dataset (16 offices, a real
traceable west-to-east chain, and a documented single-pass failure mode) is
a good fixed benchmark for that v3 work -- rerun this same Greensboro query
against whatever adaptive step gets built and check whether KSHV/KMEG/KOHX/
KFFC/KCAE start showing up alongside KRAH.

## Open questions for future test cases

- Does retrieval correctly distinguish two offices with genuinely similar
  weather (e.g. two Gulf Coast offices both discussing tropical moisture)
  rather than conflating them?
- What does a *negative* test look like -- a query with no good match in the
  corpus (e.g. "snow accumulation") -- does distance blow out appropriately,
  or does it still confidently return top-5 garbage?
- Chunk-type filtering (`WHERE chunk_type = 'DISCUSSION'`) combined with
  vector search hasn't been tried yet -- likely reduces the ridge/trough
  confusion above since `OTHER`/aviation-style chunks carry a lot of the noise.
