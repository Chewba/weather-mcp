"""Eval question set for the RAG retrieval tools (search_forecast_history,
explain_forecast_reasoning). Referenced by run_eval.py via
--question-set rag|both. Split out from questions_mcp.py so a run scoped to
just one tool family does not re-spend usage re-testing the other -- these
questions also depend on RAG corpus state (see --corpus-mode), which the
original MCP questions never did."""

RAG_QUESTIONS = [
    {
        # Mirrors the "heat ridge tracking (KFWD -> KRAH)" case in
        # scripts/rag_test_notes.md. Ground truth (found by scanning raw chunk
        # text, not retrieval): a ridge builds over KFWD ~Aug 29 and shifts
        # east, arriving as heat over KRAH by Aug 31 -- a real, traceable
        # cross-office pattern. search_forecast_history has no recency
        # weighting, so reconstructing the chronology requires the model
        # itself to notice each passage's issued_at -- this is really a
        # quality_score test: the documented result was a correct KFWD->KRAH
        # reconstruction alongside KSEW/KMPX noise at similar distance that a
        # good answer shouldn't present as equally reliable.
        "question": "Show me how the high-pressure ridge that brought hot temperatures moved across the country recently.",
        "expected_calls": [
            {
                "name": "broad_search",
                "calls": [
                    {
                        "tool": "search_forecast_history",
                        "input": {},
                        "optional_params": {"query", "top_k", "office_id"},
                    }
                ],
                "score": 10,
            },
        ],
    },
    {
        # The "multi-hop causality" case from rag_test_notes.md, kept verbatim.
        # Ground truth: "Greensboro" never appears as a word anywhere in the
        # corpus, and its station code (KGSO) appears exactly once, buried in
        # a KRAH record-highs list -- so a correct office match has to come
        # from semantics ("central NC heat/humidity"), not a keyword, and
        # there's no tool in this project that resolves a place name to an
        # NWS office code, so office_id="KRAH" here also depends on the model
        # already knowing that mapping. Documented result: even scoped
        # correctly to KRAH, the ridge's true origin (KFWD, plus the 5
        # backfilled intermediate offices) never surfaces in the top hits --
        # KRAH's own "heat" vocabulary dominates over KFWD/KMEG's "ridge"
        # vocabulary. Expect a good tool_score (this is exactly what
        # explain_forecast_reasoning was built for) but an incomplete
        # quality_score -- that gap *is* the finding, not an eval bug.
        "question": "Where did the heat wave that is hitting Greensboro, NC start?",
        "expected_calls": [
            {
                "name": "explain_krah",
                "calls": [
                    {
                        "tool": "explain_forecast_reasoning",
                        "input": {"office_id": "KRAH"},
                        "optional_params": {"query", "top_k"},
                    }
                ],
                "score": 10,
                "exclusive": "broad_search",
            },
            {
                "name": "broad_search",
                "calls": [
                    {
                        "tool": "search_forecast_history",
                        "input": {},
                        "optional_params": {"query", "top_k", "office_id"},
                    }
                ],
                "score": 6,
                "exclusive": "explain_krah",
            },
        ],
    },
    {
        # The "office-scoped topical search" case from rag_test_notes.md.
        # Ground truth: querying for severe thunderstorms + heat advisories
        # landed cleanly on KLIX (New Orleans) KEY_MESSAGES chunks, a tight
        # distance cluster (0.336-0.365) clearly separated from the rest of
        # the corpus -- the cleanest retrieval result found so far, and it
        # worked even *without* an office filter in the original
        # test_retrieval.py run. Scoring both an explicit office_id and an
        # unscoped call highly for that reason -- unscoped is not a lesser
        # strategy on this particular question.
        "question": "What has the New Orleans forecast office said recently about severe thunderstorms and heat advisories?",
        "expected_calls": [
            {
                "name": "scoped_search",
                "calls": [
                    {
                        "tool": "search_forecast_history",
                        "input": {"office_id": "KLIX"},
                        "optional_params": {"query", "top_k"},
                    }
                ],
                "score": 10,
                "exclusive": "unscoped_search",
            },
            {
                "name": "unscoped_search",
                "calls": [
                    {
                        "tool": "search_forecast_history",
                        "input": {},
                        "optional_params": {"query", "top_k", "office_id"},
                    }
                ],
                "score": 9,
                "exclusive": "scoped_search",
            },
        ],
    },
    {
        # Ground truth checked directly: chunk_text ILIKE '%mountain%' returns
        # zero hits for KRAH (mentions exist at KFFC, KSHV, KSEW -- just not
        # here). This is a sharper fabrication test than a plain out-of-domain
        # question: "do the mountains affect Raleigh's weather" *sounds*
        # meteorologically reasonable (the Appalachians genuinely do
        # influence NC weather in general), so a model is more tempted to
        # answer confidently from outside world knowledge than it would be
        # for an obviously absurd query. The tool call itself should score
        # well regardless of which RAG tool is used; the real test is
        # quality_score -- does the answer say the discussion doesn't address
        # this, or does it fabricate a plausible-sounding grounded claim from
        # zero retrieved evidence? Neither RAG tool's formatted output
        # includes a distance/confidence number (server.py only prints
        # office/type/date/text), so a low score here means the model failed
        # to notice the *content* was irrelevant, not that it ignored a
        # number it had access to.
        "question": "Do the mountains affect the weather in KRAH?",
        "expected_calls": [
            {
                "name": "explain_krah",
                "calls": [
                    {
                        "tool": "explain_forecast_reasoning",
                        "input": {"office_id": "KRAH"},
                        "optional_params": {"query", "top_k"},
                    }
                ],
                "score": 10,
                "exclusive": "search_krah",
            },
            {
                "name": "search_krah",
                "calls": [
                    {
                        "tool": "search_forecast_history",
                        "input": {"office_id": "KRAH"},
                        "optional_params": {"query", "top_k"},
                    }
                ],
                "score": 10,
                "exclusive": "explain_krah",
            },
        ],
    },
    {
        # True negative control, per the "Open questions" section of
        # rag_test_notes.md. Ground truth checked directly: chunk_text ILIKE
        # '%snow%' returns exactly one incidental hit in the entire 160-record
        # corpus (KSEW, not even really about snow) -- this is an August
        # corpus, there is no real snow content anywhere in it. Tests whether
        # the tool call and the final answer both handle "no good match
        # exists" honestly (e.g. "no discussions mention snow") rather than
        # confidently returning the nearest-available chunks as if they were
        # relevant. No tool_score strategy scores highly here beyond making a
        # reasonable attempt -- this question's entire value is quality_score.
        "question": "Is snow expected anywhere in the country right now?",
        "expected_calls": [
            {
                "name": "broad_search",
                "calls": [
                    {
                        "tool": "search_forecast_history",
                        "input": {},
                        "optional_params": {"query", "top_k", "office_id"},
                    }
                ],
                "score": 10,
            },
        ],
    },
    {
        # The "does retrieval distinguish two genuinely similar offices"
        # open question from rag_test_notes.md. Ground truth checked
        # directly: both KLIX (New Orleans, 27 chunks) and KMFL (Miami, 14
        # chunks) discuss tropical moisture heavily, but describe *opposite*
        # trends -- KLIX trending drier ("what looked like a much drier and
        # hotter setup... has been t[rending]"), KMFL trending wetter ("a
        # rather wet pattern prevailing across SoFlo"). A real confusable
        # pair, not a contrived one: scoring this well requires the answer to
        # keep the two offices' actual (differing) trends straight rather
        # than blending them into one generic "Gulf Coast tropical moisture"
        # answer. Office codes aren't derivable from the city names here the
        # way KRAH~Raleigh is, so this also exercises the same
        # office-resolution gap as the Greensboro question above.
        "question": "Compare how the New Orleans and Miami forecast offices are each describing tropical moisture trends right now.",
        "expected_calls": [
            {
                "name": "scoped_both",
                "calls": [
                    {
                        "tool": "search_forecast_history",
                        "input": {"office_id": "KLIX"},
                        "optional_params": {"query", "top_k"},
                    },
                    {
                        "tool": "search_forecast_history",
                        "input": {"office_id": "KMFL"},
                        "optional_params": {"query", "top_k"},
                    },
                ],
                "score": 10,
                "exclusive": "broad_search",
            },
            {
                "name": "broad_search",
                "calls": [
                    {
                        "tool": "search_forecast_history",
                        "input": {},
                        "optional_params": {"query", "top_k", "office_id"},
                    }
                ],
                "score": 5,
                "exclusive": "scoped_both",
            },
        ],
    },
    {
        # Ground truth checked directly against the live DB: KRAH's chunks
        # currently span 2026-08-29 18:00 -> 2026-08-31 23:42 (about 2.5
        # days), and the whole corpus tops out at 2026-09-01 00:11 -- there is
        # no "last week" in here at all, only a few days of one week. Same
        # fabrication shape as the "this week or next week" Boone question
        # and the "historically hot" Greensboro question, just pointed
        # backward instead of forward: does the answer admit it only has a
        # few days of history to compare, or does it invent two distinct
        # weekly summaries out of the same short window? The office code is
        # given directly in the question this time ("KRAH"), sidestepping the
        # address->office_id resolution gap on purpose, to isolate the
        # data-scope honesty question from that other one. explain_forecast_
        # reasoning is the better-fitting tool here (not just an acceptable
        # alternative, unlike the Greensboro causal-origin question) --
        # "how did this change over time" is exactly what its recency
        # ordering was built for, unlike search_forecast_history's
        # undated, distance-only ordering.
        "question": "Compare the weather in Greensboro, NC KRAH from last week to this week.",
        "expected_calls": [
            {
                "name": "explain_krah",
                "calls": [
                    {
                        "tool": "explain_forecast_reasoning",
                        "input": {"office_id": "KRAH"},
                        "optional_params": {"query", "top_k"},
                    }
                ],
                "score": 10,
                "exclusive": "search_krah",
            },
            {
                "name": "search_krah",
                "calls": [
                    {
                        "tool": "search_forecast_history",
                        "input": {"office_id": "KRAH"},
                        "optional_params": {"query", "top_k"},
                    }
                ],
                "score": 7,
                "exclusive": "explain_krah",
            },
        ],
    },
    {
        # Deliberate positive-case pair to the KRAH mountains question above.
        # Ground truth checked directly: KFFC's KEY_MESSAGES chunk (its most
        # prominent section, appears at high confidence in retrieval) says
        # verbatim: "Isolated thunderstorms will be possible in east-central
        # Georgia and in the mountains this afternoon." Real, specific,
        # retrievable grounding -- unlike KRAH, which has zero mountain
        # mentions. Together the two questions test both failure directions:
        # does the model correctly say "no" when there's nothing to ground on
        # (KRAH), and does it correctly find and cite the real content when
        # it exists (here)? A good answer here should reference the isolated
        # mountain thunderstorm risk specifically, not just generically
        # affirm "yes, mountains affect weather" from outside knowledge --
        # that would still be an ungrounded answer even though it happens to
        # be true, same concern as the KRAH case.
        "question": "Do the mountains affect the weather in KFFC?",
        "expected_calls": [
            {
                "name": "explain_kffc",
                "calls": [
                    {
                        "tool": "explain_forecast_reasoning",
                        "input": {"office_id": "KFFC"},
                        "optional_params": {"query", "top_k"},
                    }
                ],
                "score": 10,
                "exclusive": "search_kffc",
            },
            {
                "name": "search_kffc",
                "calls": [
                    {
                        "tool": "search_forecast_history",
                        "input": {"office_id": "KFFC"},
                        "optional_params": {"query", "top_k"},
                    }
                ],
                "score": 10,
                "exclusive": "explain_kffc",
            },
        ],
    },
    {
        # v3 candidate: reuses the same real KFWD->KRAH ridge relationship
        # that anchors the multi-hop causality benchmark above, but reframes
        # it as evaluative synthesis ("better") instead of pure
        # origin-tracing. KFWD (Dallas) is the ridge's documented origin
        # (~Aug 29); KRAH (covers Greensboro) gets the same heat arriving by
        # Aug 31, peaking Tue-Thu per its own discussion text. A shallow
        # single-pass answer risks averaging both offices' "hot" language
        # into one generic answer instead of noticing Dallas's heat was
        # already established while Raleigh's was still building -- a timing
        # nuance across two offices' data that neither office's chunks state
        # on their own. This also stacks the address->office_id resolution
        # gap on top (two place names, no raw codes given) -- a low score
        # here could be either problem, so don't attribute a miss to
        # retrieval alone without checking which office_id the model
        # actually used.
        #
        # Real run finding: the model made zero tool calls and instead
        # honestly explained that neither RAG tool provides actual historical
        # observations (only forecast discussion text), offering to attempt a
        # partial search if wanted. That's arguably the best possible
        # response to a genuinely ambiguous data question -- the same lesson
        # already documented for Atlantis/Springfield/London (a zero-call
        # decline can be the *better* outcome, not a failure to attempt
        # something) -- so it's scored on par with the grounded-attempt
        # strategies, not as a miss.
        "question": "Was the weather in Dallas, TX better than Greensboro, NC at the end of August?",
        "expected_calls": [
            {
                "name": "declined",
                "calls": [],
                "score": 6,
                "exclusive": ["explain_both", "search_both", "broad_search"],
            },
            {
                "name": "explain_both",
                "calls": [
                    {
                        "tool": "explain_forecast_reasoning",
                        "input": {"office_id": "KFWD"},
                        "optional_params": {"query", "top_k"},
                    },
                    {
                        "tool": "explain_forecast_reasoning",
                        "input": {"office_id": "KRAH"},
                        "optional_params": {"query", "top_k"},
                    },
                ],
                "score": 10,
                "exclusive": ["declined", "search_both", "broad_search"],
            },
            {
                "name": "search_both",
                "calls": [
                    {
                        "tool": "search_forecast_history",
                        "input": {"office_id": "KFWD"},
                        "optional_params": {"query", "top_k"},
                    },
                    {
                        "tool": "search_forecast_history",
                        "input": {"office_id": "KRAH"},
                        "optional_params": {"query", "top_k"},
                    },
                ],
                "score": 9,
                "exclusive": ["declined", "explain_both", "broad_search"],
            },
            {
                "name": "broad_search",
                "calls": [
                    {
                        "tool": "search_forecast_history",
                        "input": {},
                        "optional_params": {"query", "top_k", "office_id"},
                    }
                ],
                "score": 4,
                "exclusive": ["declined", "explain_both", "search_both"],
            },
        ],
    },
    {
        # v3 candidate #2: a sharper, more checkably-gradable version of the
        # multi-hop causality question above. That question asks for an open
        # origin narrative; this asks for a specific fact -- a named office
        # -- checked directly against the final answer text via
        # grade_facts/expected_facts below, rather than relying purely on the
        # subjective LLM judge. Real run finding, and exactly why this needed
        # a fact check: the model named KMEG (a real intermediate hop) as the
        # origin instead of the true origin, KFWD, but the judge still scored
        # the answer 6-8/10 -- plausible, coherent, and *wrong*. expected_facts
        # catches that; quality_score alone did not. This is the intended
        # regression benchmark for v3: re-run it once adaptive/multi-hop
        # retrieval exists and check whether fact_score actually improves.
        "question": "Which NWS office first reported the ridge that eventually brought heat to Raleigh, and around when did it start?",
        "expected_calls": [
            {
                "name": "explain_krah",
                "calls": [
                    {
                        "tool": "explain_forecast_reasoning",
                        "input": {"office_id": "KRAH"},
                        "optional_params": {"query", "top_k"},
                    }
                ],
                "score": 10,
                "exclusive": "broad_search",
            },
            {
                "name": "broad_search",
                "calls": [
                    {
                        "tool": "search_forecast_history",
                        "input": {},
                        "optional_params": {"query", "top_k", "office_id"},
                    }
                ],
                "score": 6,
                "exclusive": "explain_krah",
            },
        ],
        "expected_facts": {"must_mention": {"KFWD": 10}},
    },
    {
        # v3 candidate #3: promoted directly from the "Future work" section
        # of rag_test_notes.md, which specified this exact re-run as the
        # benchmark for adaptive/multi-hop retrieval -- previously only a
        # prose note, not an actual eval question. Ground truth (backfilled
        # and verified in that session): the ridge is independently narrated
        # by KFWD (origin) -> KSHV -> KMEG -> KOHX -> KFFC -> KCAE -> KRAH
        # (destination), 161 ground-truth ridge/heat chunks across the chain.
        # Documented result for a single top-k pass: dominated by KRAH, only
        # one intermediate hop (KMEG) cracked the top 15, none of
        # KSHV/KOHX/KFFC/KCAE appeared despite being genuinely on the path
        # and present in the corpus -- necessary but not sufficient. No
        # office_id can be given (the whole point is discovering the path
        # across offices), so today's only available strategy is a single
        # broad search; scoring it for making the attempt, since actually
        # naming most/all 7 offices in order is a v3-only bar, not a v2 one.
        #
        # Real run finding: the original phrasing ("this ridge") relied on
        # implicit context from rag_test_notes.md that the model never sees
        # -- each eval question fires as an isolated, context-free prompt, so
        # "this ridge" had no antecedent. The model correctly asked for
        # clarification and the judge correctly scored that highly (9) --
        # that was a real bug in the question, not a retrieval or model
        # failure. Reworded to be self-contained.
        #
        # Second real run finding, after the reword, and the reason this now
        # has expected_facts: the model actually got remarkably close --
        # KFWD -> KSHV -> KMEG -> KOHX -> KFFC -> KRAH, 6 of 7 offices in the
        # correct order (only missing KCAE) -- by making 3 differently-worded
        # search_forecast_history calls with a self-selected top_k of 20-30.
        # That's the model spontaneously doing the "multi-query expansion"
        # v3 strategy described above, without any adaptive-retrieval code
        # existing yet. The judge still scored this 2-5/10 -- a checkably
        # strong answer scored poorly, the mirror image of the origin
        # question's checkably wrong answer scored well. grade_facts catches
        # both; quality_score alone caught neither.
        "question": "A ridge of high pressure brought unusually hot temperatures as it moved from Texas toward North Carolina in late August 2026. List, in order, every NWS office that mentioned this ridge.",
        "expected_calls": [
            {
                "name": "broad_search",
                "calls": [
                    {
                        "tool": "search_forecast_history",
                        "input": {},
                        "optional_params": {"query", "top_k", "office_id"},
                    }
                ],
                "score": 10,
            },
        ],
        "expected_facts": {
            "ordered_sequence": ["KFWD", "KSHV", "KMEG", "KOHX", "KFFC", "KCAE", "KRAH"],
            "sequence_item_points": 1,
            "order_bonus": 3,
        },
    },
    {
        # Lower-priority than the two above -- not a v3 differentiator, but
        # worth having as a plain v2 sanity check. Unlike the Dallas/
        # Greensboro question, KRAH and KLIX aren't causally linked in the
        # corpus (Raleigh heat and New Orleans thunderstorms are independent
        # events), and both office codes are given directly, so two
        # independent scoped calls should fully answer this today -- no
        # adaptive step needed. Same shape as the existing KLIX/KMFL
        # confusable-pair question, just testing plain compare-two-offices
        # behavior rather than trend-distinguishing.
        "question": "Compare the weather in KRAH vs KLIX for the end of August.",
        "expected_calls": [
            {
                "name": "explain_both",
                "calls": [
                    {
                        "tool": "explain_forecast_reasoning",
                        "input": {"office_id": "KRAH"},
                        "optional_params": {"query", "top_k"},
                    },
                    {
                        "tool": "explain_forecast_reasoning",
                        "input": {"office_id": "KLIX"},
                        "optional_params": {"query", "top_k"},
                    },
                ],
                "score": 10,
                "exclusive": ["search_both", "broad_search"],
            },
            {
                "name": "search_both",
                "calls": [
                    {
                        "tool": "search_forecast_history",
                        "input": {"office_id": "KRAH"},
                        "optional_params": {"query", "top_k"},
                    },
                    {
                        "tool": "search_forecast_history",
                        "input": {"office_id": "KLIX"},
                        "optional_params": {"query", "top_k"},
                    },
                ],
                "score": 9,
                "exclusive": ["explain_both", "broad_search"],
            },
            {
                "name": "broad_search",
                "calls": [
                    {
                        "tool": "search_forecast_history",
                        "input": {},
                        "optional_params": {"query", "top_k", "office_id"},
                    }
                ],
                "score": 4,
                "exclusive": ["explain_both", "search_both"],
            },
        ],
    },
    {
        # Every prior question where the model skipped office_id happened to
        # still work out fine, because the offices involved had
        # substantively different content (KFWD ridge-building vs KRAH heat
        # arrival, KLIX drying vs KMFL wetting). This question is designed so
        # skipping office_id actually costs something. Ground truth checked
        # directly: KCAE and KFFC are adjacent hops in the same documented
        # ridge chain and use nearly interchangeable heat language --
        # KCAE: "well above normal temperatures with dangerous heat index
        # values possible next week"; KFFC: "heat will build through the work
        # week with triple digit heat indices returning." An unscoped search
        # risks silently returning KFFC's content as if it were Columbia's.
        # "Columbia, SC" isn't derivable to "KCAE" by spelling either, so
        # this also carries the usual office-resolution-gap risk on top --
        # a wrong answer here could be either problem, check which.
        "question": "What is the Columbia, SC forecast office saying about the heat this week?",
        "expected_calls": [
            {
                "name": "explain_kcae",
                "calls": [
                    {
                        "tool": "explain_forecast_reasoning",
                        "input": {"office_id": "KCAE"},
                        "optional_params": {"query", "top_k"},
                    }
                ],
                "score": 10,
                "exclusive": ["search_kcae", "broad_search"],
            },
            {
                "name": "search_kcae",
                "calls": [
                    {
                        "tool": "search_forecast_history",
                        "input": {"office_id": "KCAE"},
                        "optional_params": {"query", "top_k"},
                    }
                ],
                "score": 9,
                "exclusive": ["explain_kcae", "broad_search"],
            },
            {
                "name": "broad_search",
                "calls": [
                    {
                        "tool": "search_forecast_history",
                        "input": {},
                        "optional_params": {"query", "top_k", "office_id"},
                    }
                ],
                "score": 4,
                "exclusive": ["explain_kcae", "search_kcae"],
            },
        ],
    },
    {
        # Repeat of the full-chain question with different wording (no
        # anaphoric "this ridge" this time -- self-contained from the start),
        # to check whether the multi-query/wide-top_k behavior documented in
        # rag_test_notes.md (3 differently-worded searches, self-selected
        # top_k 20-30, got 6 of 7 offices correct in order) was a repeatable
        # strategy or a one-off. Same ground truth chain, same fact check.
        "question": "Trace, in chronological order, which National Weather Service offices reported an upper-level ridge of high pressure building over the southern Plains and shifting eastward across the South and Mid-Atlantic in late August 2026.",
        "expected_calls": [
            {
                "name": "broad_search",
                "calls": [
                    {
                        "tool": "search_forecast_history",
                        "input": {},
                        "optional_params": {"query", "top_k", "office_id"},
                    }
                ],
                "score": 10,
            },
        ],
        "expected_facts": {
            "ordered_sequence": ["KFWD", "KSHV", "KMEG", "KOHX", "KFFC", "KCAE", "KRAH"],
            "sequence_item_points": 1,
            "order_bonus": 3,
        },
    },
    {
        # Isolates explain_forecast_reasoning's recency ordering specifically
        # -- distinct from the existing "last week vs this week" question,
        # which is a deliberate data-scope trap (no real prior week exists).
        # This one is genuinely answerable: ground truth checked directly,
        # KRAH's chunks currently span 2026-08-29 18:00 through
        # 2026-09-01 22:44 (~3 days) -- real multi-day content to trend
        # across. A good answer needs chunks read in time order to describe
        # a trend correctly, not just the most topically relevant ones
        # regardless of when they were issued -- exactly the difference
        # between explain_forecast_reasoning's chronological ordering and
        # search_forecast_history's distance-only ordering.
        "question": "How has the weather been trending the past few days in KRAH?",
        "expected_calls": [
            {
                "name": "explain_krah",
                "calls": [
                    {
                        "tool": "explain_forecast_reasoning",
                        "input": {"office_id": "KRAH"},
                        "optional_params": {"query", "top_k"},
                    }
                ],
                "score": 10,
                "exclusive": "search_krah",
            },
            {
                "name": "search_krah",
                "calls": [
                    {
                        "tool": "search_forecast_history",
                        "input": {"office_id": "KRAH"},
                        "optional_params": {"query", "top_k"},
                    }
                ],
                "score": 7,
                "exclusive": "explain_krah",
            },
        ],
    },
    {
        # A starker two-office comparison than KRAH-vs-KLIX (both of which
        # are just "hot and stormy" to varying degrees, making a blended or
        # wrong answer easy to miss). Ground truth checked directly: KPSR
        # (Phoenix) is mid-monsoon with active thunderstorms, a Flood Watch,
        # and heavy rainfall risk; KSEW (Seattle) is dry, mild, marine
        # stratus, low fire-weather risk -- genuinely contrasting content, so
        # a wrong or conflated answer should be much easier to catch. Raw
        # codes given directly, sidestepping the office-resolution gap on
        # purpose -- this is a pure retrieval/synthesis test, not compounded
        # with resolution difficulty.
        "question": "Compare the weather in KPSR and KSEW right now.",
        "expected_calls": [
            {
                "name": "explain_both",
                "calls": [
                    {
                        "tool": "explain_forecast_reasoning",
                        "input": {"office_id": "KPSR"},
                        "optional_params": {"query", "top_k"},
                    },
                    {
                        "tool": "explain_forecast_reasoning",
                        "input": {"office_id": "KSEW"},
                        "optional_params": {"query", "top_k"},
                    },
                ],
                "score": 10,
                "exclusive": ["search_both", "broad_search"],
            },
            {
                "name": "search_both",
                "calls": [
                    {
                        "tool": "search_forecast_history",
                        "input": {"office_id": "KPSR"},
                        "optional_params": {"query", "top_k"},
                    },
                    {
                        "tool": "search_forecast_history",
                        "input": {"office_id": "KSEW"},
                        "optional_params": {"query", "top_k"},
                    },
                ],
                "score": 9,
                "exclusive": ["explain_both", "broad_search"],
            },
            {
                "name": "broad_search",
                "calls": [
                    {
                        "tool": "search_forecast_history",
                        "input": {},
                        "optional_params": {"query", "top_k", "office_id"},
                    }
                ],
                "score": 4,
                "exclusive": ["explain_both", "search_both"],
            },
        ],
    },
]
