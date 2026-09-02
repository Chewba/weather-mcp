"""Shared eval question set, referenced by any harness that scores this server's
tool-selection behavior -- headless CLI (run_cli.py) or direct API calls (run.py) --
so results are comparable against the same dataset regardless of how the model is invoked."""

TESTING_DATA = [
    {
        "question": "What is the weather in Kansas City, MO?",
        "expected_calls": [
            {
                "name": "conditions",
                "calls": [
                    {
                        "tool": "get_current_conditions",
                        "input": {"address": "Kansas City, MO"},
                    }
                ],
                "score": 10,
            },
            {
                "name": "forecast",
                "calls": [
                    {
                        "tool": "get_daily_forecast",
                        "input": {"address": "Kansas City, MO"},
                        "optional_params": {"days"},
                    }
                ],
                "score": 8,
            },
            {
                "name": "alerts",
                "calls": [
                    {
                        "tool": "get_active_alerts",
                        "input": {"address": "Kansas City, MO"},
                    }
                ],
                "score": 10,
            },
        ],
    },
    {
        "question": "What will the weather be Kansas City, MO in an hour?",
        "expected_calls": [
            {
                "name": "hourly",
                "calls": [
                    {
                        "tool": "get_hourly_forecast",
                        "input": {"address": "Kansas City, MO"},
                        "optional_params": {"hours"},
                    }
                ],
                "score": 10,
            },
        ],
    },
    {
        "question": "What will the weather be Kansas City, MO tonight?",
        "expected_calls": [
            {
                "name": "daily",
                "calls": [
                    {
                        "tool": "get_daily_forecast",
                        "input": {"address": "Kansas City, MO"},
                        "optional_params": {"days"},
                    }
                ],
                "score": 10,
            },
            {
                "name": "hourly",
                "calls": [
                    {
                        "tool": "get_hourly_forecast",
                        "input": {"address": "Kansas City, MO"},
                        "optional_params": {"hours"},
                    }
                ],
                "score": 8,
            },
        ],
    },
    {
        "question": "Which city has better weather for the next few days Greensboro, NC or Charlotte, NC?",
        "expected_calls": [
            {
                "name": "prime",
                "calls": [
                    {
                        "tool": "compare_forecasts",
                        "input": {
                            "address1": "Greensboro, NC",
                            "address2": "Charlotte, NC",
                        },
                        "optional_params": {"days"},
                    }
                ],
                "score": 10,
                "exclusive": "secondary",
            },
            {
                "name": "secondary",
                "calls": [
                    {
                        "tool": "get_daily_forecast",
                        "input": {"address": "Greensboro, NC"},
                        "optional_params": {"days"},
                    },
                    {
                        "tool": "get_daily_forecast",
                        "input": {"address": "Charlotte, NC"},
                        "optional_params": {"days"},
                    },
                ],
                "score": 4,
                "exclusive": "prime",
            },
        ],
    },
    {
        "question": "are there any active weather alerts for greensboro, nc?",
        "expected_calls": [
            {
                "name": "alerts",
                "calls": [
                    {
                        "tool": "get_active_alerts",
                        "input": {"address": "Greensboro, NC"},
                    }
                ],
                "score": 10,
            },
        ],
    },
    {
        "question": "I am heading to Boone, NC should be there in 5 hours what weather should I expect when I get there and over my stay over the weekend?",
        "expected_calls": [
            {
                "name": "arrival",
                "calls": [
                    {
                        "tool": "get_hourly_forecast",
                        "input": {"address": "Boone, NC"},
                        "optional_params": {"hours"},
                    }
                ],
                "score": 6,
            },
            {
                "name": "weekend_outlook",
                "calls": [
                    {
                        "tool": "get_daily_forecast",
                        "input": {"address": "Boone, NC"},
                        "optional_params": {"days"},
                    }
                ],
                "score": 6,
            },
        ],
    },
    {
        "question": "What is the forecast for greensboro, nc and why do they think it is going to be like that?",
        "expected_calls": [
            {
                "name": "forecast",
                "calls": [
                    {
                        "tool": "get_daily_forecast",
                        "input": {"address": "Greensboro, NC"},
                        "optional_params": {"days"},
                    }
                ],
                "score": 6,
            },
            {
                "name": "reasoning",
                "calls": [
                    {
                        "tool": "get_weather_discussion",
                        "input": {"address": "Greensboro, NC"},
                        "optional_params": {"count"},
                    }
                ],
                "score": 8,
            },
        ],
    },
    {
        "question": "What's the forecast for 27401 for the next couple days?",
        "expected_calls": [
            {
                "name": "forecast",
                "calls": [
                    {
                        "tool": "get_daily_forecast",
                        "input": {"address": "27401"},
                        "optional_params": {"days"},
                    }
                ],
                "score": 10,
            },
        ],
    },
    {
        "question": "What are the National Weather Service forecasters saying about the weather pattern for Charlotte, NC this week?",
        "expected_calls": [
            {
                "name": "discussion",
                "calls": [
                    {
                        "tool": "get_weather_discussion",
                        "input": {"address": "Charlotte, NC"},
                        "optional_params": {"count"},
                    }
                ],
                "score": 10,
            },
        ],
    },
    {
        "question": "Should I visit Asheville or Boone, NC next week for hiking?",
        "expected_calls": [
            {
                "name": "prime",
                "calls": [
                    {
                        "tool": "compare_forecasts",
                        "input": {"address1": "Asheville, NC", "address2": "Boone, NC"},
                        "optional_params": {"days"},
                    }
                ],
                "score": 10,
                "exclusive": "secondary",
            },
            {
                "name": "secondary",
                "calls": [
                    {
                        "tool": "get_daily_forecast",
                        "input": {"address": "Asheville, NC"},
                        "optional_params": {"days"},
                    },
                    {
                        "tool": "get_daily_forecast",
                        "input": {"address": "Boone, NC"},
                        "optional_params": {"days"},
                    },
                ],
                "score": 4,
                "exclusive": "prime",
            },
        ],
    },
    {
        "question": "Is it safe to drive through Greensboro, NC right now?",
        "expected_calls": [
            {
                "name": "conditions",
                "calls": [
                    {
                        "tool": "get_current_conditions",
                        "input": {"address": "Greensboro, NC"},
                    }
                ],
                "score": 6,
            },
            {
                "name": "alerts",
                "calls": [
                    {
                        "tool": "get_active_alerts",
                        "input": {"address": "Greensboro, NC"},
                    }
                ],
                "score": 6,
            },
        ],
    },
    {
        "question": "What's the forecast for the next 5 days in Kansas City, MO?",
        "expected_calls": [
            {
                "name": "forecast",
                "calls": [
                    {
                        "tool": "get_daily_forecast",
                        "input": {"address": "Kansas City, MO", "days": 5},
                    }
                ],
                "score": 10,
            },
        ],
    },
    {
        # NWS's daily forecast caps at 7 days no matter the `days` param, which
        # maps exactly onto "this week" and not at all onto "next week" -- so
        # unlike a "this weekend or next weekend" phrasing (ambiguous to humans
        # too, especially on a Sunday), any specific claim about "next week" here
        # is unambiguous fabrication, no interpretation required. The interesting
        # axis is quality (does the model admit the data limit?), not tool_score
        # -- there's only one sensible tool to call.
        "question": "Is it better to go to Boone, NC this week or next week?",
        "expected_calls": [
            {
                "name": "forecast",
                "calls": [
                    {
                        "tool": "get_daily_forecast",
                        "input": {"address": "Boone, NC"},
                        "optional_params": {"days"},
                    }
                ],
                "score": 10,
            },
        ],
    },
    {
        # Originally assumed a tool call was the only way to discover "Atlantis"
        # isn't real (exercising LocationNotFoundError). Wrong: the model's own
        # world knowledge recognizes it as mythical and declines without calling
        # anything, which is the better outcome -- no point spending an API call
        # confirming something already known. Declining is scored highest;
        # calling anyway to double-check is acceptable but wasteful.
        "question": "What's the weather in Atlantis?",
        "expected_calls": [
            {
                "name": "declined",
                "calls": [],
                "score": 10,
                "exclusive": ["attempted_conditions", "attempted_forecast"],
            },
            {
                "name": "attempted_conditions",
                "calls": [
                    {"tool": "get_current_conditions", "input": {"address": "Atlantis"}}
                ],
                "score": 6,
                "exclusive": "declined",
            },
            {
                "name": "attempted_forecast",
                "calls": [
                    {
                        "tool": "get_daily_forecast",
                        "input": {"address": "Atlantis"},
                        "optional_params": {"days"},
                    }
                ],
                "score": 6,
                "exclusive": "declined",
            },
        ],
    },
    {
        # Same wrong assumption as Atlantis, in reverse direction: expected a
        # tool call was needed to surface LocationAmbiguousError, but the model
        # already knows there are multiple US Springfields and asks for
        # clarification directly, without wasting a call to discover that.
        "question": "What's the weather in Springfield?",
        "expected_calls": [
            {
                "name": "declined",
                "calls": [],
                "score": 10,
                "exclusive": ["attempted_conditions", "attempted_forecast"],
            },
            {
                "name": "attempted_conditions",
                "calls": [
                    {
                        "tool": "get_current_conditions",
                        "input": {"address": "Springfield"},
                    }
                ],
                "score": 6,
                "exclusive": "declined",
            },
            {
                "name": "attempted_forecast",
                "calls": [
                    {
                        "tool": "get_daily_forecast",
                        "input": {"address": "Springfield"},
                        "optional_params": {"days"},
                    }
                ],
                "score": 6,
                "exclusive": "declined",
            },
        ],
    },
    {
        # Every tool docstring now says "(USA locations only)" -- this tests
        # whether that actually changes behavior. Declining up front is the
        # ideal (highest-scored) strategy since the docstring already tells the
        # model this server can't help; calling anyway and relaying whatever
        # NWS returns (likely a graceful ServiceUnavailableError, since NWS
        # doesn't cover the UK) is acceptable but wastes a call, so it's marked
        # exclusive with -- and scored lower than -- declining outright.
        "question": "What's the weather in London, UK?",
        "expected_calls": [
            {
                "name": "declined",
                "calls": [],
                "score": 10,
                "exclusive": ["attempted_conditions", "attempted_forecast"],
            },
            {
                "name": "attempted_conditions",
                "calls": [
                    {
                        "tool": "get_current_conditions",
                        "input": {"address": "London, UK"},
                    }
                ],
                "score": 6,
                "exclusive": "declined",
            },
            {
                "name": "attempted_forecast",
                "calls": [
                    {
                        "tool": "get_daily_forecast",
                        "input": {"address": "London, UK"},
                        "optional_params": {"days"},
                    }
                ],
                "score": 6,
                "exclusive": "declined",
            },
        ],
    },
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
        "question": "Was the weather in Dallas, TX better than Greensboro, NC at the end of August?",
        "expected_calls": [
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
                "exclusive": ["search_both", "broad_search"],
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
        # v3 candidate #2: a sharper, more checkably-gradable version of the
        # multi-hop causality question above. That question asks for an open
        # origin narrative; this asks for a specific fact -- a named office
        # and an approximate date -- that can be checked directly against the
        # final answer text (does it say "KFWD" and "Aug 29"?) rather than
        # relying purely on the subjective LLM judge. Documented result for
        # the narrative version applies here too: even scoped correctly to
        # KRAH, KFWD's origin doesn't surface in the top hits today, so this
        # should score decent tool_score but a wrong/missing answer on
        # quality_score -- that gap is the whole point. This is the intended
        # regression benchmark for v3: re-run it once adaptive/multi-hop
        # retrieval exists and check whether the answer starts correctly
        # naming KFWD around Aug 29.
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
        "question": "List, in order, every NWS office that mentioned this ridge as it moved from Texas to North Carolina.",
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
]
