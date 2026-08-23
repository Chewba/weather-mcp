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
]
