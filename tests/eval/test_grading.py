from grading import grade_facts, grade_question

# Kansas City style: three independent, complementary strategies. No exclusivity.
COMPLEMENTARY_STRATEGIES = [
    {
        "name": "conditions",
        "calls": [
            {"tool": "get_current_conditions", "input": {"address": "Kansas City, MO"}}
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
            {"tool": "get_active_alerts", "input": {"address": "Kansas City, MO"}}
        ],
        "score": 10,
    },
]

# Greensboro/Charlotte style: one preferred single-call strategy, one redundant
# multi-call alternative, marked mutually exclusive.
COMPARE_STRATEGIES = [
    {
        "name": "prime",
        "calls": [
            {
                "tool": "compare_forecasts",
                "input": {"address1": "Greensboro, NC", "address2": "Charlotte, NC"},
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
]


def test_single_call_full_score():
    calls = [("get_current_conditions", {"address": "Kansas City, MO"})]
    assert grade_question(calls, COMPLEMENTARY_STRATEGIES) == 10


def test_complementary_calls_all_score():
    calls = [
        ("get_current_conditions", {"address": "Kansas City, MO"}),
        ("get_daily_forecast", {"address": "Kansas City, MO"}),
        ("get_active_alerts", {"address": "Kansas City, MO"}),
    ]
    assert grade_question(calls, COMPLEMENTARY_STRATEGIES) == 28


def test_optional_param_still_matches():
    calls = [("get_daily_forecast", {"address": "Kansas City, MO", "days": 5})]
    assert grade_question(calls, COMPLEMENTARY_STRATEGIES) == 8


def test_unrecognized_param_does_not_match():
    calls = [("get_daily_forecast", {"address": "Kansas City, MO", "bogus": True})]
    assert grade_question(calls, COMPLEMENTARY_STRATEGIES) == 0


def test_unexpected_call_is_penalized():
    calls = [
        ("get_current_conditions", {"address": "Kansas City, MO"}),
        ("get_hourly_forecast", {"address": "Kansas City, MO"}),
    ]
    assert grade_question(calls, COMPLEMENTARY_STRATEGIES) == 10 - 2


def test_unexpected_call_alone_clamps_to_zero():
    calls = [("get_hourly_forecast", {"address": "Kansas City, MO"})]
    assert grade_question(calls, COMPLEMENTARY_STRATEGIES) == 0


def test_prime_strategy_alone_full_score():
    calls = [
        (
            "compare_forecasts",
            {"address1": "Greensboro, NC", "address2": "Charlotte, NC"},
        )
    ]
    assert grade_question(calls, COMPARE_STRATEGIES) == 10


def test_secondary_strategy_alone_full_score():
    calls = [
        ("get_daily_forecast", {"address": "Greensboro, NC"}),
        ("get_daily_forecast", {"address": "Charlotte, NC"}),
    ]
    assert grade_question(calls, COMPARE_STRATEGIES) == 4


def test_prime_and_secondary_together_penalizes_the_redundant_one():
    calls = [
        (
            "compare_forecasts",
            {"address1": "Greensboro, NC", "address2": "Charlotte, NC"},
        ),
        ("get_daily_forecast", {"address": "Greensboro, NC"}),
        ("get_daily_forecast", {"address": "Charlotte, NC"}),
    ]
    # prime wins (score 10), both secondary calls docked as extra calls
    assert grade_question(calls, COMPARE_STRATEGIES) == 10 - 2 * 2


def test_partial_secondary_gets_no_credit():
    # Only one of the two required calls for "secondary" -- comparison is incomplete.
    calls = [("get_daily_forecast", {"address": "Greensboro, NC"})]
    assert grade_question(calls, COMPARE_STRATEGIES) == 0


def test_no_calls_scores_zero():
    assert grade_question([], COMPLEMENTARY_STRATEGIES) == 0


# A zero-call strategy is a valid way to model "the ideal answer is to not call
# any tool" (e.g. declining a non-USA location up front because the docstring
# already says USA-only), exclusive with the tool calls it makes unnecessary.
DECLINE_STRATEGIES = [
    {"name": "declined", "calls": [], "score": 10, "exclusive": ["attempted"]},
    {
        "name": "attempted",
        "calls": [
            {"tool": "get_current_conditions", "input": {"address": "London, UK"}}
        ],
        "score": 6,
        "exclusive": "declined",
    },
]


def test_declining_with_no_calls_scores_full_credit():
    assert grade_question([], DECLINE_STRATEGIES) == 10


def test_calling_anyway_still_scores_but_less():
    calls = [("get_current_conditions", {"address": "London, UK"})]
    # declined (score 10) wins the cluster; the attempted call is docked as extra
    assert grade_question(calls, DECLINE_STRATEGIES) == 10 - 2


# grade_facts: deterministic ground-truth check against final answer text,
# added after a real eval run showed the LLM judge scoring a checkably-wrong
# answer highly and a checkably-correct one poorly on the same question shape.

ORIGIN_FACTS = {"must_mention": {"KFWD": 10}}

CHAIN_FACTS = {
    "ordered_sequence": ["KFWD", "KSHV", "KMEG", "KOHX", "KFFC", "KCAE", "KRAH"],
    "sequence_item_points": 1,
    "order_bonus": 3,
}


def test_must_mention_hit_is_case_insensitive():
    assert grade_facts("The ridge originated over kfwd on Aug 29.", ORIGIN_FACTS) == 10


def test_must_mention_miss_scores_zero():
    # Real observed failure: the model named KMEG (a real intermediate hop)
    # as the origin instead of the true origin, KFWD.
    assert grade_facts("KMEG appears to be the first office to report it.", ORIGIN_FACTS) == 0


def test_ordered_sequence_full_match_with_order_bonus():
    answer = "First KFWD, then KSHV, then KMEG, then KOHX, then KFFC, then KCAE, then KRAH."
    assert grade_facts(answer, CHAIN_FACTS) == 7 * 1 + 3


def test_ordered_sequence_partial_match_in_order_still_gets_bonus():
    # Real observed result: 6 of 7 offices named, correct relative order,
    # missing only KCAE -- should score close to full, not collapse to zero.
    answer = "KFWD, then KSHV, then KMEG, then KOHX, then KFFC, then KRAH."
    assert grade_facts(answer, CHAIN_FACTS) == 6 * 1 + 3


def test_ordered_sequence_out_of_order_gets_no_bonus():
    answer = "KRAH mentioned it, and so did KFWD earlier on."
    assert grade_facts(answer, CHAIN_FACTS) == 2 * 1


def test_ordered_sequence_no_matches_scores_zero():
    assert grade_facts("No offices are named here at all.", CHAIN_FACTS) == 0


def test_ordered_sequence_single_match_gets_no_order_bonus():
    # Order bonus requires at least two found items to mean anything.
    assert grade_facts("Only KFWD is mentioned.", CHAIN_FACTS) == 1
