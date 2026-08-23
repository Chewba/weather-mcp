UNEXPECTED_CALL_PENALTY = 2


def match_call(tool_name: str, tool_input: dict | None, expected_call: dict) -> bool:
    """Does one actual tool call satisfy one expected call spec?"""
    required = expected_call["input"]
    optional = expected_call.get("optional_params", set())
    inp = tool_input or {}
    return (
        expected_call["tool"] == tool_name
        and all(inp.get(k) == v for k, v in required.items())
        and not (set(inp) - set(required) - optional)
    )


def find_strategy_match(
    actual_calls: list[tuple[str, dict]], used: set[int], strategy: dict
) -> list[int] | None:
    """All of strategy['calls'] must be satisfied by distinct, unused actual calls."""
    matched = []
    for expected_call in strategy["calls"]:
        found = next(
            (
                i
                for i in range(len(actual_calls))
                if i not in used
                and i not in matched
                and match_call(actual_calls[i][0], actual_calls[i][1], expected_call)
            ),
            None,
        )
        if found is None:
            return None
        matched.append(found)
    return matched


def as_list(value) -> list:
    return [value] if isinstance(value, str) else (value or [])


def grade_question(actual_calls: list[tuple[str, dict]], strategies: list[dict]) -> int:
    """Scores the tool calls a model made against a list of named strategies.

    Each strategy is one or more required calls plus a score. Strategies whose
    names appear in each other's `exclusive` list are treated as redundant
    alternatives: only the highest-scoring one counts, and every call belonging
    to the others is docked as an extra call, same as a genuinely unexpected call.
    """
    used, matched_strategies = set(), []
    for strategy in strategies:
        idxs = find_strategy_match(actual_calls, used, strategy)
        if idxs is not None:
            matched_strategies.append((strategy, idxs))
            used.update(idxs)

    consumed, score, extra_calls = set(), 0, 0
    for i, (strategy, idxs) in enumerate(matched_strategies):
        if i in consumed:
            continue
        exclusive_names = as_list(strategy.get("exclusive"))
        cluster = [i] + [
            j
            for j, (other, _) in enumerate(matched_strategies)
            if j != i
            and j not in consumed
            and (
                other["name"] in exclusive_names
                or strategy["name"] in as_list(other.get("exclusive"))
            )
        ]
        consumed.update(cluster)
        cluster_items = [matched_strategies[k] for k in cluster]
        winner, winner_idxs = max(cluster_items, key=lambda x: x[0]["score"])
        score += winner["score"]
        extra_calls += sum(len(idxs_) for _, idxs_ in cluster_items) - len(winner_idxs)

    unexpected = len(actual_calls) - len(used)
    return max(score - UNEXPECTED_CALL_PENALTY * (unexpected + extra_calls), 0)
