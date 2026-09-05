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


def grade_facts(final_answer: str, expected_facts: dict) -> int:
    """Deterministic, ground-truth-based check against the model's final
    answer text. Complements quality_score for questions with an objectively
    checkable answer (a specific office code, a chain of offices in order) --
    the LLM judge has been observed scoring a checkably-wrong answer highly
    and a checkably-correct one poorly on the same kind of question, so a
    text-level fact check is a more reliable signal than the judge alone for
    these questions specifically. Only meaningful for questions that define
    `expected_facts`; not a replacement for quality_score in general, since
    most questions don't have an answer this checkable.

    `must_mention`: {fact: points} -- points awarded if `fact` (case-
    insensitive) appears anywhere in the answer.

    `ordered_sequence`: a list of facts expected to appear in this relative
    order. Each one found (regardless of order) earns `sequence_item_points`
    (default 1); an `order_bonus` (default 0) is added if every found item
    appears in the correct relative order (missing items don't break this --
    a 6-of-7 chain found in the right order still earns the bonus).

    `first_mention`: {"among": [...], "expected": fact, "points": N} --
    awards `points` only if `expected` is the one from `among` that appears
    EARLIEST in the answer (case-insensitive), not just present somewhere.
    `must_mention` alone can't distinguish "correctly identifies X as the
    answer" from "mentions X in passing while naming something else as the
    answer" -- a real false positive seen in practice: an answer that named
    the wrong office first but mentioned the right one later scored a full
    `must_mention` match despite being wrong. Scores 0 if none of `among`
    appears in the answer at all.
    """
    text = final_answer.lower()
    score = 0

    for fact, points in expected_facts.get("must_mention", {}).items():
        if fact.lower() in text:
            score += points

    first_mention = expected_facts.get("first_mention")
    if first_mention:
        among = first_mention["among"]
        positions = [(item, text.find(item.lower())) for item in among]
        found = [(item, pos) for item, pos in positions if pos != -1]
        if found:
            earliest_item, _ = min(found, key=lambda pair: pair[1])
            if earliest_item == first_mention["expected"]:
                score += first_mention.get("points", 10)

    ordered = expected_facts.get("ordered_sequence")
    if ordered:
        item_points = expected_facts.get("sequence_item_points", 1)
        order_bonus = expected_facts.get("order_bonus", 0)
        positions = [(item, text.find(item.lower())) for item in ordered]
        found = [(item, pos) for item, pos in positions if pos != -1]
        score += len(found) * item_points
        found_positions = [pos for _, pos in found]
        if len(found) > 1 and found_positions == sorted(found_positions):
            score += order_bonus

    return score
