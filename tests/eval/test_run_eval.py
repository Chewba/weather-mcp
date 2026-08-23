import pytest
from run_eval import parse_judge_score


def test_parse_judge_score_bare_integer():
    assert parse_judge_score("7") == 7


def test_parse_judge_score_markdown_wrapped():
    # Regression: a judge response of "**7**" crashed a 16-question batch
    # partway through with ValueError, wasting the rest of that run's cost.
    assert parse_judge_score("**7**") == 7


def test_parse_judge_score_prose_wrapped():
    assert parse_judge_score("I would rate this a 7 out of 10.") == 7


def test_parse_judge_score_no_integer_raises():
    with pytest.raises(ValueError):
        parse_judge_score("no numbers here")
