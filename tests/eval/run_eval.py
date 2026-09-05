import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import anthropic
from config import MODEL_JUDGE, MODEL_UNDER_TEST
from grading import grade_facts, grade_question
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from questions_mcp import MCP_QUESTIONS
from questions_rag import RAG_QUESTIONS

# Windows' stdout/stderr default to the console codepage (e.g. cp1252), not
# UTF-8, even when redirected to a file -- tool data and model answers can
# contain arbitrary Unicode (arrows, smart quotes, degree signs), so a plain
# print() of either crashes with UnicodeEncodeError the moment one shows up.
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
MCP_SERVER_PARAMS = {
    "command": "uv",
    "args": ["run", "--directory", str(REPO_ROOT), "weather-mcp"],
}

# scripts/ isn't a package under src/, so it's not importable via the normal
# weather_mcp namespace -- add it to sys.path the same way this file already
# relies on its own directory being on sys.path for `from config import ...`.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from seed_db import seed as seed_corpus

# Scoped per question-set rather than one shared allow-list: MCP_QUESTIONS
# legitimately need get_weather_discussion/compare_forecasts, but no
# RAG_QUESTIONS expected_calls ever call for those (verified against
# questions_rag.py) -- yet get_weather_discussion pulls a full live AFD
# product (multi-KB), and that raw text gets embedded verbatim in the judge
# prompt (build_judge_prompt) and re-sent once per --judge-samples. Handing
# the model access to it during RAG-only questions burned real usage on
# nothing the RAG tools were even being tested for.
MCP_TOOLS = (
    "mcp__weather__get_daily_forecast,mcp__weather__get_hourly_forecast,"
    "mcp__weather__get_current_conditions,mcp__weather__get_active_alerts,"
    "mcp__weather__get_weather_discussion,mcp__weather__compare_forecasts"
)
RAG_TOOLS = (
    "mcp__weather__search_forecast_history,mcp__weather__explain_forecast_reasoning"
)
# Headless `claude -p` runs as a full coding agent in REPO_ROOT, not a
# sandbox limited to the MCP tools above -- confirmed live when
# search_forecast_history errored (DB connection refused) and the model
# fell back to Read-ing tests/eval/questions_rag.py's answer key straight
# off disk and reciting it, scoring a perfect (and meaningless) fact_score.
# --allowedTools only pre-approves the MCP tools; it does not deny these.
DISALLOWED_BUILTIN_TOOLS = (
    "Read,Grep,Glob,Bash,Write,Edit,NotebookEdit,WebFetch,WebSearch,Task"
)
JUDGE_SAMPLES = 3
JUDGE_PROMPT_TEMPLATE = (
    "You are testing how well the question is answered, you have the question; "
    "the tool response, and the ai answer. **ONLY** answer with an integer between 0 and 10. "
    "question:{question}. tool_data:{tool_data}. AI answer:{final_answer}"
)

# List prices per million tokens (input, output), used only by the "api" backend
# to estimate real spend from token usage. Verify against
# https://www.anthropic.com/pricing before treating these as authoritative --
# actual API billing uses the account's real rate at call time, not this table.
PRICING_PER_MTOK_USD = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (3.00, 15.00),
}


def select_questions(question_set: str, text_filter: str | None = None) -> list[tuple[dict, str]]:
    """Pairs each question with the tool allow-list it should run under, so
    a --question-set both run still scopes RAG questions away from
    get_weather_discussion/compare_forecasts (and vice versa) per-question,
    not just for pure single-set runs.

    text_filter, if given, keeps only questions whose text contains it
    (case-insensitive) -- lets a retry target one or two specific questions
    (e.g. the v3 adaptive-retrieval benchmarks) without re-spending usage on
    the full set."""
    if question_set == "mcp":
        selected = [(q, MCP_TOOLS) for q in MCP_QUESTIONS]
    elif question_set == "rag":
        selected = [(q, RAG_TOOLS) for q in RAG_QUESTIONS]
    else:
        selected = [(q, MCP_TOOLS) for q in MCP_QUESTIONS] + [
            (q, RAG_TOOLS) for q in RAG_QUESTIONS
        ]
    if text_filter:
        needle = text_filter.lower()
        selected = [(q, t) for q, t in selected if needle in q["question"].lower()]
    return selected

# ---------------------------------------------------------------------------
# Shared: judge prompt construction, score averaging, cost reporting -- used
# identically by both backends so results are comparable across them.
# ---------------------------------------------------------------------------


def parse_judge_score(text: str) -> int:
    """The judge prompt asks for a bare integer, but models sometimes wrap it
    (e.g. "**7**") despite that instruction -- pull the first integer out
    rather than crashing the whole batch on a stray markdown wrapper."""
    match = re.search(r"\d+", text)
    if not match:
        raise ValueError(f"Judge response contained no integer: {text!r}")
    return int(match.group())


def build_judge_prompt(answer: dict) -> str:
    tool_data = (
        "\n\n".join(r for r in answer["tool_responses"] if r)
        or "No tool data available."
    )
    return JUDGE_PROMPT_TEMPLATE.format(
        question=answer["question"],
        tool_data=tool_data,
        final_answer=answer["final_answer"],
    )


def build_rating(
    answer: dict,
    expected_calls: list[dict],
    scores: list[int],
    judge_cost_usd: float,
    expected_facts: dict | None = None,
) -> dict:
    quality_score = round(sum(scores) / len(scores))
    print("\nai ratings", scores, "-> averaged", quality_score)
    rating = {
        "tool_calls": answer["tool_calls"],
        "tool_score": grade_question(answer["tool_calls"], expected_calls),
        "quality_scores": scores,
        "quality_score": quality_score,
        "model_cost_usd": answer["cost_usd"],
        "judge_cost_usd": judge_cost_usd,
    }
    if expected_facts:
        # Deterministic ground-truth check, only for questions with an
        # objectively checkable answer -- see grade_facts' docstring for why
        # this exists alongside (not instead of) quality_score.
        fact_score = grade_facts(answer["final_answer"], expected_facts)
        rating["fact_score"] = fact_score
        print("fact_score:", fact_score)
    return rating


def print_cost_summary(
    args,
    n: int,
    total_model_cost: float,
    total_judge_cost: float,
    total_tool_score: int,
    total_quality_score: int,
    note: str,
) -> None:
    print("\nDONE")
    print(
        f"\n--- Cost summary ({args.backend} backend, {args.model} under test, {args.judge_model} judging x{args.judge_samples}) ---"
    )
    print(note)
    print(
        f"Model-under-test: ${total_model_cost:.4f} total, ${total_model_cost / n:.4f}/question"
    )
    print(
        f"Judge:            ${total_judge_cost:.4f} total, ${total_judge_cost / n:.4f}/question"
    )
    if total_model_cost > 0:
        print(
            f"Avg tool_score {total_tool_score / n:.1f} / avg quality_score {total_quality_score / n:.1f} "
            f"at ${total_model_cost / n:.4f} per question ({(total_quality_score / n) / total_model_cost:.1f} quality points per $ spent on {args.model})"
        )


def print_fact_summary(fact_results: list[tuple[str, int]]) -> None:
    """Ground-truth fact checks, separate from the cost/quality summary above
    since they only apply to the handful of questions with an objectively
    checkable answer -- see grade_facts' docstring."""
    if not fact_results:
        return
    print("\n--- Ground-truth fact checks (deterministic, not judge-scored) ---")
    for question, score in fact_results:
        print(f"  {score:>2}  {question}")


# ---------------------------------------------------------------------------
# Headless backend: shells out to `claude -p`, billed against the caller's
# Claude subscription rather than ANTHROPIC_API_KEY.
# ---------------------------------------------------------------------------


def run_claude(
    prompt: str, model: str, use_mcp: bool, allowed_tools: str = MCP_TOOLS + "," + RAG_TOOLS
) -> list[dict]:
    # Prompt is piped via stdin rather than passed as a CLI argument -- the
    # judge prompt embeds full raw tool output (e.g. search_forecast_history
    # can return several KB of AFD passages), which blew past Windows'
    # command-line length limit (WinError 206) when passed as argv.
    cmd = [
        "claude",
        "-p",
        "--model",
        model,
        "--output-format",
        "stream-json",
        "--verbose",
    ]
    if use_mcp:
        mcp_config = json.dumps({"mcpServers": {"weather": MCP_SERVER_PARAMS}})
        cmd += [
            "--mcp-config",
            mcp_config,
            "--strict-mcp-config",
            "--allowedTools",
            allowed_tools,
            "--disallowedTools",
            DISALLOWED_BUILTIN_TOOLS,
        ]
    result = subprocess.run(
        cmd, input=prompt, capture_output=True, text=True, encoding="utf-8", check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude CLI exited {result.returncode}: {result.stderr}")
    return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]


def extract_weather_tool_calls(events: list[dict]) -> list[tuple[str, dict, str]]:
    """Returns (tool_name, tool_input, tool_use_id) for every weather-server tool
    call in the transcript, ignoring incidental calls like ToolSearch."""
    calls = []
    for event in events:
        if event.get("type") != "assistant":
            continue
        for block in event["message"]["content"]:
            if block["type"] == "tool_use" and block["name"].startswith(
                "mcp__weather__"
            ):
                calls.append(
                    (
                        block["name"].removeprefix("mcp__weather__"),
                        block["input"],
                        block["id"],
                    )
                )
    return calls


def extract_tool_result(events: list[dict], tool_use_id: str) -> str | None:
    for event in events:
        if event.get("type") != "user":
            continue
        for block in event["message"]["content"]:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_result"
                and block.get("tool_use_id") == tool_use_id
            ):
                content = block.get("content")
                try:
                    return json.loads(content).get("result", content)
                except (TypeError, ValueError):
                    return content
    return None


def extract_final_text(events: list[dict]) -> str:
    for event in reversed(events):
        if event.get("type") == "result":
            return event.get("result", "")
    return ""


def extract_cost_usd(events: list[dict]) -> float:
    """Cost the CLI estimates this call would have cost at API pricing. Since
    these calls run on subscription auth (no ANTHROPIC_API_KEY), nothing is
    actually billed per-call -- this is a relative-cost signal, not a real charge."""
    for event in reversed(events):
        if event.get("type") == "result":
            return event.get("total_cost_usd", 0.0)
    return 0.0


def run_question_headless(model: str, question: str, allowed_tools: str) -> dict:
    events = run_claude(question, model, use_mcp=True, allowed_tools=allowed_tools)
    calls = extract_weather_tool_calls(events)
    tool_responses = [
        extract_tool_result(events, tool_use_id) for _, _, tool_use_id in calls
    ]
    return {
        "question": question,
        "tool_calls": [(name, tool_input) for name, tool_input, _ in calls],
        "tool_responses": tool_responses,
        "final_answer": extract_final_text(events),
        "cost_usd": extract_cost_usd(events),
    }


def judge_headless(
    judge_model: str, judge_prompt: str, samples: int
) -> tuple[list[int], float]:
    scores, cost = [], 0.0
    for _ in range(samples):
        events = run_claude(judge_prompt, judge_model, use_mcp=False)
        scores.append(parse_judge_score(extract_final_text(events)))
        cost += extract_cost_usd(events)
    return scores, cost


def run_headless_batch(args) -> None:
    total_model_cost = total_judge_cost = 0.0
    total_tool_score = total_quality_score = 0
    fact_results = []
    questions = select_questions(args.question_set, args.filter)
    n = len(questions)

    for quest, allowed_tools in questions:
        answer = run_question_headless(args.model, quest["question"], allowed_tools)
        print(f"\n{quest['question']} DONE")
        judge_prompt = build_judge_prompt(answer)
        print("\nRating Prompt", judge_prompt)
        scores, judge_cost = judge_headless(
            args.judge_model, judge_prompt, args.judge_samples
        )
        rating = build_rating(
            answer,
            quest["expected_calls"],
            scores,
            judge_cost,
            expected_facts=quest.get("expected_facts"),
        )
        print(rating)
        total_model_cost += rating["model_cost_usd"]
        total_judge_cost += rating["judge_cost_usd"]
        total_tool_score += rating["tool_score"]
        total_quality_score += rating["quality_score"]
        if "fact_score" in rating:
            fact_results.append((quest["question"], rating["fact_score"]))

    print_cost_summary(
        args,
        n,
        total_model_cost,
        total_judge_cost,
        total_tool_score,
        total_quality_score,
        "NOTE: these are the CLI's estimated API-equivalent prices, billed against your\n"
        "subscription rather than charged per-call -- a relative cost signal, not a real charge.",
    )
    print_fact_summary(fact_results)


# ---------------------------------------------------------------------------
# API backend: direct anthropic.messages.create() calls plus a real MCP stdio
# session, billed against ANTHROPIC_API_KEY -- these charges are real.
# ---------------------------------------------------------------------------


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    input_price, output_price = PRICING_PER_MTOK_USD.get(model, (0.0, 0.0))
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


async def run_question_api(
    client: anthropic.Anthropic,
    session: ClientSession,
    anthropic_tools: list[dict],
    model: str,
    question: str,
) -> dict:
    messages = [{"role": "user", "content": question}]
    tool_calls, tool_responses = [], []
    input_tokens = output_tokens = 0

    while True:
        response = client.messages.create(
            model=model, max_tokens=1024, tools=anthropic_tools, messages=messages
        )
        input_tokens += response.usage.input_tokens
        output_tokens += response.usage.output_tokens
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            final_answer = "".join(
                block.text for block in response.content if block.type == "text"
            )
            break

        tool_result_blocks = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            tool_calls.append((block.name, block.input))
            result = await session.call_tool(block.name, block.input)
            text = result.content[0].text if result.content else ""
            tool_responses.append(text)
            tool_result_blocks.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": text}
            )
        messages.append({"role": "user", "content": tool_result_blocks})

    return {
        "question": question,
        "tool_calls": tool_calls,
        "tool_responses": tool_responses,
        "final_answer": final_answer,
        "cost_usd": estimate_cost_usd(model, input_tokens, output_tokens),
    }


def judge_api(
    client: anthropic.Anthropic, judge_model: str, judge_prompt: str, samples: int
) -> tuple[list[int], float]:
    scores, cost = [], 0.0
    for _ in range(samples):
        response = client.messages.create(
            model=judge_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": judge_prompt}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        try:
            scores.append(parse_judge_score(text))
        except ValueError:
            block_types = [block.type for block in response.content]
            raise ValueError(
                f"Judge response had no usable score. stop_reason={response.stop_reason!r}, "
                f"content block types={block_types!r}, prompt length={len(judge_prompt)} chars"
            ) from None
        cost += estimate_cost_usd(
            judge_model, response.usage.input_tokens, response.usage.output_tokens
        )
    return scores, cost


async def run_api_batch(args) -> None:
    client = anthropic.Anthropic()  # picks up ANTHROPIC_API_KEY from env automatically
    params = StdioServerParameters(**MCP_SERVER_PARAMS)

    total_model_cost = total_judge_cost = 0.0
    total_tool_score = total_quality_score = 0
    fact_results = []
    questions = select_questions(args.question_set, args.filter)
    n = len(questions)

    async with (
        stdio_client(params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        tools = await session.list_tools()
        all_anthropic_tools = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in tools.tools
        ]

        for quest, allowed_tools in questions:
            # allowed_tools carries the claude-CLI-style "mcp__weather__x"
            # names (shared with the headless backend); the raw MCP session
            # here exposes tools under their bare name, so strip the prefix
            # before filtering -- see MCP_TOOLS/RAG_TOOLS above for why this
            # scoping exists at all.
            allowed_names = {
                name.removeprefix("mcp__weather__")
                for name in allowed_tools.split(",")
            }
            anthropic_tools = [
                t for t in all_anthropic_tools if t["name"] in allowed_names
            ]
            answer = await run_question_api(
                client, session, anthropic_tools, args.model, quest["question"]
            )
            print(f"\n{quest['question']} DONE")
            judge_prompt = build_judge_prompt(answer)
            print("\nRating Prompt", judge_prompt)
            scores, judge_cost = judge_api(
                client, args.judge_model, judge_prompt, args.judge_samples
            )
            rating = build_rating(
                answer,
                quest["expected_calls"],
                scores,
                judge_cost,
                expected_facts=quest.get("expected_facts"),
            )
            print(rating)
            total_model_cost += rating["model_cost_usd"]
            total_judge_cost += rating["judge_cost_usd"]
            total_tool_score += rating["tool_score"]
            total_quality_score += rating["quality_score"]
            if "fact_score" in rating:
                fact_results.append((quest["question"], rating["fact_score"]))

    print_cost_summary(
        args,
        n,
        total_model_cost,
        total_judge_cost,
        total_tool_score,
        total_quality_score,
        "NOTE: these figures are computed from actual token usage x list pricing above --\n"
        "this backend bills ANTHROPIC_API_KEY for real, unlike the headless backend.",
    )
    print_fact_summary(fact_results)


def main(args) -> None:
    if args.skip_seed:
        print("\nSkipping RAG corpus seeding (--skip-seed) -- running against whatever's already in the DB.")
    elif args.question_set == "mcp":
        print("\nSkipping RAG corpus seeding -- --question-set mcp doesn't touch the RAG corpus.")
    else:
        print(f"\nSeeding RAG corpus (mode={args.corpus_mode})...")
        asyncio.run(seed_corpus(strict=args.corpus_mode == "strict"))

    if args.backend == "api":
        asyncio.run(run_api_batch(args))
    else:
        run_headless_batch(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["headless", "api"],
        default=os.environ.get("EVAL_BACKEND", "headless"),
    )
    parser.add_argument(
        "--model", default=os.environ.get("EVAL_MODEL", MODEL_UNDER_TEST)
    )
    parser.add_argument(
        "--judge-model", default=os.environ.get("JUDGE_MODEL", MODEL_JUDGE)
    )
    parser.add_argument(
        "--judge-samples",
        type=int,
        default=int(os.environ.get("JUDGE_SAMPLES", JUDGE_SAMPLES)),
    )
    parser.add_argument(
        "--question-set",
        choices=["mcp", "rag", "both"],
        default=os.environ.get("EVAL_QUESTION_SET", "both"),
        help=(
            "mcp: only the original, non-RAG tool questions (no DB seeding). "
            "rag: only the RAG retrieval-tool questions. both: all questions "
            "(default) -- keep the sets separate when you only need to re-run "
            "one, so a retry doesn't re-spend usage re-testing the other."
        ),
    )
    parser.add_argument(
        "--corpus-mode",
        choices=["strict", "drift"],
        default=os.environ.get("EVAL_CORPUS_MODE", "drift"),
        help=(
            "strict: reset the DB and seed only the golden fixture, for a "
            "reproducible RAG-tool corpus. drift: seed the golden fixture as a "
            "floor, then layer in the 10 most recent live discussions per "
            "office on top, without resetting existing data."
        ),
    )
    parser.add_argument(
        "--filter",
        default=os.environ.get("EVAL_FILTER"),
        help=(
            "Only run questions whose text contains this substring "
            "(case-insensitive). Useful for cheaply re-testing one or two "
            "questions instead of a whole question set."
        ),
    )
    parser.add_argument(
        "--skip-seed",
        action="store_true",
        default=os.environ.get("EVAL_SKIP_SEED", "").lower() in ("1", "true", "yes"),
        help=(
            "Skip corpus seeding entirely and run against whatever's already "
            "in the DB -- useful when iterating on harness bugs without "
            "paying the reseed cost (and its HF Hub calls) on every retry."
        ),
    )
    main(parser.parse_args())
