import argparse
import asyncio
import json
import os
import re
import subprocess
from pathlib import Path

import anthropic
from config import MODEL_JUDGE, MODEL_UNDER_TEST
from grading import grade_question
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from questions import TESTING_DATA

REPO_ROOT = Path(__file__).resolve().parents[2]
MCP_SERVER_PARAMS = {
    "command": "uv",
    "args": ["run", "--directory", str(REPO_ROOT), "weather-mcp"],
}

ALLOWED_TOOLS = (
    "mcp__weather__get_daily_forecast,mcp__weather__get_hourly_forecast,"
    "mcp__weather__get_current_conditions,mcp__weather__get_active_alerts,"
    "mcp__weather__get_weather_discussion,mcp__weather__compare_forecasts"
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
    answer: dict, expected_calls: list[dict], scores: list[int], judge_cost_usd: float
) -> dict:
    quality_score = round(sum(scores) / len(scores))
    print("\nai ratings", scores, "-> averaged", quality_score)
    return {
        "tool_calls": answer["tool_calls"],
        "tool_score": grade_question(answer["tool_calls"], expected_calls),
        "quality_scores": scores,
        "quality_score": quality_score,
        "model_cost_usd": answer["cost_usd"],
        "judge_cost_usd": judge_cost_usd,
    }


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


# ---------------------------------------------------------------------------
# Headless backend: shells out to `claude -p`, billed against the caller's
# Claude subscription rather than ANTHROPIC_API_KEY.
# ---------------------------------------------------------------------------


def run_claude(prompt: str, model: str, use_mcp: bool) -> list[dict]:
    cmd = [
        "claude",
        "-p",
        prompt,
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
            ALLOWED_TOOLS,
        ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
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


def run_question_headless(model: str, question: str) -> dict:
    events = run_claude(question, model, use_mcp=True)
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
    n = len(TESTING_DATA)

    for quest in TESTING_DATA:
        answer = run_question_headless(args.model, quest["question"])
        print(f"\n{quest['question']} DONE")
        judge_prompt = build_judge_prompt(answer)
        print("\nRating Prompt", judge_prompt)
        scores, judge_cost = judge_headless(
            args.judge_model, judge_prompt, args.judge_samples
        )
        rating = build_rating(answer, quest["expected_calls"], scores, judge_cost)
        print(rating)
        total_model_cost += rating["model_cost_usd"]
        total_judge_cost += rating["judge_cost_usd"]
        total_tool_score += rating["tool_score"]
        total_quality_score += rating["quality_score"]

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
    n = len(TESTING_DATA)

    async with (
        stdio_client(params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        tools = await session.list_tools()
        anthropic_tools = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in tools.tools
        ]

        for quest in TESTING_DATA:
            answer = await run_question_api(
                client, session, anthropic_tools, args.model, quest["question"]
            )
            print(f"\n{quest['question']} DONE")
            judge_prompt = build_judge_prompt(answer)
            print("\nRating Prompt", judge_prompt)
            scores, judge_cost = judge_api(
                client, args.judge_model, judge_prompt, args.judge_samples
            )
            rating = build_rating(answer, quest["expected_calls"], scores, judge_cost)
            print(rating)
            total_model_cost += rating["model_cost_usd"]
            total_judge_cost += rating["judge_cost_usd"]
            total_tool_score += rating["tool_score"]
            total_quality_score += rating["quality_score"]

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


def main(args) -> None:
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
    main(parser.parse_args())
