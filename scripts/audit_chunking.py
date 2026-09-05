"""One-off quality audit of the deterministic AFD chunking logic
(rag/chunking.py's parse_chunks), not a recurring eval-harness check.

Runs the real chunking code directly against a sample of scripts/test_data.json
-- no DB, no network, since chunking is pure regex over already-downloaded
text. Two layers of checking:

1. Free deterministic checks (coverage, tiny/empty chunks, OTHER-type
   fraction, in-discussion duplicate text) -- catch obvious regex problems
   for $0.
2. One bounded LLM judge call per sampled discussion (full raw text + the
   resulting chunks, single 0-10 rating + a short critique) -- NOT averaged
   over multiple samples like quality_score elsewhere in this project,
   because this is a one-time audit, not a repeated benchmark that needs
   variance smoothed out. Runs via the `claude -p` CLI (billed against the
   subscription, same as the eval harness's headless backend), not the
   Anthropic SDK directly -- ANTHROPIC_API_KEY in this environment isn't
   authorized for direct API billing.

Deliberately not wired into run_eval.py or CI: chunking logic doesn't change
between eval runs, so there's nothing to gain from re-running this on a
schedule -- run it by hand after touching chunking.py.

Usage:
    uv run python scripts/audit_chunking.py [--sample-size 8]
"""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

from weather_mcp.rag.chunking import parse_chunks
from weather_mcp.rag.ingest import _parse_afd_feed

JUDGE_MODEL = "claude-sonnet-5"
DUMMY_OFFICE = {"longitude": 0.0, "latitude": 0.0}
TINY_CHUNK_CHARS = 15
OTHER_FRACTION_WARN = 0.30
COVERAGE_WARN = 0.60

JUDGE_PROMPT_TEMPLATE = """You are auditing a deterministic (regex-based) text chunker for a RAG \
system. Below is a raw NWS Area Forecast Discussion (AFD) product, followed \
by the chunks it was split into (type / subsection / text).

Rate chunk quality from 0-10, where 10 means every chunk is a coherent, \
complete thought, correctly labeled, with nothing important dropped or \
duplicated, and 0 means the chunking is unusable (chunks cut off mid-sentence, \
sections merged together that shouldn't be, or real content missing). Then \
give a one-sentence critique of the single worst issue you see, or "none" if \
there isn't one.

Respond in exactly this format:
SCORE: <integer 0-10>
CRITIQUE: <one sentence, or "none">

--- RAW DISCUSSION ---
{raw_text}

--- RESULTING CHUNKS ---
{chunks_text}
"""


def _load_fixture() -> list[dict]:
    with open(Path(__file__).parent / "test_data.json") as f:
        return json.load(f)


def pick_sample(records: list[dict], sample_size: int) -> list[dict]:
    """Evenly spaced across text length (shortest, longest, and spread
    between) -- length extremes are where a regex chunker is most likely to
    misbehave, more so than picking a particular office."""
    by_length = sorted(records, key=lambda r: len(r["productText"]))
    if sample_size >= len(by_length):
        return by_length
    last = len(by_length) - 1
    indices = {round(i * last / (sample_size - 1)) for i in range(sample_size)}
    return [by_length[i] for i in sorted(indices)]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def run_deterministic_checks(raw_text: str, chunks: list[dict]) -> list[str]:
    issues = []

    chunk_texts = [c["chunk_text"] for c in chunks]
    coverage = sum(len(normalize(t)) for t in chunk_texts) / max(len(normalize(raw_text)), 1)
    if coverage < COVERAGE_WARN:
        issues.append(f"low text coverage ({coverage:.0%} of raw discussion made it into a chunk)")

    tiny = [c for c in chunks if len(c["chunk_text"].strip()) < TINY_CHUNK_CHARS]
    if tiny:
        issues.append(f"{len(tiny)} suspiciously tiny chunk(s) (<{TINY_CHUNK_CHARS} chars)")

    other_fraction = sum(1 for c in chunks if c["chunk_type"] == "OTHER") / max(len(chunks), 1)
    if other_fraction > OTHER_FRACTION_WARN:
        issues.append(f"{other_fraction:.0%} of chunks fell through to OTHER type")

    seen = {}
    for c in chunks:
        key = normalize(c["chunk_text"])
        if key in seen:
            issues.append(
                f"duplicate chunk text within this discussion "
                f"({seen[key]!r} and {c['chunk_type']!r} match)"
            )
        seen[key] = c["chunk_type"]

    return issues


def call_claude_headless(prompt: str, model: str) -> tuple[str, float]:
    """Single-turn, no-tools call to `claude -p`, billed against the CLI's
    subscription rather than ANTHROPIC_API_KEY -- mirrors run_eval.py's
    judge_headless. ANTHROPIC_API_KEY must be unset for this call: when set,
    the CLI errors out instead of using its own subscription login (see
    FINDINGS_RAG.md's "Headless auth conflict" entry)."""
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    result = subprocess.run(
        ["claude", "-p", "--model", model, "--output-format", "stream-json", "--verbose"],
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude CLI exited {result.returncode}: {result.stderr}")
    events = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    for event in reversed(events):
        if event.get("type") == "result":
            return event.get("result", ""), event.get("total_cost_usd", 0.0)
    return "", 0.0


def run_llm_judge(raw_text: str, chunks: list[dict]) -> tuple[int, str, float]:
    chunks_text = "\n\n".join(
        f"[{c['chunk_type']} / {c['subsection']}]\n{c['chunk_text']}" for c in chunks
    )
    prompt = JUDGE_PROMPT_TEMPLATE.format(raw_text=raw_text, chunks_text=chunks_text)
    text, cost = call_claude_headless(prompt, JUDGE_MODEL)
    score_match = re.search(r"SCORE:\s*(\d+)", text)
    critique_match = re.search(r"CRITIQUE:\s*(.+)", text)
    score = int(score_match.group(1)) if score_match else -1
    critique = critique_match.group(1).strip() if critique_match else text.strip()
    return score, critique, cost


def main(sample_size: int) -> None:
    records = _load_fixture()
    sample = pick_sample(records, sample_size)

    print(f"Auditing chunking quality on {len(sample)} of {len(records)} discussions.\n")

    total_cost = 0.0
    scores = []
    for record in sample:
        discussion_data = _parse_afd_feed(record)
        discussion_data["product_id"] = discussion_data["source_url"]
        discussion_data["source"] = "audit"
        chunks = parse_chunks(discussion_data, DUMMY_OFFICE)

        issues = run_deterministic_checks(record["productText"], chunks)
        score, critique, cost = run_llm_judge(record["productText"], chunks)
        total_cost += cost
        scores.append(score)

        print(f"{record['issuingOffice']}  ({len(record['productText'])} chars, {len(chunks)} chunks)")
        for issue in issues:
            print(f"  [deterministic] {issue}")
        if not issues:
            print("  [deterministic] no issues found")
        print(f"  [llm judge] score={score}  critique={critique}")
        print()

    valid_scores = [s for s in scores if s >= 0]
    avg = sum(valid_scores) / len(valid_scores) if valid_scores else float("nan")
    print(f"--- Summary: avg judge score {avg:.1f}/10 across {len(valid_scores)} discussions ---")
    print(
        f"Judge cost: ${total_cost:.4f} total (one-off, not a recurring run -- "
        "this is the CLI's own subscription-usage estimate, not a real "
        "per-call charge)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=8)
    main(**vars(parser.parse_args()))
