import datetime

import pytest

from weather_mcp.rag import chunking


@pytest.fixture(autouse=True)
def fake_encode_vectors(monkeypatch):
    def _fake(texts):
        return [f"vec{i}" for i in range(len(texts))]

    monkeypatch.setattr(chunking, "encode_vectors", _fake)


def test_build_section_chunks_merges_intro_into_first_item():
    """The section's own intro line (e.g. "As of 854 AM...") should be folded
    into the first numbered sub-item rather than left as an orphan chunk."""
    text = (
        "KEY MESSAGES...\n\n"
        "As of 854 AM, hot conditions continue across the area.\n\n"
        "1) Heat index values will approach 105 this afternoon.\n\n"
        "2) Little relief is expected overnight.\n\n"
    )

    chunks = chunking._build_section_chunks({}, text)

    assert len(chunks) == 2
    assert chunks[0]["chunk_type"] == "KEY_MESSAGES"
    assert chunks[0]["subsection"] == "1"
    assert "As of 854 AM" in chunks[0]["chunk_text"]
    assert "Heat index values" in chunks[0]["chunk_text"]
    assert chunks[1]["subsection"] == "2"
    assert "Little relief" in chunks[1]["chunk_text"]


def test_build_section_chunks_handles_aviation_qualifier():
    """Regression test: AVIATION headers carry a qualifier like
    "/12Z SUNDAY THROUGH THURSDAY/" before the "..." marker. The old regex
    required the character before "..." to be in [A-Z1-9 ], so this whole
    section was silently dropped."""
    text = "AVIATION /12Z SUNDAY THROUGH THURSDAY/...\n\nVFR conditions expected throughout the period.\n\n"

    chunks = chunking._build_section_chunks({}, text)

    assert len(chunks) == 1
    assert chunks[0]["chunk_type"] == "AVIATION"
    assert chunks[0]["subsection"] == "AVIATION"
    assert "VFR conditions" in chunks[0]["chunk_text"]


def test_build_section_chunks_falls_back_to_other():
    text = "SYNOPSIS...\n\nHigh pressure will dominate the region.\n\n"

    chunks = chunking._build_section_chunks({}, text)

    assert len(chunks) == 1
    assert chunks[0]["chunk_type"] == "OTHER"
    assert chunks[0]["subsection"] == "SYNOPSIS"


def test_build_section_chunks_empty_text_returns_no_chunks():
    assert chunking._build_section_chunks({}, "") == []


def test_parse_chunks_builds_header_and_section_chunks_in_order():
    issuance_time = datetime.datetime(2026, 8, 31, 12, 54, tzinfo=datetime.UTC)
    discussion = {
        "product_id": 42,
        "source": "live_capture",
        "issuing_office": "KRAH",
        "issuance_time": issuance_time,
        "raw_product_text": (
            "AREA FORECAST DISCUSSION\n"
            "National Weather Service Testville\n"
            "854 AM EDT Sun Aug 31 2026\n"
            ".KEY MESSAGES...\n\n"
            "As of 854 AM, hot conditions continue.\n\n"
            "1) Heat index values will approach 105.\n\n"
            "&&\n\n"
            ".DISCUSSION...\n\n"
            "As of the previous forecast, little has changed.\n\n"
            ".NEAR TERM...\n\n"
            "Expect continued heat through the evening.\n\n"
            "$$\n\n"
        ),
    }
    office = {"longitude": -78.6382, "latitude": 35.7796}

    chunks = chunking.parse_chunks(discussion, office)

    assert len(chunks) == 3

    header = chunks[0]
    assert header["chunk_type"] == "HEADER"
    assert header["chunk_order"] == 0
    assert "AREA FORECAST DISCUSSION" in header["chunk_text"]
    assert header["office_latitude"] == 35.7796
    assert header["office_longitude"] == -78.6382
    assert header["issued_at"] == issuance_time
    assert header["product_id"] == 42

    key_messages = chunks[1]
    assert key_messages["chunk_type"] == "KEY_MESSAGES"
    assert key_messages["chunk_order"] == 1
    assert "As of 854 AM" in key_messages["chunk_text"]
    assert "Heat index values" in key_messages["chunk_text"]

    discussion_chunk = chunks[2]
    assert discussion_chunk["chunk_type"] == "DISCUSSION"
    assert discussion_chunk["chunk_order"] == 2
    assert discussion_chunk["subsection"] == "NEAR TERM"
    assert "As of the previous forecast" in discussion_chunk["chunk_text"]
    assert "Expect continued heat" in discussion_chunk["chunk_text"]

    # embeddings assigned in the same order as chunk_order
    assert [c["embedding"] for c in chunks] == ["vec0", "vec1", "vec2"]
