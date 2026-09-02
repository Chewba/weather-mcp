import pytest

from weather_mcp.rag import embeddings


@pytest.fixture(autouse=True)
def reset_model_singleton():
    embeddings._model = None
    yield
    embeddings._model = None


class FakeModel:
    def __init__(self):
        self.encode_calls = []

    def encode(self, texts):
        self.encode_calls.append(texts)
        return [[0.1, 0.2, 0.3] for _ in texts]


def test_encode_vectors_returns_model_output(monkeypatch):
    fake_model = FakeModel()
    monkeypatch.setattr(embeddings, "SentenceTransformer", lambda name: fake_model)

    result = embeddings.encode_vectors(["hello", "world"])

    assert result == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    assert fake_model.encode_calls == [["hello", "world"]]


def test_get_model_is_a_singleton(monkeypatch):
    """Regression test: encode_vectors previously called `_get_model.encode(...)`
    (missing call parens on the function itself) instead of `_get_model().encode(...)`.
    """
    construction_calls = []

    def fake_constructor(name):
        construction_calls.append(name)
        return FakeModel()

    monkeypatch.setattr(embeddings, "SentenceTransformer", fake_constructor)

    embeddings.encode_vectors(["first call"])
    embeddings.encode_vectors(["second call"])

    assert construction_calls == ["all-MiniLM-L6-v2"]
