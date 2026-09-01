from sentence_transformers import SentenceTransformer

_model : SentenceTransformer | None = None

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def encode_vectors(texts: list)-> list:
    return _get_model().encode(texts)
