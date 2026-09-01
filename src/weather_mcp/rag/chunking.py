import re

from weather_mcp.rag.embeddings import encode_vectors


def parse_chunks(discussion:dict, office:dict) -> list:
    CHUNK_BASE = {
        "product_id" : discussion["product_id"],
        "source" : discussion["source"],
        "issuing_office": discussion["issuing_office"],
        "office_longitude": office["longitude"],
        "office_latitude": office["latitude"],
        "issued_at" : discussion['issuance_time'],
        "subsection" : None,
        "valid_from" : None,
        "valid_to" : None,
        "topics" : None,
    }
    first_chunk = discussion['raw_product_text'].split("$$\n\n")
    header = first_chunk[0].split("&&\n\n")
    chunks = []
    front_header = header[0].split("\n.")
    chunk = CHUNK_BASE.copy()
    chunk['chunk_type'] = "HEADER"
    chunk['chunk_text'] = front_header[0]
    chunks.append(chunk)
    sections = front_header[1:]
    sections.extend(header[1:])
    if len(first_chunk) > 1:
        sections.extend(first_chunk[1].split("&&\n\n"))
    for sub_section in sections:
        chunks.extend(_build_section_chunks(CHUNK_BASE.copy(), sub_section))
    chunk_order = 0
    texts = []
    for chunk in chunks:
        chunk['chunk_order'] = chunk_order
        chunk_order = chunk_order + 1
        texts.append(chunk['chunk_text'])
    embedding = encode_vectors(texts)
    for i in range(chunk_order):
        chunks[i]['embedding'] = embedding[i]
    return chunks

def _build_section_chunks(base:dict, chunk_text:str) -> list[dict]:
    APPROVED_SECTIONS = ['HEADER','KEY_MESSAGES', 'DISCUSSION', 'AVIATION', 'MARINE']
    regex = re.compile(
        r"\n?\.?(?:([A-Z1-9 ]+)(?:\s*/[^/\n]*/)?\.\.\.|(\d+)\))\s*(.*?)"
        r"(?=\n?\.?(?:[A-Z1-9 ]+(?:\s*/[^/\n]*/)?\.\.\.|\d+\))|\Z)",
        re.DOTALL,
    )
    chunks = []
    texts = regex.findall(chunk_text)
    chunk_base = base.copy()
    if len(texts) > 0:
        name = texts[0][0].strip().replace(" ", "_")
        if name not in APPROVED_SECTIONS:
            name = "OTHER"
        chunk_base['chunk_type'] = name
    for text in texts:
        chunk = chunk_base.copy()
        chunk['subsection'] = text[0].strip()
        if len(chunk['subsection']) == 0:
            chunk['subsection'] = text[1]
        chunk['chunk_text'] = text[2]
        chunks.append(chunk)
    # Fold the section's own intro line (e.g. "As of 130 AM Sunday...")
    # into the first real sub-item instead of leaving it as an orphan chunk.
    if len(chunks) > 1:
        chunks[1]['chunk_text'] = chunks[0]['chunk_text'] + chunks[1]['chunk_text']
        chunks.pop(0)
    return chunks
