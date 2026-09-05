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
    raw_parts = discussion['raw_product_text'].split("$$\n\n")
    # The product's forecaster-attribution/signature block (e.g.
    # "AVIATION...Rorke") always follows the LAST "$$" marker -- it isn't
    # real content, but matches the section-header regex closely enough to
    # get parsed as one, mislabeling garbage as real AVIATION/MARINE/etc.
    # sections. Some offices also use "$$" as a mid-document separator
    # between two real content sections (a real DISCUSSION section has been
    # seen sitting between two "$$" markers), so only the final part is
    # dropped -- everything before it is real content.
    body_parts = raw_parts[:-1] if len(raw_parts) > 1 else raw_parts
    header = body_parts[0].split("&&\n\n")
    chunks = []
    front_header = header[0].split("\n.")
    chunk = CHUNK_BASE.copy()
    chunk['chunk_type'] = "HEADER"
    chunk['chunk_text'] = front_header[0]
    chunks.append(chunk)
    sections = front_header[1:]
    sections.extend(header[1:])
    for extra_part in body_parts[1:]:
        sections.extend(extra_part.split("&&\n\n"))
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
    # (?:\s*\([^)]*\))? tolerates a parenthetical qualifier between the
    # section name and "..." (e.g. "SHORT TERM (TDY-TUE)..."), the same
    # class of bug already fixed once for slash-delimited qualifiers
    # (e.g. "AVIATION /12Z SUNDAY THROUGH THURSDAY/...") below -- without
    # it, the whole section silently matched nothing and its real content
    # was dropped entirely, not just mislabeled.
    regex = re.compile(
        r"\n?\.?(?:([A-Z1-9 ]+)(?:\s*\([^)]*\))?(?:\s*/[^/\n]*/)?\.\.\.|(\d+)\))\s*(.*?)"
        r"(?=\n?\.?(?:[A-Z1-9 ]+(?:\s*\([^)]*\))?(?:\s*/[^/\n]*/)?\.\.\.|\d+\))|\Z)",
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
    # into the first real sub-item instead of leaving it as an orphan chunk
    # -- but only when that next item is a bare numbered item ("1)") with no
    # name of its own. Two distinct NAMED sections can land in the same call
    # (e.g. "SHORT TERM (TDY-TUE)...text...LONG TERM (WED-SAT)...text" with
    # no "&&" between them) -- folding those together would silently merge
    # SHORT TERM's content into LONG TERM's chunk and discard its own label.
    if len(chunks) > 1 and texts[1][0].strip() == "":
        chunks[1]['chunk_text'] = chunks[0]['chunk_text'] + chunks[1]['chunk_text']
        chunks.pop(0)
    return chunks
