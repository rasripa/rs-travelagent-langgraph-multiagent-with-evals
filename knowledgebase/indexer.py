"""Knowledge-base index build and retrieval for the orchestrator's KB tool.

The markdown files in this directory are chunked by heading, embedded with Titan
Text Embeddings v2, and cached to .kb_index.json. Retrieval is cosine
similarity in process -- there is no vector database, because at five documents
a database would be ceremony.

Retrieval degrades rather than fails: if Bedrock embeddings are unavailable
(no credentials, no model access, offline), the index falls back to a lexical
scorer so the demo still answers policy questions. The fallback is reported in
the index metadata so you always know which mode you are in.

The retrieval interface is deliberately the same shape Bedrock Knowledge Bases
returns -- a ranked list of {text, source, score} -- so swapping the local
index for a managed Knowledge Base later does not change the agent code.

    .venv/bin/python knowledgebase/indexer.py          # build
    .venv/bin/python knowledgebase/indexer.py "query"  # build + test a query
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import AWS_REGION, KB_DIR, KB_INDEX_PATH, embed_model_id  # noqa: E402

# Chunking: split on level-2 headings. These documents are short and
# heading-scoped, so a heading is a natural semantic unit and keeps the cited
# source human-meaningful ("railcards.md > Two Together Railcard").
HEADING = re.compile(r"^##\s+(.+)$", re.MULTILINE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
    "from", "has", "have", "how", "i", "if", "in", "is", "it", "my", "no", "not",
    "of", "on", "or", "that", "the", "to", "was", "what", "when", "which", "will",
    "with", "you", "your",
}


# ---------------------------------------------------------------- chunking
def _chunk_markdown(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    title_match = re.match(r"^#\s+(.+)$", raw, re.MULTILINE)
    doc_title = title_match.group(1).strip() if title_match else path.stem

    matches = list(HEADING.finditer(raw))
    chunks: list[dict[str, Any]] = []

    if not matches:
        chunks.append({"section": doc_title, "text": raw.strip()})
    else:
        preamble = raw[: matches[0].start()].strip()
        if len(preamble) > 120:
            chunks.append({"section": f"{doc_title} (overview)", "text": preamble})
        for i, match in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
            body = raw[match.start():end].strip()
            if body:
                chunks.append({"section": match.group(1).strip(), "text": body})

    return [
        {
            "chunk_id": f"{path.stem}#{i}",
            "source": path.name,
            "doc_title": doc_title,
            "section": chunk["section"],
            "text": chunk["text"],
        }
        for i, chunk in enumerate(chunks)
    ]


def collect_chunks() -> list[dict[str, Any]]:
    docs = sorted(p for p in KB_DIR.glob("*.md"))
    chunks: list[dict[str, Any]] = []
    for doc in docs:
        chunks.extend(_chunk_markdown(doc))
    return chunks


# ---------------------------------------------------------------- embeddings
def _embed(texts: list[str]) -> list[list[float]] | None:
    """Embed with Titan v2. Returns None if Bedrock is unreachable."""
    try:
        import boto3

        client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
        vectors = []
        for text in texts:
            response = client.invoke_model(
                modelId=embed_model_id(),
                body=json.dumps({"inputText": text[:8000]}),
            )
            vectors.append(json.loads(response["body"].read())["embedding"])
        return vectors
    except Exception as exc:  # noqa: BLE001
        print(f"  embeddings unavailable ({type(exc).__name__}: {exc})", file=sys.stderr)
        print("  falling back to lexical retrieval", file=sys.stderr)
        return None


# ---------------------------------------------------------------- lexical fallback
def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def _lexical_vector(text: str) -> dict[str, float]:
    counts: dict[str, float] = {}
    for token in _tokens(text):
        counts[token] = counts.get(token, 0.0) + 1.0
    norm = math.sqrt(sum(v * v for v in counts.values())) or 1.0
    return {k: v / norm for k, v in counts.items()}


def _lexical_score(query: dict[str, float], doc: dict[str, float]) -> float:
    if len(query) > len(doc):
        query, doc = doc, query
    return sum(weight * doc.get(term, 0.0) for term, weight in query.items())


# ---------------------------------------------------------------- build
def build_index() -> dict[str, Any]:
    chunks = collect_chunks()
    print(f"chunked {len({c['source'] for c in chunks})} documents "
          f"into {len(chunks)} sections")

    embeddable = [f"{c['doc_title']}. {c['section']}. {c['text']}" for c in chunks]
    vectors = _embed(embeddable)
    mode = "titan-embeddings" if vectors else "lexical"

    for i, chunk in enumerate(chunks):
        if vectors:
            chunk["embedding"] = vectors[i]
        else:
            chunk["lexical"] = _lexical_vector(embeddable[i])

    index = {
        "mode": mode,
        "embed_model_id": embed_model_id() if vectors else None,
        "region": AWS_REGION if vectors else None,
        "chunk_count": len(chunks),
        "sources": sorted({c["source"] for c in chunks}),
        "chunks": chunks,
    }
    KB_INDEX_PATH.write_text(json.dumps(index), encoding="utf-8")
    print(f"wrote {KB_INDEX_PATH.name}  mode={mode}  chunks={len(chunks)}")
    return index


# ---------------------------------------------------------------- retrieval
_CACHE: dict[str, Any] | None = None


def load_index() -> dict[str, Any]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    if not KB_INDEX_PATH.exists():
        _CACHE = build_index()
    else:
        _CACHE = json.loads(KB_INDEX_PATH.read_text(encoding="utf-8"))
    return _CACHE


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def search(query: str, top_k: int = 4) -> list[dict[str, Any]]:
    """Return the top_k most relevant KB sections for a query.

    Shape matches a Bedrock Knowledge Base retrieve() result closely enough
    that swapping implementations does not change the calling agent.
    """
    index = load_index()
    chunks = index["chunks"]

    if index["mode"] == "titan-embeddings":
        query_vector = _embed([query])
        if query_vector:
            scored = [
                (_cosine(query_vector[0], c["embedding"]), c) for c in chunks
            ]
        else:
            # Index was built with embeddings but Bedrock is down right now.
            qv = _lexical_vector(query)
            scored = [
                (_lexical_score(qv, _lexical_vector(c["text"])), c) for c in chunks
            ]
    else:
        qv = _lexical_vector(query)
        scored = [(_lexical_score(qv, c["lexical"]), c) for c in chunks]

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [
        {
            "text": chunk["text"],
            "source": chunk["source"],
            "section": chunk["section"],
            "score": round(score, 4),
        }
        for score, chunk in scored[:top_k]
        if score > 0
    ]


if __name__ == "__main__":
    build_index()
    if len(sys.argv) > 1:
        probe = " ".join(sys.argv[1:])
        print(f"\nquery: {probe}\n" + "-" * 62)
        for hit in search(probe):
            print(f"[{hit['score']}] {hit['source']} > {hit['section']}")
            print("    " + hit["text"].replace("\n", " ")[:220] + "...\n")
