"""Compute recall@3 on DSC's 12 public queries and update EVAL.md.

Run from the repo root with: uv run python evaluate.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bayyina.bm25 import BM25Index
from bayyina.chunking import chunk_text
from bayyina.embedder import HashingEmbedder
from bayyina.hybrid import HybridRetriever


def replace_section(content: str, name: str, replacement: str) -> str:
    begin, end = f"<!-- BEGIN_{name} -->", f"<!-- END_{name} -->"
    if begin not in content or end not in content:
        return content
    prefix, rest = content.split(begin, 1)
    _, suffix = rest.split(end, 1)
    return prefix + begin + "\n" + replacement.rstrip() + "\n" + end + suffix


def main() -> None:
    files = sorted(Path("data/contracts").glob("*.txt"))
    queries = [json.loads(line) for line in Path("data/queries.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    assert len(files) == 8, f"Expected 8 contracts, found {len(files)}"
    assert len(queries) == 12, f"Expected 12 queries, found {len(queries)}"
    chunks = []
    for file in files:
        chunks += chunk_text(file.stem, file.read_text(encoding="utf-8"),
                             max_chars=500, overlap=50)
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    sparse = BM25Index()
    sparse.add(chunks)
    embedder = HashingEmbedder()
    matrix = embedder.embed([c.text for c in chunks])
    hybrid = HybridRetriever(chunks, embedder)

    misses: dict[str, list[str]] = {name: [] for name in ("BM25 only", "Dense only (HashingEmbedder)", "Hybrid (RRF)")}
    for query in queries:
        question = query["query"]
        bm25_ids = [cid for cid, _ in sparse.search(question, top_k=3)]
        qv = embedder.embed([question])[0]
        similarities = np.asarray(matrix) @ np.asarray(qv)
        indices = sorted(range(len(chunks)), key=lambda i:
                         (-float(similarities[i]), chunks[i].chunk_id))[:3]
        dense_ids = [chunks[i].chunk_id for i in indices]
        hybrid_ids = [hit.chunk_id for hit in hybrid.search(question, top_k=3)]
        for name, ids in (("BM25 only", bm25_ids),
                          ("Dense only (HashingEmbedder)", dense_ids),
                          ("Hybrid (RRF)", hybrid_ids)):
            found = any(by_id[cid].doc_id == r["doc_id"]
                        and by_id[cid].start < r["end"]
                        and r["start"] < by_id[cid].end
                        for cid in ids for r in query["relevant"])
            if not found:
                misses[name].append(query["qid"])

    lines = [
        "# Retrieval evaluation", "",
        "Setup: all 8 synthetic contracts in `data/contracts/`, "
        "`chunk_text(..., max_chars=500, overlap=50)`, `HashingEmbedder()`, "
        "and all 12 queries from `data/queries.jsonl`. "
        "A hit is relevant when the returned span overlaps a labeled span in the same document.",
        "", "| Retriever | recall@3 | Queries missed |",
        "| --- | ---: | --- |",
    ]
    for name, missing in misses.items():
        found = len(queries) - len(missing)
        recall = found / len(queries)
        print(f"{name}: {found}/{len(queries)} = {recall:.3f}; missed: {', '.join(missing) or 'none'}")
        lines.append(f"| {name} | {found}/{len(queries)} = {recall:.3f} | "
                     f"{', '.join(missing) if missing else 'None'} |")
    lines.extend(["", "All scores were computed locally using the provided offline embedder; "
                  "they are public-query results, not hidden grader scores.", ""])
    Path("EVAL.md").write_text("\n".join(lines), encoding="utf-8")

    usage = Path("AI_USAGE.md")
    if usage.exists():
        content = usage.read_text(encoding="utf-8")
        recall_table = "\n".join(lines[4:9]) + "\n"
        content = replace_section(content, "RECALL", recall_table)
        b, d, h = (len(queries) - len(misses[name]) for name in
                   ("BM25 only", "Dense only (HashingEmbedder)", "Hybrid (RRF)"))
        if h > max(b, d):
            why = "Hybrid won on the public set because the BM25 and character-ngram lists complemented each other under RRF."
        elif h == max(b, d):
            why = "Hybrid tied the strongest individual retriever on the public set; lexical and character-ngram signals did not improve top-3 coverage further."
        else:
            why = "Hybrid underperformed the best individual retriever on the public set; reciprocal-rank fusion can promote a distracting passage when the two lists disagree."
        content = replace_section(content, "INTERPRETATION", why)
        usage.write_text(content, encoding="utf-8")
        print("Updated AI_USAGE.md")
    print("Updated EVAL.md")


if __name__ == "__main__":
    main()
