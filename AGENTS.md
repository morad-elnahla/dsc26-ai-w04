# Bayyina W4 — agent rules

## Runtime and commands
- Target Python 3.12; keep the exact dependency pins in `pyproject.toml`.
- Set up the environment with `uv sync`.
- Run the project tests with `uv run pytest`; do not claim tests passed unless run.
- Recompute public retrieval metrics with `uv run python evaluate.py`.
- Review changes to `EVAL.md` and `AI_USAGE.md` after running the evaluator; it rewrites both.
- Do not add packages, search libraries, FAISS, or network-dependent setup.

## Project map and data
- Retrieval code is in `bayyina/`; public tests are in `tests/test_public.py`.
- Synthetic source contracts live in `data/contracts/`; public queries and labeled spans live in `data/queries.jsonl`.
- Use synthetic contracts only; never introduce real contracts, names, or personal data.
- Chunk offsets are character offsets into the original, unmodified document; preserve them through retrieval and citations.
- Read `README.md` for the full normalization, chunking, BM25, RRF, and citation contracts instead of copying the specification here.
- Do not edit `bayyina/types.py` or `bayyina/embedder.py`; DSC supplies these interfaces and replaces them during grading.

## Retrieval and evaluation rules
- Normalize and tokenize through `bayyina.normalize.normalize_ar` and `tokenize`; do not create a second tokenizer.
- BM25 is implemented in-repo; keep the specified IDF/TF formula, unique query terms, and score-then-chunk-ID ordering.
- Hybrid retrieval combines BM25 and dense rankings with RRF; preserve dense candidates even when BM25 returns no match.
- Citation checks must use original-document offsets and whitespace-collapsed quote matching, as specified in `README.md`.
- The public evaluator uses all 8 synthetic contracts, 12 labeled queries, 500-character chunks, 50-character overlap, and `HashingEmbedder()`.
- The approved embedder for this offline W4 evaluation is `HashingEmbedder(dim=512, n=3)`; do not substitute a downloaded model in the graded path.
- `intfloat/multilingual-e5-small` is mentioned only as a possible future swap in the embedder docstring, not as a required or pinned model.

## Prompts, models, and safety
- This W4 repository implements retrieval and citation logic; the required evaluation does not call a hosted LLM or use a production prompt template.
- Do not invent an approved hosted model, API, prompt directory, or model version that is not documented in the repository.
- If a future LLM prompt is added, keep its source under `prompts/` and add tests for its inputs and expected behavior.
- Never read, print, commit, or include API keys, tokens, credentials, or local `.env` contents in prompts or logs.

## Completion checks
- For chunking changes, verify exact source slices, full non-whitespace coverage, overlap bounds, and legal cut points.
- For retrieval changes, use the public evaluator and report public-query results separately from hidden-grader results.
- For citation changes, test invalid boundaries, blank quotes, and whitespace-collapsed matching.
- Preserve deterministic tie-breaking by plain string chunk-ID order.
