# AI usage — Bayyina W4

## Tools and purpose

- **ChatGPT (GPT-6):** reviewed the written task specification, drafted the five retrieval modules, and helped construct random property checks for offsets/overlap, BM25-reference comparisons, citations, and the evaluation script.
- **Python and pytest:** used to run local verification against a recreated copy of the supplied type/embedder interfaces. The official DSC public tests must also be run in the cloned repository before submission.
- **Git:** used for version control when the implementation is committed and pushed.

## A chunking-rule failure and how it was identified

A naive overlap prototype used `next_start = previous_end - overlap` without checking word boundaries. For `alpha bravo charlie delta echo foxtrot` with `max_chars=20` and `overlap=4`, it cuts the first chunk after `charlie` (end 19) but restarts at offset 15, in the middle of `charlie`. The previous chunk did **not** end in the middle of a word, so this violates rule 6. The boundary property check identified the bad start. The final implementation now looks for a valid word start inside the allowed overlap after a normal boundary cut; it only permits a mid-word start after a mid-word cut. Source offsets are always taken from the original document.

## Public-query recall@3

<!-- BEGIN_RECALL -->
| Retriever | recall@3 | Queries missed |
| --- | ---: | --- |
| BM25 only | 11/12 = 0.917 | q12 |
| Dense only (HashingEmbedder) | 11/12 = 0.917 | q12 |
| Hybrid (RRF) | 11/12 = 0.917 | q12 |
<!-- END_RECALL -->

## Interpretation

<!-- BEGIN_INTERPRETATION -->
Hybrid tied the strongest individual retriever on the public set; lexical and character-ngram signals did not improve top-3 coverage further.
<!-- END_INTERPRETATION -->

The public scores are not substitutes for the hidden-grader score.
