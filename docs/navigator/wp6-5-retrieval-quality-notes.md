# WP6.5 Retrieval Quality Notes

WP6.5 repairs the governed retrieval evaluation logic after live testing showed a false negative on:

- `how is fallback different from final model truth`

What was actually happening:

- live retrieval already returned a strong fallback-truth chunk at rank 1
- live retrieval also returned the fallback-vs-runtime drift chunk at ranks 2 and 3
- the validator failed because it required one single chunk to satisfy multiple governed signals that are intentionally distributed across separate chunks

What changed:

- retrieval evaluation now supports grouped governed expectations across the top-k result set
- the fallback-vs-final-truth query now requires:
  - at least one fallback-truth result
  - at least one fallback-vs-runtime drift result

Why this is a governed fix:

- it does not weaken the requirement to something trivial
- it matches how the retrieval corpus is intentionally structured
- it preserves the distinction between canonical truth chunks and drift chunks
