# HiSE Architecture Decisions

This document records the design decisions that fix the prototype before building more training infrastructure.

## A1. Encoding unit: multi-chunk window over token sequences

We use a token-level window of up to 512 tokens as the structural unit on which SGAM is computed. In a chunked retrieval system, each chunk is still a logical unit, but the SGAM bias is only non-zero when multiple tokens from the same document window share a structural relation. This is where SGAM(i, j) != 0 happens.

For the first pilot, the concrete window packing proposal is documented in [Windowing Design](../Windowing_Design.md): up to three consecutive chunks, tokenizer-counted to a 512-token maximum including special tokens, with one-chunk overlap. This is a proposed protocol, not implemented evidence; it must pass the listed alignment checks before building A2 evaluation pairs.

The key point is that a single chunk on its own is structurally homogeneous. If all tokens inside a chunk have the same `(d, p, s)`, then SGAM(i, j) = 0 for every pair in that chunk. To get a non-zero signal, we need at least two tokens or chunks in the same window that differ in structural address or belong to structurally related sections. Example: a heading token at depth 2 and a paragraph token at the same parent at depth 3 produce a non-zero `w_depth` and `w_parent` term; nearby sibling items produce a non-zero `w_sibling` term.

This is why the correct unit is a multi-chunk window or document segment, not an isolated chunk vector.

## A2. Defining "correct structural position" in a query

We define structural intent as an explicit part of the retrieval query. The query carries either:

- an explicit position cue, such as "in the conclusion", "under Method A", or "at depth 2"; or
- a section-title context, such as "the risks section" or "the discussion under the benchmark heading".

The query is not allowed to be a verbatim copy of a positive chunk sentence, because that would allow a lexical shortcut. The system should distinguish a positive and a structural negative from the query alone.

Examples:

1. Query: "What does the conclusion say about model drift?"
   - Positive: conclusion paragraph
   - Structural negative: introduction paragraph with the same wording but different depth

2. Query: "Which Method B result is reported for latency?"
   - Positive: sibling section Method B
   - Structural negative: sibling section Method A with similar wording

3. Query: "What is stated under the risk section?"
   - Positive: risk section paragraph
   - Structural negative: performance section paragraph with similar vocabulary

These examples are the design basis for the Stage 3 negative set and the SDS benchmark.

## A3. Sign and initialization convention

The recommended convention is:

- initialize SGAM weights near 0 at step 0 so the base BGE model is left nearly unchanged at startup;
- apply a warmup schedule that gradually drives mismatched-structure terms negative; and
- optionally clamp or normalize depth and sibling distances before the final bias so the term remains bounded even for long documents.

This preserves model quality at the beginning of training and prevents the dead-weight failure mode. The earlier design of negative weights at initialization may be useful in a later stage, but the safer default for a real training run is 0-init + warmup, not a hard negative bias from the start.

Boundedness should be enforced algorithmically, not assumed by the weight sign alone. We will cap the maximum structural distance term, normalize by the maximum allowed depth, and enforce a finite range during the attention hook.

## A4. Trainable set

Trainable set: frozen encoder + HiSA/SGAM layer only.

This is the recommended first implementation because:

- the base BGE encoder already has strong semantics;
- the structural signal is a small, interpretable addition;
- SGAM has only 3 learned scalar weights in the shared version or 3 × 12 in the per-layer version;
- the capacity matches the intended research claim: structure-aware bias, not a full model rewrite.

LoRA or full fine-tuning can be evaluated later as ablations, but the initial system should not spend capacity on rebuilding the whole BERT stack.

## A5. Architecture: HiSA on token hidden states, not pooled embeddings

The architecture is: raw text -> tokenizer -> token hidden states -> HiSA structural encoding on each token state -> SGAM bias over token-token attention -> pooled retrieval embedding.

This is the correct design because the base BGE model is a token encoder and the attention mechanism operates on token states. A pooled `(B, 768)` embedding is not the place where SGAM is injected; SGAM must be attached to the per-token attention matrix before softmax.

This architecture is now the canonical one in code and documentation.

## Stability and repo constraints

The implementation should also respect the following constraints:

- one canonical RoPE implementation
- one canonical parser package
- no `sys.path` hacks in notebooks
- no claim of empirical improvement without dataset and benchmark numbers
- separate documentation for implemented but unvalidated vs empirically validated claims
