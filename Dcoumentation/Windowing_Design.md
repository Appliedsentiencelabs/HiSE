# HiSE Windowing Design

**Status:** proposed protocol for the first retrieval pilot; not yet implemented or empirically validated.

## Goal and unit

The encoder input is one document-local window containing up to three consecutive retrieval chunks, with a hard maximum of 512 BGE tokenizer tokens including model special tokens. Chunks remain the retrieval and evaluation units. A window supplies neighboring structural context to SGAM; it is not itself a retrieval result.

This is a deliberately small first protocol, not a guarantee that SGAM is active in every window. Same-section fragments share a structural address, so windows containing only fragments from one section may have no structural contrast. The builder must report the fraction of windows with at least two distinct parent IDs and, separately, the fraction with at least two distinct `(depth, parent_id, sibling_index)` addresses. If either prevalence is low, report that limitation or revise packing using training documents only; do not silently cherry-pick windows or tune packing on evaluation queries.

## Building retrieval chunks

1. Parse the source into ordered content units while retaining the document ID, complete section path, depth, and sibling order. Headings establish paths but are not retrieval chunks by themselves.
2. Form chunks from contiguous body content under the same section path. Prefer paragraph boundaries; merge short adjacent paragraphs only when their path is identical.
3. Target at most 160 BGE body tokens per chunk, measured without model special tokens. If a paragraph exceeds the target, split at sentence boundaries where possible; otherwise split at tokenizer token boundaries. Every fragment retains the original section path and receives a stable fragment index. Do not drop or duplicate source text.
4. Keep empty headings and non-text layout metadata out of the model input, but retain them in parser metadata if needed to reconstruct paths.

The 160-token target is a packing aid, not the window limit. Tokenizer behavior and separators vary, so the final window must always be checked after concatenation.

## Packing windows and handling edges

- Pack consecutive chunks from one document in source order, greedily, up to three chunks per window.
- Join chunk text with a blank line. Tokenize the complete joined text using the BGE tokenizer with its normal model special tokens. The resulting `input_ids` length, including `[CLS]` and `[SEP]`, must be at most 512.
- If adding a chunk would exceed 512, close the current window and start the next at that chunk. Never rely on silent tokenizer truncation. A single chunk that still exceeds the limit must be split again and repacked.
- Begin each window at the next unassigned chunk. After packing, if the window contains at least two chunks and more chunks remain, start the next window at the last chunk in the current window. This gives a one-chunk overlap even when the token budget closes a window before it reaches three chunks. If a window contains only one chunk, advance to the following chunk to avoid repeating the same oversized chunk indefinitely. The final window may contain two chunks; it must not be padded with chunks from another document.
- A document with two chunks produces one two-chunk window. A document with one chunk produces one single-chunk window; SGAM has no inter-chunk structural relation to express in that window.
- Do not add cross-document overlap. Keep window IDs and chunk IDs so overlap does not cause duplicate retrieval labels or duplicate examples in evaluation.

If variable token lengths make a three-chunk pack exceed the budget, close it early. The hard token limit takes precedence over the preferred chunk count. Windows with only one chunk are valid only when the document has no adjacent chunk that can be included without violating the limit.

## Structural addresses and token alignment

For each input token, assign the address `(m, d, p, s)`:

- `m`: token position in the current encoded window.
- `d`: depth of the structural section/item represented by the chunk.
- `p`: compact ID of that item's parent path, namespaced by document ID.
- `s`: stable sibling ordinal of the section/item among children of `p`. All fragments and retrieval chunks belonging to the same structural item inherit the same `(d, p, s)`.

`s` is not a fragment counter. Keep fragment index, chunk ID, and source offsets as separate metadata. This prevents long-section fragments from being mistaken for distinct sibling sections. A sibling section shares its parent's `p` with the other sibling sections and differs by `s`; a nested item uses a different parent-path ID and depth.

Use the tokenizer's offset mapping against the concatenated text to assign content tokens to their source chunk. Separators and `[CLS]`/`[SEP]` receive a neutral address and are excluded from structural evaluation metrics. Padding is masked and likewise excluded. If an offset overlaps a boundary, assign it only when it belongs unambiguously to one chunk; otherwise treat it as neutral and record the count.

Parent IDs must remain distinct after conversion to the tensor dtype used by SGAM. Do not put arbitrary 64-bit hashes directly into float32 address tensors: compact IDs to a collision-free integer range within the document/window before tensor conversion, or use an integer representation and compare IDs before casting. Persist the original path-to-ID map in metadata.

The current parsers and `build_token_level_addresses` helper do not yet implement this complete chunk/window pipeline. In particular, an integration must preserve per-token chunk boundaries, carry section sibling ordinals separately from fragment IDs, and compact parent IDs rather than assigning one address to an entire concatenated string.

## Model outputs and retrieval labels

The encoder consumes the whole window so attention can use all included chunk relations. The retrieval target remains the original chunk: pool only contextualized token states aligned to that chunk. Never label the entire window as the positive when the query identifies only one chunk. Overlapping windows are context views, not new positive documents.

BGE is normally evaluated with CLS pooling, so chunk-token masked mean pooling is a real pooling change and must not be conflated with the effect of window context or SGAM. The pilot must include:

1. Standalone chunk text encoded with the stock BGE CLS pooling baseline.
2. Standalone chunk text with the exact chunk-token pooling rule intended for contextual windows, to isolate pooling choice.
3. Windowed chunk-token pooling with SGAM attached but all three weights fixed at zero, to measure neighbor-context effects with no structural bias.

All arms use identical chunk text and splits, and record the pooling rule. Compare these first before interpreting nonzero SGAM results.

For the initial pilot, freeze and record the chunking output once. All comparison arms must use identical chunk text, IDs, window membership, tokenization, and train/evaluation split. The plain-text arm uses the same chunk text without structural changes; the prefix arm adds the recorded section path to the chunk text; the windowed zero-SGAM arm uses the same text and token alignment without that prefix.

Queries are encoded as ordinary text in every arm. Structural cues such as "in the conclusion" are therefore carried by the query wording, not by query-side structural addresses. A2 examples must make the intended section explicit enough to distinguish the positive from its structural negative.

## Chunk-boundary separators

The locally cached BGE tokenizer was checked with representative text. Concatenating two sentences, adding one space, or inserting a blank line produced the same WordPiece IDs; blank-line whitespace added no boundary token. The offset mappings still reflected the character offsets in the concatenated string. Inserting the literal `[SEP]` produced an extra special-token ID.

For this pilot, join chunks with a blank line and do **not** insert `[SEP]` between chunks. Use the original character spans and tokenizer offsets for alignment; whitespace is not a structural marker. The model's normal outer `[CLS]` and `[SEP]` remain present. If a later experiment adds internal separators, treat that as a separate input-format ablation and apply it consistently to its controls.

## Required checks before A2 evaluation

For each prepared document, record and validate:

- every source text span is represented exactly once in its document's chunk sequence;
- every encoded window is at most 512 tokens including special tokens;
- no window contains chunks from multiple documents;
- every content token maps to exactly one chunk and its expected section path;
- the share of windows with at least two distinct parent IDs and the share with at least two distinct structural addresses `(d, p, s)` are reported over the complete frozen corpus;
- special, separator, and padding tokens are neutral and excluded from structural metrics;
- overlapping chunks retain one canonical chunk ID and are not duplicated in retrieval scoring;
- parent IDs are distinct after conversion to the model's address representation.

The five short, manually authored windows in the component report test whether a bias changes attention. They do not validate this document-level chunking protocol or retrieval effectiveness.
