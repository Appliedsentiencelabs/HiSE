# Embedding vector space: variables and interpretation

This note defines the quantities used for HiSE's current embedding-space experiments. It distinguishes BGE's content representation from structural metadata and from SGAM's attention bias.

## 1. Content embedding

For a text chunk \(x_i\), the BGE encoder produces contextual token states:

\[
H_i = f_\theta(x_i) \in \mathbb{R}^{L_i \times d}
\]

where:

- \(f_\theta\): the pretrained BGE encoder with parameters \(\theta\)
- \(L_i\): token count after tokenization, including special tokens
- \(d\): hidden/embedding dimension; for `BAAI/bge-base-en-v1.5`, \(d=768\)
- \(h_{i,t} \in \mathbb{R}^{768}\): hidden vector for token position \(t\)

For the BGE base model's standard sentence-transformer representation, use CLS pooling:

\[
v_i = h_{i,0} \in \mathbb{R}^{768}
\]

and L2 normalization:

\[
z_i = \frac{v_i}{\max(\lVert v_i \rVert_2,\epsilon)}
\]

Here \(v_i\) is the raw pooled vector, \(z_i\) is its unit-length version, and \(\epsilon\) prevents division by zero. For normalized vectors, cosine similarity is the dot product:

\[
\operatorname{cos}(z_i,z_j)=z_i^\top z_j \in [-1,1]
\]

Do not mix pooling rules silently. Mean pooling,
\[
v_i^{\text{mean}} =
\frac{\sum_t m_{i,t}h_{i,t}}{\max(\sum_t m_{i,t},1)},
\]
uses the attention mask \(m_{i,t}\) and is a different representation from CLS pooling. A result must state which one it used.

## 2. Query-to-candidate comparison

For query \(q\), positive chunk \(p\), and structural negative \(n\):

- \(z_q,z_p,z_n\): normalized vectors under the same encoder, pooling, and normalization rule
- \(s_p = z_q^\top z_p\): query/positive cosine
- \(s_n = z_q^\top z_n\): query/negative cosine
- \(\Delta = s_p-s_n\): pairwise margin

The pair is ranked correctly when \(\Delta > 0\). A tie (\(\Delta=0\)) is not evidence of a correct ranking. For a set of \(N\) pairs, pairwise accuracy is the fraction with positive margin; report ties separately. Mean margin describes separation magnitude but is not itself a retrieval metric such as Recall@k or nDCG.

## 3. Structural address variables

The prototype represents structural coordinates as:

\[
a_{i,t}=(m_{i,t},d_{i,t},p_{i,t},s_{i,t})
\]

- \(m\): token position in the current sequence/window
- \(d\): depth in the document's hierarchy
- \(p\): compact parent-path identity (categorical identity, not a meaningful scalar distance)
- \(s\): sibling ordinal within the parent

Important: these coordinates are not BGE embedding dimensions. In particular, a parent ID is categorical; subtracting arbitrary hashed IDs and treating the result as semantic distance is invalid. IDs should be compacted within the document/window before use in float tensors. Fragment/chunk order is separate metadata, not the sibling ordinal.

## 4. SGAM versus vector injection

The token-level SGAM prototype computes a pairwise additive attention bias from address differences:

\[
B_{t,u} =
w_d\frac{|d_t-d_u|}{D_{\max}}
+w_p\mathbf{1}[p_t\ne p_u]
+w_s\frac{|s_t-s_u|}{S_{\max}}
\]

The weights \(w_d,w_p,w_s\) are learned scalars, initialized to zero for the identity-at-init behavior. \(B\) is applied to attention logits before softmax; it is not directly added to the final 768-dimensional sentence vector. The sign and scaling determine whether an address relation raises or lowers attention.

The separate `HiSEEmbeddingInjector` instead implements a vector-space residual of the form:

\[
\hat v_i = v_i + W_{\text{struct}}\phi(a_i)
\]

where \(\phi\) maps an address to a structural feature vector and \(W_{\text{struct}}\) maps that feature to the embedding dimension. This injector is a distinct design path. Its randomly initialized output is not evidence of useful learned structure; it needs training and controlled evaluation before interpretation.

## 5. Variables to record in every experiment

| Category | Variables |
|---|---|
| Model | model ID, immutable revision/snapshot, tokenizer revision, framework versions |
| Input | document/chunk ID, exact text, section path, token IDs/count, truncation status |
| Structure | \(m,d,p,s\), path-to-compact-ID mapping, window/chunk membership |
| Representation | pooling rule, raw \(v\), normalized \(z\), dimension \(d=768\), normalization epsilon |
| Similarity | query/candidate cosine scores, margin \(\Delta\), tie tolerance |
| SGAM | shared/per-layer mode, \(w_d,w_p,w_s\), initialization, effective attention path |
| Evaluation | fixed query set, positive, structural negative, arm, pairwise accuracy, ties, confidence interval |
| Reproducibility | random seed, device, code commit, data version |

Do not interpret individual embedding coordinates as named semantic features. Dimensions are learned jointly; rotation of the coordinate basis can preserve all cosine relationships while changing each coordinate's apparent meaning.

## 6. Current small vector-space probe

Run from the repository root:

```powershell
.\hise\.venv\Scripts\python.exe .\hise\scripts\analyze_embedding_vector_space.py
```

The script uses the locally cached `BAAI/bge-base-en-v1.5` snapshot and generates:

- ten handwritten query/positive/structural-negative pairs
- plain candidate text versus candidate text prefixed with its section path
- cosine scores and margins for each arm
- a cosine heatmap, 2D PCA visualization, margin chart, and raw CLS norm chart
- JSON raw values, a compressed NPZ archive containing actual vectors, and a Markdown summary under `Documentation/reports/vector_space/`

The candidate body is intentionally identical within each pair so the plain-text arm is a no-structure control and should tie. This is a mechanism demonstration, not a representative benchmark and not a HiSE retrieval win. The PCA coordinates are visual aids only.
