# HiSE System Support Bundle

This bundle consolidates the engineering evidence for the HiSE prototype: the structural-address model, SGAM attention bias, parser pipeline, evaluation logic, and the mounting plan for the next full training cycle.

## 1. System thesis and implementation summary

HiSE is a structure-aware retrieval system that keeps a frozen semantic backbone and adds a structural signal on top. The working representation is the structural address `(m, d, p, s)`:

- `m`: sequence position
- `d`: document depth
- `p`: parent identity
- `s`: sibling index

This is implemented in the project as a unified structural-address abstraction and used as the basis for the subsequent HiSA and SGAM layers.

## 2. Evidence map

### A. Structural address foundation

Claim: the system models structure as a stable 4-tuple, not as ad hoc string prefixes.

Evidence:
- `hise/src/hise/address.py` defines `StructuralAddress` and stable parent IDs.
- `hise/tests/test_address.py` checks address stability and uniqueness.
- `hise/tests/test_demo.py` demonstrates adversarial same-text/different-structure pairs.

### B. SGAM attention bias is implemented and validated

Claim: the system adds a learnable structural attention bias before softmax and enforces the expected invariants.

Evidence:
- `hise/src/hise/injection.py` defines `SGAMBias` and `sgam_bias`.
- The formula is: `SGAM(i, j) = w_d * |d_i - d_j| + w_p * 1[parent_i != parent_j] + w_s * |s_i - s_j|`.
- `hise/tests/test_sgam.py` includes checks for:
  - zero address -> zero bias
  - same structure -> zero penalty
  - symmetry
  - boundedness
  - learnable gradients
- `hise/src/hise/evaluation.py` adds `evaluate_sgam()` for structured system diagnostics.
- `python -m hise eval` returns a green SGAM validation payload.

### C. Parser and document-to-structure pipeline already exist

Claim: the pipeline converts raw documents into structural addresses, which is required before training/evaluation.

Evidence:
- `notebooks/02_parsers.ipynb` implements the parser sequence.
- `hise/tests/test_parsers.py` validates parser output contracts.
- The project design expects parsing output to produce `(m, d, p, s)` tuples.

### D. Pre-attention HiSA / RoPE stack is implemented

Claim: the system uses a multi-dimensional structural positional encoding approach before attention.

Evidence:
- `hise/src/hise/config.py` defines the HiSA configuration and subspace decomposition.
- `hise/src/hise/encoding/` contains path encoding, subspace logic, and the direct-sum encoder.
- `hise/tests/test_pre_attention_degenerate.py`, `test_pre_attention_collision.py`, and `test_pre_attention_invertibility.py` validate layer invariants.
- `notebooks/HiSA_PreAttention_Implementation.ipynb` demonstrates the pre-attention flow.

### E. Real evaluation and external embedding experiment pipeline are active

Claim: the system is not only symbolic; it includes a working content-vs-structure retrieval experiment path against an external embedding service.

Evidence:
- `hise/scripts/run_glory_experiment.py` performs an Ollama/Nomic embedding call and reads/writes JSONL logs.
- It checks baseline content-only retrieval against a path-prefixed control.
- It logs both embedding latency and false-positive outcomes.

### F. Repo validation is green

Claim: the current structural core is stable and executable.

Evidence:
- `pytest -q hise/tests` passes successfully.
- `python -m hise eval` returns a positive structure-validation result.

## 3. Architecture graph

```mermaid
flowchart TD
    A[Raw document / HTML / JSON / Markdown] --> B[Parser layer]
    B --> C[(m, d, p, s) structural addresses]
    C --> D[HiSA pre-attention encoder]
    D --> E[Base embedding model backbone]
    E --> F[SGAM attention bias]
    F --> G[Structure-aware retrieval score]
    G --> H[Evaluation / benchmark / ablations]

    I[World graph / knowledge priors] --> F
    J[Hard negatives / triplets] --> H
    K[StructEval-Bench] --> H
```

## 4. Engineering interpretation

The system is intentionally designed around a simple but strong principle:

- the semantic backbone remains the frozen representation engine
- the structural signal is injected as a controllable additive prior
- the evaluation focuses on whether structure changes ranking when text is held nearly constant

This is the correct direction for a paper-ready prototype because it directly measures the novel claim: structure matters independently of lexical semantics.

## 5. What is already supported

The following are already supported by code and tests in the repository:

- unified structural-address abstraction
- direct-sum HiSA / RoPE-style structural encoding
- structural collision and invertibility checks
- SGAM bias with negative learned weights
- symmetry, boundedness, and gradient checks
- parser pipeline contract for structural outputs
- evaluation route for content-only vs structure-aware ranking
- repo-level validation with automated tests

## 6. What remains for the final research system

The remaining work is the next-stage build-out, not a redesign:

- full BGE attention-hook integration
- curriculum training with InfoNCE and hard negatives
- StructEval-Bench v1 dataset construction
- SGAM ablation studies
- final benchmark table and paper-ready metrics

## 7. Evidence summary

The project has evidence in code, notebooks, and execution logs that the following are true:

- the structural representation is real and consistent
- the SGAM mechanism is implemented and passes invariant tests
- the architecture is aligned with the intended system design
- the repository is a credible engineering foundation for the next research phase

This is enough to support the actual implementation plan and the evaluation story that follows it.
