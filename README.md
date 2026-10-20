# HiSE

Hierarchical Position Encoding for Structure-Sensitive Semantic Dense Retrieval.

HiSE investigates whether dense retrieval can represent not only what information says, but also where that information exists in structured data. The project targets tabular data, JSON, HTML/XML, Markdown, and complex nested documents.

## Research Idea

Current embedding models primarily encode linear sequence position through Rotary Position Embedding (RoPE). HiSE introduces HiSA (Hierarchical Structural Addressing), a composite address containing sequence position, structural depth, parent identity, and sibling index:

`(m, d, p, s)`

The address is combined through multiple rotary components and embedded into the model representation. A POMDP-based structural state estimator is planned to estimate high-probability structural state without exhaustively computing every structural relationship.

## Repository Layout

- `hise/` - active implementation prototype
- `hise/src/hise/` - structural addresses, parsers, and multi-dimensional RoPE
- `hise/tests/` - executable research checks
- `hise/data/` - reproducible examples and future benchmark data
- `hise/configs/` - laptop and server experiment configurations
- `Dcoumentation/` - project reports and reference material
- `Project_discussion/` - project planning and review material
- `RoPE/` - original RoPE learning prototype

## Quick Start

From the repository root:

```powershell
python -m pytest .\hise\tests
python -m hise demo
python -m hise eval
python .\hise\scripts\run_structural_demo.py
```

The package can also be imported directly from the repository root via ``import hise`` and ``import hise.config``.

## Active implementation notebooks

- `notebooks/HiSA_PreAttention_Implementation.ipynb` — pre-attention HiSA prototype and math stack
- `notebooks/02_parsers.ipynb` — Markdown, JSON, HTML, and table parsing to `(m, d, p, s)`
- `notebooks/03_sgam_injection.ipynb` — SGAM bias and gradient-flow validation
- `notebooks/04_data_pipeline.ipynb` — synthetic corpus and hard-negative triplet generation

The prototype currently validates the structural-address mechanism. It does not claim measured retrieval improvement until the Nomic embedding baseline and HiSE comparison experiments are completed.

## Hardware

- Development laptop: NVIDIA RTX 4070 Laptop GPU, 8 GB VRAM
- Research server: NVIDIA RTX 5070 Ti, 16 GB VRAM

The laptop configuration is intended for parser, encoding, and small-batch experiments. The server configuration is intended for larger benchmark runs and model experiments.

## Architecture decisions (reconciled)

The prototype now follows a consistent architecture:

- The structural signal operates on token hidden states, not on pooled `(B, 768)` embeddings.
- SGAM is computed over token-level structural addresses inside the attention matrix before softmax.
- The default initialization is near-zero weights with warmup, not a hard negative bias from the start.
- The trainable set is the frozen BGE encoder plus the HiSA/SGAM structural layer, with full fine-tuning treated as a later ablation.
- The query must carry explicit or section-context structural intent so Stage 3 negatives are actually discriminative.

This design is documented in [Dcoumentation/reports/HiSE_Architecture_Decisions.md](Dcoumentation/reports/HiSE_Architecture_Decisions.md).

## Project Status

The first milestone is complete: a unified structural-address schema, format-specific parser prototypes, a multi-dimensional rotary encoder, adversarial examples, and automated tests. The next milestone is a real token-level SGAM integration into the BGE attention stack, followed by structural reranking and controlled evaluation.

## License

Released under the MIT License. See [LICENSE](LICENSE).