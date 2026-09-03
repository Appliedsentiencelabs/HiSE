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
python .\hise\scripts\run_structural_demo.py
```

The prototype currently validates the structural-address mechanism. It does not claim measured retrieval improvement until the Nomic embedding baseline and HiSE comparison experiments are completed.

## Hardware

- Development laptop: NVIDIA RTX 4070 Laptop GPU, 8 GB VRAM
- Research server: NVIDIA RTX 5070 Ti, 16 GB VRAM

The laptop configuration is intended for parser, encoding, and small-batch experiments. The server configuration is intended for larger benchmark runs and model experiments.

## Project Status

The first milestone is complete: a unified structural-address schema, format-specific parser prototypes, a multi-dimensional rotary encoder, adversarial examples, and automated tests. The next milestone is baseline retrieval with the locally available Nomic embedding model, followed by structural reranking and controlled evaluation.

## License

Released under the MIT License. See [LICENSE](LICENSE).