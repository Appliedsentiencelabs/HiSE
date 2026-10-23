# HiSE Prototype

HiSE is a structure-aware embedding and retrieval research prototype. The initial implementation tests one claim:

> Similar content should receive different structural representations when it occupies different roles in data.

## Layout

- `hise/src/hise/address.py` - structural address schema and parent stability
- `hise/src/hise/rope.py` - multi-dimensional RoPE encoder
- `hise/src/hise/injection.py` - learned HiSE embedding injection and SGAM bias helpers
- `hise/src/hise/evaluation.py` - baseline retrieval scoring and false-positive checks
- `hise/scripts/run_glory_experiment.py` - logged Nomic experiment against Glory

## Structural injection

The prototype now includes a lightweight injection layer that turns a chunk embedding `e` into a structure-aware vector using a learned projection from the structural address `(m, d, p, s)`:

```python
from hise.address import StructuralAddress
from hise.injection import HiSEEmbeddingInjector

injector = HiSEEmbeddingInjector(embedding_dim=768)
vector = injector(base_embedding, StructuralAddress(...))
```

This is the clean version of the Review 2 architecture: keep the base embedding model frozen, learn only the structural projection, and evaluate the vector-space separation on structurally adversarial pairs.

## Run

From this directory or the repository root:

```powershell
python -m pytest
python -m hise demo
python -m hise eval
python scripts/run_structural_demo.py
python scripts/run_embedding_test.py
python scripts/run_glory_experiment.py
```

The package can be imported directly from the repo root via ``import hise`` and ``import hise.config``.

## Real BGE / SGAM validation

From the repository root, run the end-to-end test against the cached `BAAI/bge-base-en-v1.5` checkpoint:

```powershell
.\hise\.venv\Scripts\python.exe .\hise\scripts\run_system_validation.py
```

The test runs the package test suite and notebooks, then checks the real BGE attention hook, zero-weight equivalence, a five-window attention-shift gate, SGAM gradients, and 512-token GPU latency/memory at batch sizes 1, 8, and 32. It requires CUDA and a locally cached BGE model; it does not silently download the model. Markdown and JSON evidence are written to `Dcoumentation/reports/HiSE_System_Validation_Report.md` and `.json`.

For the stricter component evidence report, run:

```powershell
.\hise\.venv\Scripts\python.exe .\hise\scripts\run_component_validation.py
```

This also records per-notebook kernel executables, checks hook activation after a fresh model reload and in train mode, reports per-sentence zero-weight equivalence and per-layer attention-shift metrics with shuffled-label controls, executes shared/per-layer SGAM optimizer steps, and compares stock-SDPA, stock-eager, and hooked GPU overhead. It requires CUDA and the locally cached model. The generated `Dcoumentation/reports/evidence_report.json` and `evidence_report.md` are component-validation evidence only; retrieval quality and training convergence are not tested.

## Embedding vector space

The variable definitions and interpretation guide are in `../Documentation/Embedding_Vector_Space.md`. To run the small local BGE probe and generate vector-space plots plus raw embedding arrays:

```powershell
.\hise\.venv\Scripts\python.exe .\hise\scripts\analyze_embedding_vector_space.py
```

It compares plain candidate text with section-path-prefixed candidate text on ten handcrafted structural pairs. This is an exploratory mechanism probe, not a held-out retrieval benchmark or trained HiSE/SGAM result. Outputs are written to `../Documentation/reports/vector_space/`.

## Current notebook sequence

- `notebooks/HiSA_PreAttention_Implementation.ipynb` — pre-attention HiSA prototype and math stack
- `notebooks/02_parsers.ipynb` — document-to-structure parsing
- `notebooks/03_sgam_injection.ipynb` — SGAM bias prototype and invariant checks
- `notebooks/04_data_pipeline.ipynb` — triplet generation and BM25-style negative mining

The prototype currently validates the structural-address mechanism and provides an embedding-service benchmark. `run_glory_experiment.py` uses the Glory Ollama Nomic service and writes JSONL results to `outputs/glory_nomic_experiment.jsonl`. It also records whether the `asl-chat` service is reachable. The current `asl-chat` server is a language model service and must not be treated as an embedding model.
