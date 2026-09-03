# HiSE Prototype

HiSE is a structure-aware embedding and retrieval research prototype. The initial implementation tests one claim:

> Similar content should receive different structural representations when it occupies different roles in data.

## Layout

- `hise/src/hise/evaluation.py` - baseline retrieval scoring and false-positive checks
- `hise/scripts/run_glory_experiment.py` - logged Nomic experiment against Glory

## Run

From this directory:

```powershell
python -m pytest
python scripts/run_structural_demo.py
python scripts/run_embedding_test.py
python scripts/run_glory_experiment.py
```

The prototype currently validates the structural-address mechanism and provides an embedding-service benchmark. `run_glory_experiment.py` uses the Glory Ollama Nomic service and writes JSONL results to `outputs/glory_nomic_experiment.jsonl`. It also records whether the `asl-chat` service is reachable. The current `asl-chat` server is a language model service and must not be treated as an embedding model.
