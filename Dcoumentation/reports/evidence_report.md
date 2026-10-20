# Component validation

- Model: `BAAI/bge-base-en-v1.5` @ `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`
- Device: NVIDIA GeForce RTX 4070 Laptop GPU; CUDA 11.8
- Python: `C:\Users\Admin\RESEARCH\HiSE\hise\.venv\Scripts\python.exe` (3.12.7 | packaged by Anaconda, Inc. | (main, Oct  4 2024, 13:17:27) [MSC v.1929 64 bit (AMD64)])
- Packages: torch 2.7.1+cu118, Transformers 4.51.3, NumPy 1.26.4, pandas 2.2.2
- Git: `51d77da28a3b9b7c7a7c8e559224ae21471d791d`; seed: 20261021

## Tests and notebooks

- Pytest: 29 passed, 0 failed, 0 skipped
- Notebooks: 4 passed, 0 failed

| Notebook | Result | Kernel executable |
|---|---|---|
| 02_parsers.ipynb | PASS | `C:\Users\Admin\RESEARCH\HiSE\hise\.venv\Scripts\python.exe` |
| 03_sgam_injection.ipynb | PASS | `C:\Users\Admin\RESEARCH\HiSE\hise\.venv\Scripts\python.exe` |
| 04_data_pipeline.ipynb | PASS | `C:\Users\Admin\RESEARCH\HiSE\hise\.venv\Scripts\python.exe` |
| HiSA_PreAttention_Implementation.ipynb | PASS | `C:\Users\Admin\RESEARCH\HiSE\hise\.venv\Scripts\python.exe` |

## Hook and equivalence

- Default: `BertSdpaSelfAttention` / sdpa; after attach: `BertSelfAttention` / `eager`
- Hook calls: 12 first load; 12 after reload (expected 12 each)
- Train-mode attention dropout calls: 12 (expected 12); same-seed stock-vs-hook max diff: 0; different-seed stock output diff: 1.8205464
- 20 padded sentences, shape `[20, 16]`
- Non-pad hidden-state max-abs-diff min/mean/max: 1.1920929e-06 / 1.8060207e-06 / 2.4586916e-06
- CLS cosine min/mean/max: 0.999999881 / 0.999999982 / 1.000000000
- Mean-pool cosine min/mean/max: 0.999999940 / 1.000000006 / 1.000000119

## Attention shift

Tokenizer offsets map actual text spans to sections. The metric excludes special and padded tokens and averages same-section attention mass over valid queries, heads, and layers.

| Parent weight | Window | Zero mass | Biased mass | Gain | CLS cosine |
|---:|---:|---:|---:|---:|---:|
| -0.5 | 1 | 0.244935 | 0.304843 | 0.059908 | 0.995787 |
| -0.5 | 2 | 0.252796 | 0.314577 | 0.061781 | 0.995554 |
| -0.5 | 3 | 0.257115 | 0.319910 | 0.062795 | 0.996329 |
| -0.5 | 4 | 0.254254 | 0.315779 | 0.061526 | 0.996307 |
| -0.5 | 5 | 0.236023 | 0.293330 | 0.057307 | 0.996694 |
| -1.0 | 1 | 0.244935 | 0.369967 | 0.125031 | 0.981659 |
| -1.0 | 2 | 0.252796 | 0.382195 | 0.129399 | 0.981360 |
| -1.0 | 3 | 0.257115 | 0.389782 | 0.132667 | 0.980874 |
| -1.0 | 4 | 0.254254 | 0.384998 | 0.130744 | 0.983645 |
| -1.0 | 5 | 0.236023 | 0.356527 | 0.120504 | 0.986315 |
| -2.0 | 1 | 0.244935 | 0.501111 | 0.256176 | 0.912108 |
| -2.0 | 2 | 0.252796 | 0.517849 | 0.265052 | 0.909223 |
| -2.0 | 3 | 0.257115 | 0.534932 | 0.277817 | 0.893888 |
| -2.0 | 4 | 0.254254 | 0.528316 | 0.274062 | 0.918969 |
| -2.0 | 5 | 0.236023 | 0.488466 | 0.252443 | 0.946508 |

- Shuffled labels true-section gains: `[0.034103567401568086, 0.035147978613774, 0.032320693135261536, 0.04070675621430081, 0.0377104543149471]`; max 0.040707.
- Zero-control maximum mass difference: 0.

Per-layer same-section mass (12 values per window):

- Weight -0.5:
  - Window 1: `[0.413115, 0.309318, 0.402238, 0.348837, 0.227313, 0.227054, 0.222078, 0.270061, 0.244127, 0.285613, 0.353089, 0.355272]`
  - Window 2: `[0.413994, 0.307193, 0.414833, 0.362207, 0.245891, 0.23947, 0.234189, 0.303982, 0.250166, 0.272675, 0.359133, 0.371192]`
  - Window 3: `[0.424318, 0.326334, 0.395869, 0.368002, 0.271335, 0.242659, 0.267206, 0.319547, 0.264249, 0.300071, 0.325049, 0.334284]`
  - Window 4: `[0.427563, 0.341124, 0.400683, 0.375574, 0.253064, 0.249177, 0.257364, 0.302238, 0.233569, 0.268302, 0.328, 0.352695]`
  - Window 5: `[0.402079, 0.258474, 0.387279, 0.340596, 0.199754, 0.223596, 0.237243, 0.282607, 0.245014, 0.249247, 0.332881, 0.361195]`
- Weight -1.0:
  - Window 1: `[0.511759, 0.382447, 0.452807, 0.417905, 0.272055, 0.264109, 0.259458, 0.313374, 0.306722, 0.358688, 0.453383, 0.446895]`
  - Window 2: `[0.513156, 0.378362, 0.471985, 0.43382, 0.293763, 0.278445, 0.273023, 0.341218, 0.316776, 0.351339, 0.461129, 0.473328]`
  - Window 3: `[0.522364, 0.399801, 0.452413, 0.440746, 0.325044, 0.284869, 0.306446, 0.364544, 0.330073, 0.388408, 0.429625, 0.433051]`
  - Window 4: `[0.5251, 0.418632, 0.457714, 0.450713, 0.302373, 0.290597, 0.297878, 0.346246, 0.30041, 0.347144, 0.431074, 0.45209]`
  - Window 5: `[0.500713, 0.323304, 0.442191, 0.410064, 0.239172, 0.26422, 0.275908, 0.315715, 0.299263, 0.320327, 0.42924, 0.458211]`
- Weight -2.0:
  - Window 1: `[0.699638, 0.54491, 0.562179, 0.5666, 0.373483, 0.34354, 0.342241, 0.395629, 0.432571, 0.490017, 0.644661, 0.617863]`
  - Window 2: `[0.701045, 0.535697, 0.593486, 0.590291, 0.402229, 0.357934, 0.349989, 0.404141, 0.449607, 0.497448, 0.661859, 0.670456]`
  - Window 3: `[0.707385, 0.561862, 0.577515, 0.599942, 0.445139, 0.376092, 0.387507, 0.445087, 0.465508, 0.571884, 0.647614, 0.633647]`
  - Window 4: `[0.709817, 0.587285, 0.582541, 0.612265, 0.417383, 0.377345, 0.379562, 0.427962, 0.445652, 0.507571, 0.650511, 0.6419]`
  - Window 5: `[0.688948, 0.473589, 0.556726, 0.561238, 0.331643, 0.353992, 0.359394, 0.382983, 0.4101, 0.472821, 0.635926, 0.634236]`

## Gradients and optimizer step

| Mode | Params | Nonzero grads | Encoder frozen | Step executed | Loss |
|---|---:|---:|---|---|---:|
| shared | 3 | 3/3 | False | True | 0.33493385 |
| per_layer | 36 | 36/36 | False | True | 0.33526513 |

Per-parameter values and gradients:

### shared

| Layer | Parameter | Before | Gradient | After step |
|---:|---|---:|---:|---:|
| shared | depth_weight | -0.2 | -0.00029967056 | -0.19999701 |
| shared | parent_weight | -0.2 | 0.00012785103 | -0.20000128 |
| shared | sibling_weight | -0.2 | 0.00109343 | -0.20001094 |
### per_layer

| Layer | Parameter | Before | Gradient | After step |
|---:|---|---:|---:|---:|
| 0 | depth_weight | -0.2 | -3.6570787e-06 | -0.19999997 |
| 0 | parent_weight | -0.2 | 0.00024291223 | -0.20000243 |
| 0 | sibling_weight | -0.2 | -2.2803841e-05 | -0.19999978 |
| 1 | depth_weight | -0.2 | 3.0623487e-06 | -0.20000003 |
| 1 | parent_weight | -0.2 | 0.00013545997 | -0.20000136 |
| 1 | sibling_weight | -0.2 | -0.00018428557 | -0.19999816 |
| 2 | depth_weight | -0.2 | -1.5609865e-05 | -0.19999985 |
| 2 | parent_weight | -0.2 | 0.00035774539 | -0.20000358 |
| 2 | sibling_weight | -0.2 | -6.8041911e-05 | -0.19999932 |
| 3 | depth_weight | -0.2 | 5.7026991e-06 | -0.20000006 |
| 3 | parent_weight | -0.2 | -8.9259003e-05 | -0.19999911 |
| 3 | sibling_weight | -0.2 | -4.6240457e-05 | -0.19999954 |
| 4 | depth_weight | -0.2 | -2.1646116e-05 | -0.19999978 |
| 4 | parent_weight | -0.2 | -0.00073947967 | -0.19999261 |
| 4 | sibling_weight | -0.2 | 8.7082917e-06 | -0.20000009 |
| 5 | depth_weight | -0.2 | 7.5068136e-05 | -0.20000075 |
| 5 | parent_weight | -0.2 | 0.00040203531 | -0.20000403 |
| 5 | sibling_weight | -0.2 | 0.00011433663 | -0.20000115 |
| 6 | depth_weight | -0.2 | -0.00012549484 | -0.19999875 |
| 6 | parent_weight | -0.2 | -0.00069644296 | -0.19999304 |
| 6 | sibling_weight | -0.2 | 6.4413267e-05 | -0.20000064 |
| 7 | depth_weight | -0.2 | 0.00012101775 | -0.20000121 |
| 7 | parent_weight | -0.2 | 0.0019529872 | -0.20001954 |
| 7 | sibling_weight | -0.2 | 0.00063922215 | -0.2000064 |
| 8 | depth_weight | -0.2 | -0.00011047855 | -0.1999989 |
| 8 | parent_weight | -0.2 | 0.00028473116 | -0.20000285 |
| 8 | sibling_weight | -0.2 | 0.00022725662 | -0.20000228 |
| 9 | depth_weight | -0.2 | 7.9227721e-05 | -0.20000079 |
| 9 | parent_weight | -0.2 | 0.00263518 | -0.20002635 |
| 9 | sibling_weight | -0.2 | 0.00020796788 | -0.20000209 |
| 10 | depth_weight | -0.2 | -0.00011882008 | -0.19999881 |
| 10 | parent_weight | -0.2 | -0.00054587651 | -0.19999455 |
| 10 | sibling_weight | -0.2 | -5.5063254e-05 | -0.19999945 |
| 11 | depth_weight | -0.2 | -0.00020155005 | -0.19999799 |
| 11 | parent_weight | -0.2 | -0.0023716893 | -0.19997628 |
| 11 | sibling_weight | -0.2 | -0.00028772894 | -0.19999713 |

## GPU overhead: 512 tokens, 5 warmups and 20 timed forwards

| Arm | Batch | Median ms | P95 ms | Peak allocated MB | Peak reserved MB |
|---|---:|---:|---:|---:|---:|
| stock_sdpa | 1 | 11.2297 | 12.2149 | 652.20 | 712.00 |
| stock_sdpa | 8 | 82.6080 | 83.4882 | 766.25 | 904.00 |
| stock_sdpa | 32 | 334.9388 | 336.3477 | 1162.44 | 1576.00 |
| stock_eager | 1 | 12.1435 | 12.4732 | 672.20 | 716.00 |
| stock_eager | 8 | 113.4804 | 114.4738 | 910.26 | 968.00 |
| stock_eager | 32 | 469.3918 | 486.3151 | 1738.50 | 1880.00 |
| hooked | 1 | 20.6437 | 37.6019 | 653.17 | 722.00 |
| hooked | 8 | 140.0469 | 154.6127 | 898.24 | 1058.00 |
| hooked | 32 | 595.2403 | 617.4266 | 1750.48 | 2210.00 |

## Gates

- `unit_tests`: **PASS**
- `notebooks`: **PASS**
- `hook_all_layers_first_and_reload`: **PASS**
- `train_mode_dropout_preserved`: **PASS**
- `zero_weight_equivalence`: **PASS**
- `attention_shift`: **PASS**
- `shuffled_control`: **FAIL**
- `zero_weight_control`: **PASS**
- `shared_gradient_step`: **FAIL**
- `per_layer_gradient_step`: **FAIL**
- `three_arm_overhead_recorded`: **PASS**

## Not tested

- Retrieval quality (MRR/nDCG) on a held-out structure-aware benchmark.
- Training convergence beyond the reported one-step dummy-loss gradient check.
- Cross-domain transfer, production throughput, and energy consumption.
- The five attention windows are manually authored synthetic examples, not retrieval evidence.

Raw results: `evidence_report.json`
