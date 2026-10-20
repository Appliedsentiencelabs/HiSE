# HiSE System Validation Report

- Model: `BAAI/bge-base-en-v1.5` (BertModel)
- Device: NVIDIA GeForce RTX 4070 Laptop GPU (CUDA 11.8)
- Transformers: 4.51.3; PyTorch: 2.7.1+cu118
- Default attention: sdpa / `BertSdpaSelfAttention`
- Layers: 12; hook calls: 12

## Project checks

- Unit tests: 29 passed
- Notebooks: Validated 4 notebook(s).

## Zero-weight equivalence (20 fixed sentences; padded batch)

- Batch shape: `[20, 16]`
- Maximum absolute hidden-state difference: `2.4586916e-06`
- Mean CLS cosine: `0.999999982`

| # | Tokens | Max abs diff | CLS cosine | Sentence |
|---:|---:|---:|---:|---|
| 1 | 12 | 1.385808e-06 | 1.000000000 | The company reported higher revenue during the first quarter. |
| 2 | 10 | 1.5497208e-06 | 1.000000000 | Operating margin improved as manufacturing costs declined. |
| 3 | 12 | 1.8775463e-06 | 1.000000000 | The board approved the annual budget at its meeting. |
| 4 | 11 | 1.7881393e-06 | 1.000000000 | Customer retention increased after the support program launched. |
| 5 | 11 | 1.9073486e-06 | 0.999999940 | The research team evaluated three retrieval architectures. |
| 6 | 12 | 2.0638108e-06 | 0.999999881 | A new distribution center opened near the regional market. |
| 7 | 11 | 1.4901161e-06 | 1.000000000 | The policy requires approval from the department manager. |
| 8 | 10 | 1.1920929e-06 | 1.000000000 | Energy consumption fell after the equipment upgrade. |
| 9 | 13 | 1.937151e-06 | 0.999999940 | The product roadmap includes a redesigned mobile application. |
| 10 | 12 | 1.6689301e-06 | 1.000000000 | Quarterly cash flow exceeded the forecast by five percent. |
| 11 | 11 | 2.4586916e-06 | 1.000000000 | The clinical study enrolled participants across four hospitals. |
| 12 | 10 | 1.6391277e-06 | 1.000000000 | The security review identified two configuration issues. |
| 13 | 11 | 1.5497208e-06 | 1.000000000 | The supplier will deliver replacement components next month. |
| 14 | 11 | 2.4065375e-06 | 1.000000000 | The proposal compares implementation cost and expected benefit. |
| 15 | 9 | 1.7881393e-06 | 0.999999940 | The software release includes accessibility improvements. |
| 16 | 12 | 2.0861626e-06 | 0.999999940 | The committee postponed its decision until additional data arrives. |
| 17 | 11 | 1.6689301e-06 | 1.000000000 | The warehouse recorded fewer damaged packages this week. |
| 18 | 13 | 1.847744e-06 | 1.000000000 | The contract renews automatically unless either party gives notice. |
| 19 | 12 | 1.9073486e-06 | 1.000000000 | The sales team expanded coverage in the northern region. |
| 20 | 16 | 1.9073486e-06 | 1.000000000 | The report summarizes risks, milestones, and open questions. |

## Attention-shift gate

- Metric: average fraction of attention probability on same-section tokens, averaged over valid section queries, heads, and layers
- Fixed nonzero weights: depth=0, parent=-1.0, sibling=0
- Pass threshold set before measurement: gain >= 0.10 and CLS cosine >= 0.90 on >=4/5 windows

| Window | Tokens | Within-section before | After | Gain | CLS cosine |
|---:|---:|---:|---:|---:|---:|
| 1 | 29 | 0.244935 | 0.369967 | 0.125031 | 0.981659 |
| 2 | 29 | 0.252796 | 0.382195 | 0.129399 | 0.981360 |
| 3 | 30 | 0.257115 | 0.389782 | 0.132667 | 0.980874 |
| 4 | 32 | 0.254254 | 0.384998 | 0.130744 | 0.983645 |
| 5 | 27 | 0.236023 | 0.356527 | 0.120504 | 0.986315 |

## Gradient checks

| Mode | Unique params | Expected | Nonzero gradients | Checked | Loss |
|---|---:|---:|---:|---:|---:|
| shared | 3 | 3 | 3 | 3 | 0.32649451 |
| per_layer | 36 | 36 | 36 | 36 | 0.32649451 |

Parameter-level values and gradients:

| Mode | Layer | Parameter | Value | Gradient |
|---|---:|---|---:|---:|
| shared | shared | depth_weight | -0.2 | -0.00012962104 |
| shared | shared | parent_weight | -0.2 | 0.0017393306 |
| shared | shared | sibling_weight | -0.2 | 0.00054481457 |
| per_layer | 0 | depth_weight | -0.2 | -8.8816378e-05 |
| per_layer | 0 | parent_weight | -0.2 | -0.0010769041 |
| per_layer | 0 | sibling_weight | -0.2 | -0.00021178449 |
| per_layer | 1 | depth_weight | -0.2 | 3.4842786e-05 |
| per_layer | 1 | parent_weight | -0.2 | 0.00013170153 |
| per_layer | 1 | sibling_weight | -0.2 | -9.6082455e-05 |
| per_layer | 2 | depth_weight | -0.2 | 1.7238053e-05 |
| per_layer | 2 | parent_weight | -0.2 | -0.00010249998 |
| per_layer | 2 | sibling_weight | -0.2 | -7.0637223e-05 |
| per_layer | 3 | depth_weight | -0.2 | 0.00013369892 |
| per_layer | 3 | parent_weight | -0.2 | 0.0012045769 |
| per_layer | 3 | sibling_weight | -0.2 | 7.6059092e-05 |
| per_layer | 4 | depth_weight | -0.2 | 5.0997289e-05 |
| per_layer | 4 | parent_weight | -0.2 | 0.0004784418 |
| per_layer | 4 | sibling_weight | -0.2 | 0.00017368693 |
| per_layer | 5 | depth_weight | -0.2 | 5.5370059e-05 |
| per_layer | 5 | parent_weight | -0.2 | 0.00031449948 |
| per_layer | 5 | sibling_weight | -0.2 | 0.0002028015 |
| per_layer | 6 | depth_weight | -0.2 | -1.4657689e-05 |
| per_layer | 6 | parent_weight | -0.2 | 4.5852765e-05 |
| per_layer | 6 | sibling_weight | -0.2 | 0.00034437084 |
| per_layer | 7 | depth_weight | -0.2 | 0.00017370282 |
| per_layer | 7 | parent_weight | -0.2 | 0.0020609137 |
| per_layer | 7 | sibling_weight | -0.2 | 0.00058986252 |
| per_layer | 8 | depth_weight | -0.2 | -0.00019829729 |
| per_layer | 8 | parent_weight | -0.2 | -0.00060524599 |
| per_layer | 8 | sibling_weight | -0.2 | -0.00040607672 |
| per_layer | 9 | depth_weight | -0.2 | 0.00012178207 |
| per_layer | 9 | parent_weight | -0.2 | 0.0020711059 |
| per_layer | 9 | sibling_weight | -0.2 | 0.0003322626 |
| per_layer | 10 | depth_weight | -0.2 | -0.00015123529 |
| per_layer | 10 | parent_weight | -0.2 | -0.00030695097 |
| per_layer | 10 | sibling_weight | -0.2 | -1.5145066e-05 |
| per_layer | 11 | depth_weight | -0.2 | -0.00026424642 |
| per_layer | 11 | parent_weight | -0.2 | -0.0024761604 |
| per_layer | 11 | sibling_weight | -0.2 | -0.00037450297 |

## 512-token GPU overhead (median of 3 timed forwards after warmup)

Stock uses the model's default SDPA path; hooked uses eager attention because the SGAM score bias is added before softmax. Timings include that required path change. Inputs are full-length synthetic token-id sequences; this is inference-forward overhead, not end-to-end tokenization or embedding-pooling latency.

| Batch | Stock ms | Hooked ms | Latency overhead | Stock peak alloc MB | Hooked peak alloc MB | Peak delta MB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12.187 | 15.621 | 28.18% | 445.87 | 474.42 | 28.55 |
| 8 | 81.413 | 121.686 | 49.47% | 559.92 | 719.53 | 159.61 |
| 32 | 331.257 | 525.749 | 58.71% | 956.11 | 1571.92 | 615.81 |

## Gate summary

- `hook_reached_all_layers`: **PASS**
- `zero_weight_equivalence_pass`: **PASS**
- `attention_shift_pass`: **PASS**
- `gradient_parameter_counts_pass`: **PASS**
- `gradient_nonzero_pass`: **PASS**

## Scope and limitations

The attention-shift windows are hand-authored synthetic examples and use fixed, manually selected SGAM weights; they are not evidence of retrieval improvement or trained behavior. The 512-token timing uses repeated valid token IDs and measures model forward latency and allocator peaks only. Retrieval metrics, training convergence, energy use, and throughput under a production corpus were not evaluated.

JSON detail: `HiSE_System_Validation_Report.json`
