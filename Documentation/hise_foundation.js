const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, BorderStyle, WidthType, ShadingType,
  LevelFormat, Footer, SimpleField, PageBreak
} = require('docx');
const fs = require('fs');

// ── Design tokens ──────────────────────────────────────────────
const C = {
  navy:    "1A2E4A",
  blue:    "2563EB",
  steel:   "3B82F6",
  light:   "EFF6FF",
  mid:     "DBEAFE",
  gray:    "6B7280",
  lgray:   "F9FAFB",
  white:   "FFFFFF",
  dark:    "111827",
  border:  "CBD5E1",
  accent:  "0EA5E9",
};

const border  = { style: BorderStyle.SINGLE, size: 1, color: C.border };
const borders = { top: border, bottom: border, left: border, right: border };
const hBorder = { style: BorderStyle.SINGLE, size: 2, color: C.navy };
const hBorders= { top: hBorder, bottom: hBorder, left: hBorder, right: hBorder };

// ── Helpers ─────────────────────────────────────────────────────
const sp = (b=60,a=80) => ({ before:b, after:a });

function run(text, opts={}) {
  return new TextRun({ text, font:"Inter", size:22, color:C.dark, ...opts });
}
function runB(text, opts={}) {
  return run(text, { bold:true, ...opts });
}
function runI(text, opts={}) {
  return run(text, { italics:true, ...opts });
}

function h1(text) {
  return new Paragraph({
    spacing: sp(400,120),
    border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: C.blue, space:4 } },
    children: [new TextRun({ text, font:"Inter", size:36, bold:true, color:C.navy })]
  });
}
function h2(text) {
  return new Paragraph({
    spacing: sp(280,80),
    children: [new TextRun({ text, font:"Inter", size:28, bold:true, color:C.blue })]
  });
}
function h3(text) {
  return new Paragraph({
    spacing: sp(180,60),
    children: [new TextRun({ text, font:"Inter", size:24, bold:true, color:C.navy })]
  });
}
function p(...children) {
  return new Paragraph({ spacing: sp(60,80), children });
}
function pb() {
  return new Paragraph({ children: [new TextRun({ children: [new PageBreak()] })] });
}
function spacer(h=80) {
  return new Paragraph({ spacing: { before:h, after:h }, children:[run("")] });
}
function divider() {
  return new Paragraph({
    spacing: sp(140,140),
    border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: C.steel, space:1 } },
    children:[run("")]
  });
}
function bullet(children_arr) {
  return new Paragraph({
    numbering: { reference:"bullets", level:0 },
    spacing: sp(40,40),
    children: children_arr
  });
}
function subbullet(children_arr) {
  return new Paragraph({
    numbering: { reference:"subbullets", level:0 },
    spacing: sp(30,30),
    children: children_arr
  });
}

function callout(label, text, fill=C.light) {
  return new Table({
    width:{ size:9360, type:WidthType.DXA },
    columnWidths:[9360],
    rows:[new TableRow({ children:[
      new TableCell({
        borders: { top:{style:BorderStyle.SINGLE,size:6,color:C.blue}, bottom:{style:BorderStyle.SINGLE,size:1,color:C.border}, left:{style:BorderStyle.SINGLE,size:8,color:C.blue}, right:{style:BorderStyle.SINGLE,size:1,color:C.border} },
        shading:{ fill, type:ShadingType.CLEAR },
        margins:{ top:120, bottom:120, left:200, right:160 },
        children:[
          new Paragraph({ spacing:sp(0,40), children:[runB(label, {color:C.blue, size:20})] }),
          new Paragraph({ spacing:sp(0,0), children:[run(text, {size:21})] }),
        ]
      })
    ]})]
  });
}

function twoColCallout(left_label, left_text, right_label, right_text) {
  return new Table({
    width:{ size:9360, type:WidthType.DXA },
    columnWidths:[4600, 4760],
    rows:[new TableRow({ children:[
      new TableCell({
        borders:hBorders,
        shading:{ fill:C.light, type:ShadingType.CLEAR },
        margins:{ top:120, bottom:120, left:160, right:160 },
        children:[
          new Paragraph({ spacing:sp(0,40), children:[runB(left_label, {color:C.navy, size:20})] }),
          new Paragraph({ spacing:sp(0,0), children:[run(left_text, {size:21})] }),
        ]
      }),
      new TableCell({
        borders:hBorders,
        shading:{ fill:C.mid, type:ShadingType.CLEAR },
        margins:{ top:120, bottom:120, left:160, right:160 },
        children:[
          new Paragraph({ spacing:sp(0,40), children:[runB(right_label, {color:C.navy, size:20})] }),
          new Paragraph({ spacing:sp(0,0), children:[run(right_text, {size:21})] }),
        ]
      }),
    ]})]
  });
}

function makeTable(headers, rows, colWidths) {
  return new Table({
    width:{ size:colWidths.reduce((a,b)=>a+b,0), type:WidthType.DXA },
    columnWidths:colWidths,
    rows:[
      new TableRow({ tableHeader:true, children: headers.map((h,i)=>
        new TableCell({
          borders:hBorders,
          width:{ size:colWidths[i], type:WidthType.DXA },
          shading:{ fill:C.navy, type:ShadingType.CLEAR },
          margins:{ top:80, bottom:80, left:120, right:120 },
          children:[new Paragraph({ children:[runB(h,{color:C.white,size:20})] })]
        })
      )}),
      ...rows.map((row,ri)=> new TableRow({ children: row.map((cell,ci)=>
        new TableCell({
          borders,
          width:{ size:colWidths[ci], type:WidthType.DXA },
          shading:{ fill: ri%2===0 ? C.lgray : C.white, type:ShadingType.CLEAR },
          margins:{ top:80, bottom:80, left:120, right:120 },
          children:[new Paragraph({ children:[run(cell,{size:20})] })]
        })
      )}))
    ]
  });
}

// ── Document ────────────────────────────────────────────────────
const doc = new Document({
  numbering:{
    config:[
      { reference:"bullets", levels:[{ level:0, format:LevelFormat.BULLET, text:"\u2022", alignment:AlignmentType.LEFT, style:{ paragraph:{ indent:{ left:720, hanging:360 } } } }] },
      { reference:"subbullets", levels:[{ level:0, format:LevelFormat.BULLET, text:"\u25E6", alignment:AlignmentType.LEFT, style:{ paragraph:{ indent:{ left:1080, hanging:360 } } } }] },
    ]
  },
  sections:[{
    properties:{
      page:{ size:{ width:12240, height:15840 }, margin:{ top:1200, right:1260, bottom:1200, left:1260 } }
    },
    footers:{
      default: new Footer({ children:[new Paragraph({
        alignment:AlignmentType.RIGHT,
        children:[
          run("Applied Sentience Labs — Confidential  |  HiSE Research Foundation Doc  |  Page ", {size:18,color:C.gray}),
          new SimpleField("PAGE","Arabic")
        ]
      })]})
    },
    children:[

      // ── COVER ──────────────────────────────────────────────
      spacer(240),
      new Paragraph({
        alignment:AlignmentType.CENTER, spacing:sp(0,60),
        children:[new TextRun({ text:"Applied Sentience Labs Pvt. Ltd.", font:"Inter", size:22, color:C.gray, bold:true })]
      }),
      new Paragraph({
        alignment:AlignmentType.CENTER, spacing:sp(0,80),
        children:[new TextRun({ text:"ASL-TR-2026-002  |  Internal Research Document  |  Confidential", font:"Inter", size:19, color:C.gray, italics:true })]
      }),
      spacer(120),
      new Paragraph({
        alignment:AlignmentType.CENTER, spacing:sp(0,40),
        children:[new TextRun({ text:"HiSE", font:"Inter", size:72, bold:true, color:C.navy })]
      }),
      new Paragraph({
        alignment:AlignmentType.CENTER, spacing:sp(0,60),
        children:[new TextRun({ text:"Hierarchical Position Encoding for", font:"Inter", size:36, color:C.blue })]
      }),
      new Paragraph({
        alignment:AlignmentType.CENTER, spacing:sp(0,120),
        children:[new TextRun({ text:"Structure-Sensitive Semantic Dense Retrieval", font:"Inter", size:36, color:C.blue })]
      }),
      new Paragraph({
        alignment:AlignmentType.CENTER, spacing:sp(0,60),
        children:[new TextRun({ text:"Research Foundation Document", font:"Inter", size:24, color:C.gray, italics:true })]
      }),
      new Paragraph({
        alignment:AlignmentType.CENTER, spacing:sp(0,40),
        children:[new TextRun({ text:"Problem Statement  ·  Abstract  ·  Research Summary  ·  Artifact Registry", font:"Inter", size:22, color:C.gray })]
      }),
      spacer(120),
      makeTable(
        ["Field","Detail"],
        [
          ["Document Type","Internal Research Foundation — Confidential"],
          ["Prepared by","Nilesh Sanyasi, Founder & Lead AI Engineer, ASL"],
          ["Institution","VPPCOE Mumbai — Final Year B.E. (AI & Data Science), 2026–27"],
          ["Version","v1.0 — June 2026"],
          ["Primary Target","ICNLP 2027 (IEEE Xplore) — Submission Deadline Nov 30, 2026"],
          ["Research Code","ASL-TR-2026-002"],
        ],
        [2800, 6560]
      ),
      pb(),

      // ── ARTIFACT REGISTRY ──────────────────────────────────
      h1("Artifact Registry"),
      spacer(60),
      makeTable(
        ["Artifact","Name","Notes"],
        [
          ["Paper Title","HiSE: Hierarchical Position Encoding for Structure-Sensitive Semantic Dense Retrieval","Primary submission: ICNLP 2027"],
          ["Architecture / Mechanism","HiSA — Hierarchical Structural Addressing","The positional encoding mechanism; ASL's original contribution"],
          ["Trained Embedding Model","HiSE-Embed","Open-source release on HuggingFace"],
          ["Benchmark","StructEval-Bench v1","Open-source release on GitHub + HuggingFace"],
          ["Retrieval System","StructRAG","Built on HiSE-Embed; production integration target for DataSense"],
          ["GitHub / HuggingFace","asl-research/hise","Primary open-source repo"],
          ["arxiv Preprint","Submitted parallel to conference","Increases visibility; does not conflict with IEEE submission"],
        ],
        [2400, 3200, 3760]
      ),
      spacer(),
      divider(),

      // ── PROBLEM STATEMENT ──────────────────────────────────
      h1("1. Problem Statement"),
      spacer(40),

      h2("1.1 The Core Observation"),
      p(run("Every major embedding model in production today — BGE-M3, Qwen3-Embedding, E5-Mistral, Nomic Embed — encodes position the same way: as a linear integer. Token at position 1, token at position 2, token at position 47. The dominant mechanism doing this encoding is "), runB("Rotary Position Embedding (RoPE)"), run(", which rotates query and key vectors by an angle proportional to their absolute sequence position, producing attention scores that reflect relative linear distance between tokens.")),
      spacer(40),
      p(run("This works well for prose, conversation, and linear narrative. It works "), runB("poorly"), run(" for the dominant format of real-world enterprise and technical data: "), runB("hierarchically structured documents"), run(". In these documents, meaning is not a function of token sequence position alone — it is a function of "), runB("structural address"), run(": which section, which heading, which table row and column, which JSON key-path a token belongs to.")),
      spacer(40),
      callout(
        "The Precise Failure",
        "The sentence 'Revenue increased 40%' embedded under the heading '## Q3 Performance > Revenue' and the same sentence under '## Q3 Risks > Revenue' will produce near-identical embedding vectors in every current production model. The structural context that distinguishes them — one is a positive signal, one is a warning — is architecturally invisible. A RAG system retrieving against a query 'what are the company's Q3 risks' may surface the performance-framed chunk as equally or more relevant than the risk-framed chunk. The LLM downstream then reasons over structurally wrong context and produces a confidently wrong answer."
      ),
      spacer(),

      h2("1.2 Why This Has Not Been Solved"),
      p(run("Three independent reasons, each sufficient on its own:")),
      spacer(40),
      bullet([runB("Research silo separation. "), run("Table understanding researchers (TUTA, TAPAS, TabBiN) built structure-aware architectures for classification tasks — cell type classification, table type classification. Retrieval/embedding researchers (BGE, E5, Nomic) optimized for MTEB/BEIR benchmark scores. These two communities do not overlap, and MTEB/BEIR have no structurally-adversarial test cases that would expose the failure mode.")]),
      bullet([runB("A misleading 2022 finding. "), run("Wang et al. (2022) concluded 'Table Retrieval May Not Necessitate Table-specific Model Design,' suggesting generic text embeddings were competitive with structure-specific architectures on retrieval. This likely discouraged follow-up architecture work in the retrieval community — but the finding predates the RAG era and the more nuanced, adversarial structural failure modes that production RAG surfaces.")]),
      bullet([runB("Benchmarks don't test structure. "), run("If a benchmark has no structurally-adversarial pairs (same text, different structural context, opposite ground-truth relevance), no model will optimize for structural sensitivity, and no researcher will notice the gap. StructEval-Bench v1 is the first benchmark explicitly designed to test this — its existence is itself part of the research contribution.")]),
      spacer(),

      h2("1.3 The Gap, Stated Precisely"),
      spacer(40),
      callout(
        "Research Gap",
        "No embedding model trained for dense retrieval via contrastive learning incorporates a positional encoding mechanism that is sensitive to hierarchical structural address (document heading path, table row/column coordinates, JSON key-path depth) rather than linear sequence position alone. Every current production retrieval embedding model has this architectural blind spot, and no benchmark exists to measure it."
      ),
      spacer(),

      h2("1.4 Structural Failure Modes — Evidence Table"),
      spacer(40),
      makeTable(
        ["Document Type","Structural Element","Concrete Failure","Consequence in RAG"],
        [
          ["Markdown / Technical Docs","Heading hierarchy\n(H1 > H2 > H3)","'pip install x' under ## Installation and ## Troubleshooting embed identically","Model retrieves installation instructions for a troubleshooting query"],
          ["Financial Reports","Section hierarchy\n(Performance vs Risks)","'Revenue grew 40%' under ## Performance and ## Risks embed identically","Risk analysis query surfaces positive-framed content; analyst gets wrong signal"],
          ["Tables / Spreadsheets","Row × Column coordinate","Cell value '47' has no row/column anchor; headers and values detached in vector space","Query about Q3 revenue surfaces Q3 loss row; numerical analysis breaks"],
          ["JSON / Dict / Config","Key-path nesting\n(parent.child.key)","config.primary.timeout and config.fallback.timeout embed near-identically","System config query returns fallback values as primary; silent misconfiguration"],
          ["Legal / Contracts","Clause hierarchy\n(Article > Exception)","Exception clause text embeds close to obligation clause text","Legal query retrieves exception when obligation is needed; compliance risk"],
        ],
        [2000, 2000, 2800, 2800]
      ),
      spacer(),
      divider(),

      // ── ABSTRACT ───────────────────────────────────────────
      pb(),
      h1("2. Abstract"),
      spacer(40),
      new Table({
        width:{ size:9360, type:WidthType.DXA },
        columnWidths:[9360],
        rows:[new TableRow({ children:[new TableCell({
          borders:{ top:{style:BorderStyle.SINGLE,size:8,color:C.navy}, bottom:{style:BorderStyle.SINGLE,size:8,color:C.navy}, left:{style:BorderStyle.NONE,size:0,color:C.white}, right:{style:BorderStyle.NONE,size:0,color:C.white} },
          shading:{ fill:C.lgray, type:ShadingType.CLEAR },
          margins:{ top:200, bottom:200, left:280, right:280 },
          children:[
            new Paragraph({ spacing:sp(0,100), children:[runB("HiSE: Hierarchical Position Encoding for Structure-Sensitive Semantic Dense Retrieval", {size:24, color:C.navy})] }),
            new Paragraph({ spacing:sp(0,80), children:[run("Nilesh Sanyasi — Applied Sentience Labs Pvt. Ltd., Mumbai, India", {size:20, color:C.gray, italics:true})] }),
            spacer(40),
            new Paragraph({ spacing:sp(0,80), children:[
              run("Dense retrieval systems depend on embedding models to produce vector representations that preserve semantic meaning across queries and documents. Current embedding models encode token position using linear mechanisms — predominantly Rotary Position Embedding (RoPE) — which capture sequential distance between tokens but are architecturally blind to ", {size:21}),
              runB("hierarchical structural position", {size:21}),
              run(": the heading path, table row-column coordinate, or JSON key-path that governs a token's contextual meaning in structured documents. This produces a systematic failure mode in which tokens with near-identical surface text but structurally-opposed meanings — a metric framed as a performance indicator versus a risk indicator, a configuration value under a primary versus fallback key — are assigned dangerously similar embedding vectors, causing retrieval systems to surface structurally incorrect context for downstream language model reasoning.", {size:21}),
            ]}),
            new Paragraph({ spacing:sp(0,80), children:[
              run("We present ", {size:21}),
              runB("HiSE", {size:21}),
              run(" (Hierarchical Structure-aware Embeddings), a contrastive retrieval embedding model incorporating ", {size:21}),
              runB("HiSA (Hierarchical Structural Addressing)", {size:21}),
              run(": a novel positional encoding mechanism that extends RoPE from one-dimensional sequence position to multi-dimensional structural address, encoding depth, parent path, and sibling index as explicit rotational components of the position representation. HiSE is trained from a small SLM backbone via contrastive learning with structurally-adversarial hard negatives — pairs that are surface-text-similar but structurally-opposed — forcing the model to develop structural sensitivity as a first-class representational property rather than an emergent side effect.", {size:21}),
            ]}),
            new Paragraph({ spacing:sp(0,80), children:[
              run("We validate HiSE on ", {size:21}),
              runB("StructEval-Bench v1", {size:21}),
              run(", the first retrieval benchmark explicitly designed to test structural discrimination: 300–500 structurally-adversarial pairs across three document formats (markdown heading hierarchies, tabular row-column structures, and nested JSON key-paths), with ground-truth relevance labels requiring structural sensitivity to resolve. We additionally evaluate on standard BEIR and MTEB retrieval subsets to confirm that structural gains do not degrade general retrieval performance. HiSE is compared against same-size standard-architecture baselines trained on identical data, isolating the HiSA positional encoding as the experimental variable. Ablations confirm that the architectural modification — not data distribution — drives structural retrieval improvement.", {size:21}),
            ]}),
            new Paragraph({ spacing:sp(0,0), children:[
              run("HiSE-Embed and StructEval-Bench v1 are released open-source. HiSA is designed as a format-agnostic structural addressing mechanism applicable to any document format representable as a tree, providing a principled positional foundation for structure-sensitive retrieval beyond the three validated domains.", {size:21}),
            ]}),
          ]
        })]})],
      }),
      spacer(),
      divider(),

      // ── RESEARCH SUMMARY ───────────────────────────────────
      pb(),
      h1("3. Research Summary"),
      spacer(40),

      h2("3.1 What We Are Building"),
      spacer(40),
      makeTable(
        ["Component","What It Is","What Makes It Novel"],
        [
          ["HiSA\n(Hierarchical Structural Addressing)","A positional encoding mechanism that assigns every token a structural address — depth in document tree, parent section identity, sibling index — and encodes this as multi-dimensional rotational components, extending RoPE from 1D to structural N-D","No prior work applies tree-positional encoding to contrastive retrieval embedding models. TUTA/DOM-LM applied structural encoding to classification tasks only. RethinkRoPE (2025) extended RoPE to N-D for vision/spatial — not applied to document hierarchy or retrieval."],
          ["HiSE-Embed\n(The Trained Model)","A small (~100-300M parameter) SLM-backbone encoder with HiSA replacing standard RoPE, trained via contrastive learning with structurally-adversarial hard negatives","First contrastive retrieval model with explicit hierarchical positional encoding validated on structurally-adversarial retrieval benchmarks"],
          ["StructEval-Bench v1\n(The Benchmark)","300-500 structurally-adversarial pairs across markdown, tables, and JSON — same or similar surface text, structurally-opposed ground-truth relevance. Metrics: SDS (Structural Discrimination Score) and RIR (Retrieval Inversion Rate)","First retrieval benchmark designed specifically to measure structural sensitivity. BEIR/MTEB have no structurally-adversarial test cases."],
          ["StructRAG\n(The Retrieval System)","A retrieval pipeline built on HiSE-Embed with structure-preserving chunking that maintains heading/table/key-path metadata through to encoding","Demonstrates end-to-end production viability; serves as integration point for ASL DataSense"],
        ],
        [2000, 3200, 4160]
      ),
      spacer(),

      h2("3.2 The Architectural Contribution — HiSA in Plain Terms"),
      p(run("Standard RoPE assigns every token one number — its sequence position — and rotates its query/key vectors by an angle proportional to that number. The result: attention scores reflect how far apart tokens are in the token stream. Two tokens 10 positions apart get the same attention relationship regardless of whether they are in the same section or different sections.")),
      spacer(40),
      p(run("HiSA replaces that one number with a "), runB("structural address tuple"), run(": (sequence_position, depth, parent_id, sibling_index). Each dimension gets its own rotational component, applied jointly so that the final attention score reflects both sequential proximity and structural proximity. Two tokens in the same heading section get a structurally-close representation even if they are 200 tokens apart sequentially. Two tokens in sibling sections under different parents get a structurally-distant representation even if they are adjacent in the token stream.")),
      spacer(40),
      twoColCallout(
        "Standard RoPE (current models)",
        "Attention(token_A, token_B) ~ f(|pos_A - pos_B|)\n\nOnly one signal: how far apart in the token stream.",
        "HiSA (our contribution)",
        "Attention(token_A, token_B) ~ f(|pos_A - pos_B|, depth_dist, parent_match, sibling_dist)\n\nFour signals: sequence + structural address."
      ),
      spacer(),

      h2("3.3 Prior Art Map — Where We Fit"),
      spacer(40),
      makeTable(
        ["Work","What It Does","Why It's Not Enough","Our Relationship"],
        [
          ["TUTA (Microsoft, KDD 2021)","Bi-dimensional coordinate tree for table structure; tree-based attention and position embedding","Trained for cell/table type classification only. No contrastive retrieval training. No cross-format claim.","Foundational motivation; we adapt the structural coordinate concept to RoPE for retrieval"],
          ["DOM-LM (2022)","Injects node index, parent index, depth, sibling index as additive position embeddings into BERT for HTML","HTML-only. Additive not rotational. Not trained for retrieval/RAG use case.","Most direct architectural prior art; we extend the idea to RoPE and to retrieval training"],
          ["Tree-Enhanced CodeBERTa (2025)","Integrates AST depth and sibling index as positional embeddings into CodeBERTa for code representation","Code/AST only. Classification tasks. Not a contrastive retrieval model.","Validates that tree PE improves transformer representations; we apply to document retrieval"],
          ["Struc-EMB (Georgia Tech, Oct 2025)","Tests sequential concatenation vs parallel caching of structural context in LLM-based embedders","Tests prompt/context-level approaches, not architectural position encoding changes. No structural hard negatives in training.","Directly motivates why architecture-level fix beats prompt-level workaround"],
          ["RethinkRoPE (arXiv 2504.06308, 2025)","Mathematical blueprint for extending RoPE to N-dimensional positional embedding","Proposed for vision/spatial domains. Not applied to document hierarchy or retrieval.","Mathematical foundation for HiSA's multi-dimensional rotational extension"],
          ["TabBiN (EDBT 2025)","Bi-dimensional tabular coordinates and visibility matrix for 2D tabular data embedding","Table-specific. Not designed for dense retrieval RAG. No cross-format generalization.","Validates that bi-dimensional structural encoding improves table representations"],
          ["BGE-M3, Qwen3-Embedding, Nomic Embed","Current SOTA retrieval embedding models","Linear RoPE only. No structural address. Fail on structurally-adversarial pairs (confirmed by failure-mode survey).","Primary baselines; same-size variants used in ablation study"],
        ],
        [2000, 2400, 2400, 2160]
      ),
      spacer(),

      h2("3.4 What We Prove — The Experimental Claim"),
      spacer(40),
      callout(
        "Primary Hypothesis (Falsifiable)",
        "A contrastive retrieval embedding model incorporating HiSA (hierarchical structural addressing via multi-dimensional RoPE extension) will achieve statistically significant improvement in structural discrimination (SDS) and retrieval inversion rate (RIR) on StructEval-Bench v1 compared to a same-size same-data same-compute baseline using standard linear RoPE — and will not degrade general retrieval performance (nDCG@10) on BEIR/MTEB subsets beyond margin of error."
      ),
      spacer(),
      p(runB("The ablation design:"), run(" same SLM backbone, same training data, same compute budget, same contrastive loss — only the positional encoding mechanism changes. This isolates HiSA as the experimental variable and makes the result unambiguous.")),
      spacer(),

      h2("3.5 Why This Is Not Integration Engineering"),
      p(run("The three document formats (markdown, tables, JSON) are not three separate contributions assembled into one paper. They are three experimental domains that share a single structural property — all three are trees — used to validate that HiSA's positional encoding is "), runB("format-agnostic"), run(", not format-specific. The contribution is the mechanism (HiSA) and the claim (linear position is the wrong inductive bias for hierarchical documents). The formats are evidence for the generality of that claim.")),
      spacer(40),
      twoColCallout(
        "Integration Engineering (what we are NOT doing)",
        "Combine markdown encoder + table encoder + JSON encoder into one system. Contribution = unified pipeline.\n\nReviewer reads: 'Systems engineering.'",
        "Representational Research (what we ARE doing)",
        "Propose one positional encoding mechanism (HiSA) that handles all tree-structured documents. Validate on three domains to prove format-agnosticism.\n\nReviewer reads: 'Foundational positional theory with strong empirical validation.'"
      ),
      spacer(),
      divider(),

      // ── RESEARCH CONTEXT ───────────────────────────────────
      pb(),
      h1("4. Research Context & Motivation"),
      spacer(40),

      h2("4.1 Origin — DataSense & Production RAG Failure"),
      p(run("This research originated from a production failure observed during development of "), runB("DataSense"), run(", ASL's AI-powered data analysis product for SMBs. DataSense's current architecture (v1.0, March 2026) compensates for AI models' inability to reason over structured data by generating an explicit 'data manifest' — a statistical description of column types, relationships, and domain signals — passed to the model as prompt context.")),
      spacer(40),
      p(run("This is a prompt-engineering workaround for a representational problem. The correct fix is not a better prompt — it is an embedding model that encodes structural context natively, so that retrieval over structured documents (financial tables, configuration files, technical specifications) surfaces the right structural chunk without requiring an explicit manifest intermediary. HiSE-Embed is that fix. StructRAG is its integration point into DataSense Phase 2.")),
      spacer(),

      h2("4.2 Placement in the Research Landscape"),
      p(run("The embedding model field's 2023–2025 focus was scale and context length (BGE-M3, E5-Mistral, Qwen3-Embedding) competing on MTEB leaderboard scores. A 2025 Google DeepMind paper explicitly flagged theoretical limitations of single-vector embedding-based retrieval, noting that models increasingly face fundamental representational ceilings as the diversity of content to represent grows. HiSE addresses one specific, well-defined ceiling: structural sensitivity. It is not a general-purpose improvement — it is a targeted fix for a measurable, demonstrated failure mode.")),
      spacer(),

      h2("4.3 The LLM Extension (Future Work)"),
      p(run("If HiSA improves retrieval embeddings by making the model structurally sensitive to its input, the same mechanism applied to a "), runB("generative LLM's"), run(" positional encoding would make the model structurally sensitive to its context window during generation — attending differently to tokens in the same structural section versus structurally-opposed sections, even at the same sequential distance. This is a legitimate and interesting extension (active work on RoPE modification for LLMs exists in 2024–2025) but requires a full generative LLM training run and is explicitly scoped as future work in the paper. It is named here to establish the research trajectory.")),
      spacer(),
      divider(),

      // ── PUBLICATION STRATEGY ───────────────────────────────
      h1("5. Publication Strategy"),
      spacer(40),
      makeTable(
        ["Priority","Venue","Deadline","Strategy"],
        [
          ["Primary","ICNLP 2027 (IEEE Xplore)","Nov 30, 2026","Main submission. Best timeline match. NLP + IR scope fits perfectly."],
          ["Backup 1","PUNECON 2026 (IEEE Pune Section)","~Sep–Oct 2026","India-based. Strong for college final review. Check punecon.ieeepunesection.org for exact deadline."],
          ["Backup 2","MLNLP 2026 (IEEE)","Check IEEE CIS calendar","ML + NLP focus. Confirms topic fit if ICNLP rejects."],
          ["Parallel","arXiv preprint","Anytime post-submission","Post simultaneously with conference submission for visibility. Does not conflict with IEEE."],
          ["Stretch","ACL/EMNLP Workshop","Mar–May 2027","If results are strong enough post-primary submission. Workshop track receptive to novel systems work."],
          ["Second Paper","StructEval-Bench standalone","2027","If benchmark results are rich enough to stand alone as a dataset/evaluation contribution paper."],
        ],
        [1200, 2400, 1800, 4000]
      ),
      spacer(),
      p(runB("Rule: "), run("sequential submissions only, never simultaneous. One venue at a time. Rejected paper gets reviewer feedback incorporated before next submission.")),
      spacer(),
      divider(),

      // ── OPEN SOURCE STRATEGY ───────────────────────────────
      h1("6. Open Source & ASL Research Identity"),
      spacer(40),
      p(run("All research artifacts release open-source under the "), runB("asl-research"), run(" GitHub organization. This builds ASL's public research identity — the same strategy that allowed small labs like EleutherAI and MosaicML to attract research talent and investment before being acquired.")),
      spacer(40),
      makeTable(
        ["Artifact","Release Platform","License","Timing"],
        [
          ["HiSE-Embed (trained model)","HuggingFace: asl-research/hise-embed","Apache 2.0","With paper submission (or arXiv preprint)"],
          ["StructEval-Bench v1 (benchmark)","HuggingFace Datasets + GitHub","CC BY 4.0","With paper submission"],
          ["HiSA implementation (code)","GitHub: asl-research/hise","MIT","With paper submission"],
          ["StructRAG pipeline","GitHub: asl-research/structrag","Apache 2.0","Phase 5 completion"],
          ["Training code & configs","GitHub: asl-research/hise","MIT","With paper submission"],
        ],
        [2400, 2800, 1600, 2560]
      ),
      spacer(),
      divider(),

      // ── CLOSING ────────────────────────────────────────────
      spacer(120),
      new Paragraph({
        alignment:AlignmentType.CENTER, spacing:sp(0,40),
        children:[runB("Applied Sentience Labs Pvt. Ltd.", {color:C.navy, size:22})]
      }),
      new Paragraph({
        alignment:AlignmentType.CENTER, spacing:sp(0,40),
        children:[run("appliedsentiencelabs.com  |  nilesh@appliedsentiencelabs.com", {color:C.gray, size:19, italics:true})]
      }),
      new Paragraph({
        alignment:AlignmentType.CENTER, spacing:sp(0,0),
        children:[run("ASL-TR-2026-002  |  HiSE Research Foundation Document  |  v1.0  |  June 2026", {color:C.gray, size:18, italics:true})]
      }),
    ]
  }]
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync('/mnt/user-data/outputs/ASL_HiSE_Foundation_Doc.docx', buf);
  console.log('Done');
});