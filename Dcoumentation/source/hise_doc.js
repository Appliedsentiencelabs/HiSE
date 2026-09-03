const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  WidthType, AlignmentType, BorderStyle, HeadingLevel, PageNumber,
  Footer, Header, ShadingType
} = require('docx');
const fs = require('fs');
const path = require('path');

const FONT = 'Times New Roman';
const BLUE = '1F4E79';
const LIGHT_BLUE = 'D9EAF7';
const BORDER = '808080';

const run = (value, options = {}) => new TextRun({
  text: value, font: FONT, size: 24, color: '000000', ...options
});

const body = (...children) => new Paragraph({
  children,
  alignment: AlignmentType.JUSTIFIED,
  spacing: { line: 360, before: 0, after: 120 }
});

const text = (value) => run(value);
const bold = (value) => run(value, { bold: true });
const italic = (value) => run(value, { italics: true });

const heading = (value, level = HeadingLevel.HEADING_1) => new Paragraph({
  children: [run(value, { bold: true, size: 28 })],
  heading: level,
  alignment: AlignmentType.LEFT,
  spacing: { line: 360, before: 240, after: 120 }
});

const subheading = (value) => new Paragraph({
  children: [run(value, { bold: true, italics: true, size: 24 })],
  alignment: AlignmentType.LEFT,
  spacing: { line: 360, before: 160, after: 80 }
});

const bullet = (value) => new Paragraph({
  children: [text(value)],
  bullet: { level: 0 },
  alignment: AlignmentType.JUSTIFIED,
  spacing: { line: 360, before: 0, after: 80 }
});

const centered = (...children) => new Paragraph({
  children,
  alignment: AlignmentType.CENTER,
  spacing: { line: 360, before: 0, after: 100 }
});

const caption = (value) => centered(run(value, { bold: true, size: 24 }));

const cell = (value, header = false) => new TableCell({
  borders: {
    top: { style: BorderStyle.SINGLE, size: 4, color: BORDER },
    bottom: { style: BorderStyle.SINGLE, size: 4, color: BORDER },
    left: { style: BorderStyle.SINGLE, size: 4, color: BORDER },
    right: { style: BorderStyle.SINGLE, size: 4, color: BORDER }
  },
  shading: header ? { type: ShadingType.CLEAR, fill: LIGHT_BLUE } : undefined,
  margins: { top: 80, bottom: 80, left: 100, right: 100 },
  children: [new Paragraph({
    children: [run(value, { bold: header, size: 20 })],
    alignment: AlignmentType.JUSTIFIED,
    spacing: { line: 300, before: 0, after: 0 }
  })]
});

const makeTable = (headers, rows, widths) => new Table({
  alignment: AlignmentType.CENTER,
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: widths,
  rows: [
    new TableRow({ children: headers.map((item) => cell(item, true)) }),
    ...rows.map((row) => new TableRow({ children: row.map((item) => cell(item)) }))
  ]
});

const doc = new Document({
  creator: 'Applied Sentience Labs',
  title: 'Project Topic Registration Report - HiSE',
  description: 'Project Topic Registration Report for HiSE',
  styles: {
    default: {
      document: {
        run: { font: FONT, size: 24 },
        paragraph: { spacing: { line: 360 } }
      }
    }
  },
  sections: [{
    properties: { page: { margin: { top: 1080, bottom: 1080, left: 1080, right: 1080 } } },
    headers: {
      default: new Header({ children: [centered(run('Project Topic Registration Report', { bold: true, size: 20, color: BLUE }))] })
    },
    footers: {
      default: new Footer({ children: [centered(run('BE AI & DS SEM VII | HiSE | Page ', { size: 18 }), PageNumber.CURRENT)] })
    },
    children: [
      centered(run('Project Title: Hierarchical Position Encoding for Structure-Sensitive Semantic Dense Retrieval', { bold: true, size: 36, color: BLUE })),
      centered(run('HiSE', { bold: true, size: 30, color: BLUE })),
      centered(text('Nilesh Sanyasi | VU2F2324013')),
      centered(text('Maitreya Channe | VU2F2324015')),
      centered(text('Sarthak Gavas | VU2F2324007')),
      centered(text('Sairaj Gawde | VU2F2324008')),
      centered(text('BE AI & DS SEM VII - Major Project Review 1 | Group No: 18')),
      centered(text('Project Guide: Dr. Alam Shaikh')),
      centered(text("Vasantdada Patil Pratishthan's College of Engineering and Visual Arts")),
      centered(text('Department of Artificial Intelligence & Data Science | Academic Year 2026-27')),

      heading('1. Problem Definition'),
      body(italic('When the same sentence means opposite things depending on where it appears in data, can the embedding model tell the difference?')),
      body(text('Production embedding models encode token position as a single integer: sequence position 1, 2, 3 and so on. The dominant mechanism, Rotary Position Embedding (RoPE), produces attention scores based primarily on linear distance between tokens. It does not represent the structural address of a token, such as its heading path, table row-column coordinate, or JSON key-path.')),
      body(text('This creates a systematic retrieval failure. Chunks with similar surface text but structurally opposite meanings can receive nearly identical embedding vectors. A retrieval-augmented generation system may therefore return the wrong context and allow a language model to produce a confident but incorrect answer.')),
      body(text('The research problem is to design and evaluate a structure-sensitive positional encoding mechanism that represents both sequence position and the hierarchical location of information across Markdown, tables, JSON, HTML/XML, and complex documents.')),

      heading('2. Introduction'),
      subheading('2.1 Retrieval-Augmented Generation and the Retrieval Bottleneck'),
      body(text('Retrieval-Augmented Generation (RAG) systems depend on the quality of document embeddings. The production pipeline is: Document -> Chunk -> Embed -> Vector Database -> Query Match -> Language Model. When the embedding stage ignores document hierarchy, retrieval can fail before generation begins. Enterprise data such as filings, legal contracts, Kubernetes configurations, clinical protocols, and technical documentation is highly structured, so the impact increases with structural complexity.')),
      body(text('Existing embedding models such as BGE-M3, Qwen3-Embedding, E5-Mistral, and Nomic Embed preserve sequence order but do not explicitly encode where a token lives in a document structure. This limits their ability to discriminate between structurally opposed passages.')),
      subheading('2.2 HiSE Overview'),
      body(text('HiSE, or Hierarchical Position Encoding for Structure-Sensitive Semantic Dense Retrieval, proposes HiSA (Hierarchical Structural Addressing). HiSA extends RoPE from one-dimensional sequence position to a composite structural address containing sequence position, tree depth, parent section identity, and sibling index.')),
      body(bold('Core formulation: '), text('standard RoPE uses theta = theta1 multiplied by m, while HiSA uses theta = theta1m + theta2d + theta3p + theta4s. Here m is sequence position, d is structural depth, p is parent identity, and s is sibling index.')),

      heading('3. Literature Review'),
      body(text('Research on structure-aware representation has progressed through table encoders, HTML encoders, code structure encoders, and prompt-based approaches. However, these approaches do not provide one architecture-level rotational encoding for hierarchical dense retrieval across multiple data formats.')),
      caption('Table 1: Literature review of structure-aware representation methods'),
      makeTable(
        ['Paper or System', 'Year', 'Contribution', 'Limitation'],
        [
          ['Attention Is All You Need', '2017', 'Foundational transformer and sinusoidal positional encoding.', 'Models sequence position, not hierarchical document structure.'],
          ['TUTA', '2021', 'Tree-based coordinates for generally structured tables.', 'Focused on classification and table pre-training, not dense retrieval.'],
          ['BEIR', '2021', 'Heterogeneous zero-shot information retrieval benchmark.', 'Does not include structurally adversarial test pairs.'],
          ['DOM-LM', '2022', 'Depth, parent, and sibling information for HTML documents.', 'HTML-specific and additive; no unified retrieval architecture.'],
          ['MTEB', '2023', 'Large benchmark for text embedding models.', 'Does not directly expose structural discrimination failure.'],
          ['Struc-EMB', '2025', 'Structural context through LLM-based embedding prompts.', 'Prompt-level workaround; positional encoding remains unchanged.'],
          ['Rethinking RoPE', '2025', 'Blueprint for N-dimensional rotary position embedding.', 'Primarily discussed for vision and spatial domains.']
        ],
        [2200, 900, 3000, 3260]
      ),
      subheading('3.1 Research Gap in Dense Retrieval'),
      body(text('No existing production embedding model encodes a composite structural address across tabular, JSON, HTML/XML, Markdown, and complex document formats. Existing solutions address individual formats or use prompt concatenation, rather than changing the positional encoding layer of the attention mechanism.')),
      subheading('3.2 Need for a New Benchmark'),
      body(text('BEIR and MTEB evaluate broad retrieval quality but contain zero structurally adversarial pairs designed to test whether identical or similar text is distinguished by its location in a hierarchy. StructEval-Bench v1 is proposed to fill this gap with 300-500 query-document pairs and explicit Structural Discrimination Score (SDS) and Retrieval Inversion Rate (RIR) metrics.')),

      heading('4. Motivation'),
      bullet('Structure carries meaning: headings, table coordinates, JSON key-paths, and HTML node hierarchies provide semantic information beyond the words themselves.'),
      bullet('Current models are structure-blind: modern embedding architectures preserve token order but do not directly encode document hierarchy.'),
      bullet('Retrieval fails before generation: an incorrect embedding propagates structural errors through the entire RAG pipeline.'),
      bullet('Embedding quality determines retrieval quality: improving the positional representation can improve the relevance of retrieved context.'),
      bullet('HiSE addresses the root cause by modeling where information exists in a document, rather than relying only on larger models or more training data.'),

      heading('5. Objectives'),
      bullet('Design HiSA as a format-agnostic extension of RoPE using a composite structural address (m, d, p, s).'),
      bullet('Develop HiSE-Embed, a contrastive retrieval model with fewer than 300 million parameters.'),
      bullet('Construct StructEval-Bench v1 with 300-500 structurally adversarial query-document pairs across Markdown, tables, and JSON.'),
      bullet('Evaluate Structural Discrimination Score and Retrieval Inversion Rate against standard RoPE-based baselines.'),
      bullet('Maintain general retrieval quality on BEIR and MTEB subsets, with nDCG@10 within a two percent margin of the baseline.'),
      bullet('Use controlled ablation to isolate the effect of the positional encoding while keeping backbone, data, and compute constant.'),
      bullet('Release the model, benchmark, and HiSA implementation as reproducible open-source research artifacts.'),

      heading('6. Methodology Used'),
      body(text('The project uses a five-phase experimental methodology. The architecture is changed first, the benchmark is designed specifically for the failure mode, and the final claim is tested through controlled comparison and ablation.')),
      subheading('6.1 Failure-Mode Survey'),
      body(text('BGE-M3, Qwen3-Embedding, and Nomic Embed will be evaluated on structurally adversarial pairs from tabular, JSON, HTML, and Markdown data. Baseline cosine similarity and Retrieval Inversion Rate will be recorded. The initial target is cosine similarity of at least 0.90 for structurally opposed pairs under existing models.')),
      subheading('6.2 HiSA Implementation'),
      body(text('The standard RoPE layer will be replaced with a four-dimensional structural address. Markdown AST, pandas table, JSON recursive, and HTML/XML DOM parsers will produce unified (m, d, p, s) tuples. The degenerate case d=0, p=0, s=0 must produce the same output as standard RoPE.')),
      subheading('6.3 StructEval-Bench v1 Construction'),
      body(text('A benchmark of 300-500 structurally adversarial query-document pairs will be created. The pairs will cover Markdown heading hierarchy, table row-column structure, and nested JSON key-paths. Annotation guidelines and the SDS and RIR metrics will be published with the benchmark.')),
      subheading('6.4 Contrastive Training and Ablation'),
      body(text('HiSE-Embed will be trained with InfoNCE contrastive loss using both semantic hard negatives and structurally adversarial hard negatives. A controlled ablation will compare the HiSA model with an otherwise identical standard RoPE model.')),
      subheading('6.5 Evaluation'),
      body(text('The trained model will be evaluated on StructEval-Bench v1 for structural discrimination and on BEIR/MTEB subsets for general retrieval parity. Statistical significance will be tested for the SDS improvement, with p less than 0.05 as the target criterion.')),

      heading('7. Development Plan'),
      caption('Table 2: HiSE development plan and success criteria'),
      makeTable(
        ['Phase', 'Timeline', 'Planned Work', 'Success Criterion'],
        [
          ['Failure Survey', 'Month 1-2', 'Measure the structural discrimination gap across baseline embedding models and formats.', 'Cosine similarity at least 0.90 on structurally opposed pairs; RIR baseline established.'],
          ['HiSA Build', 'Month 2-3', 'Implement the four-dimensional address, parsers, and POMDP structural state estimator.', 'Zero-structure case matches standard RoPE.'],
          ['Benchmark', 'Month 3-4', 'Build and document StructEval-Bench v1 with SDS and RIR metrics.', 'Benchmark released with annotation guidelines.'],
          ['Training', 'Month 4-5', 'Train HiSE-Embed with structural hard negatives and run the ablation.', 'Statistically significant SDS improvement over the RoPE baseline.'],
          ['Evaluation and Release', 'Month 6', 'Evaluate on BEIR/MTEB, release artifacts, and prepare the research submission.', 'nDCG@10 within two percent; submission by November 30, 2026.']
        ],
        [1500, 1200, 3600, 3060]
      ),
      subheading('7.1 Technology and Requirements'),
      body(text('The implementation will use Python, PyTorch 2.x, CUDA, HuggingFace Transformers, Sentence-Transformers, FAISS, Python-Markdown, pandas, JSON parsing, and HTML/XML DOM parsing. BEIR, MTEB, and StructEval-Bench v1 will support evaluation. Docker will be used for reproducible development and training.')),
      body(text('Training and prototype validation will use locally hosted NVIDIA GPU systems with at least 8 GB VRAM, 16-32 GB RAM, and approximately 500 GB SSD storage for checkpoints and embedding indexes.')),

      heading('8. Conclusion'),
      body(text('HiSE addresses a fundamental limitation in dense retrieval: existing positional encodings represent where a token appears in a sequence but not where it exists in a document structure. HiSA introduces a composite structural address and embeds it into the rotational geometry of attention.')),
      body(text('The project will deliver three primary artifacts: HiSE-Embed, a structure-sensitive embedding model under 300 million parameters; StructEval-Bench v1, a benchmark for structural discrimination; and an open-source HiSA implementation. The controlled experimental design keeps the backbone, data, and compute constant so that any retrieval improvement can be attributed to the positional encoding change.')),
      body(text('Future work includes applying HiSA to generative language models, extending structural retrieval to Indic and other multilingual settings, and expanding StructEval-Bench to legal, clinical, and financial documents with expert annotation.')),

      heading('9. References'),
      body(text('[1] Vaswani, A. et al. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems, 30.')),
      body(text('[2] Wang, Z. et al. (2021). TUTA: Tree-based Transformers for Generally Structured Table Pre-training. KDD 2021.')),
      body(text('[3] Deng, X. et al. (2022). DOM-LM: Learning Generalizable Representations for HTML Documents. arXiv:2201.10608.')),
      body(text('[4] Thakur, N. et al. (2021). BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. NeurIPS 2021.')),
      body(text('[5] Muennighoff, N. et al. (2023). MTEB: Massive Text Embedding Benchmark. arXiv:2210.07316.')),
      body(text('[6] Su, J. et al. (2024). RoFormer: Enhanced Transformer with Rotary Position Embedding. Neurocomputing, 568, 127063.')),
      body(text('[7] Liu, S. et al. (2025). Struc-EMB: The Potential of Structure-Aware Encoding in Language Embeddings. arXiv:2510.08774.')),
      body(text('[8] Liu, H. et al. (2025). Rethinking RoPE: A Mathematical Blueprint for N-dimensional Positional Embedding. arXiv:2504.06308.')),
      body(text('[9] Wang, Z. et al. (2022). Table Retrieval May Not Necessitate Table-specific Model Design. arXiv:2205.09843.'))
    ]
  }]
});

Packer.toBuffer(doc).then((buffer) => {
  const outputPath = path.join(__dirname, '..', 'reports', 'Project Topic Registration Report SH 26.docx');
  fs.writeFileSync(outputPath, buffer);
  console.log(`Done: ${outputPath}`);
});
