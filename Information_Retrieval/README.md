# IR Lab - CA Copilot tax-law retrieval and browsing prototype

Run it:

    pip install -r requirements.txt
    python3 app.py         # open http://127.0.0.1:8000

The supplied consolidated Income Tax Ordinance PDF in `../Material/` loads as the starter collection in 50-page
batches. The Week 5 Browse laws view organizes loaded pages into chapters, sections, subsections, and paragraphs,
with breadcrumbs, source-page links, related provisions, recent items, and browser history. Search and long chapter
lists are paginated at 50 items per page. Use "Load sample data" for the mock retrieval demo,
or upload `.txt`, `.md`, `.pdf`, `.docx`, or `.json` sources and paste text. Each time documents change, indexes
rebuild lazily (IDF depends on the whole collection).

## Metadata analysis agent
Use **Analyze metadata** beside the file picker to ask the optional Gemini agent for
source metadata: title, law, source type, tax year, dates, jurisdiction, and confidence.
Use **Structure & add** to ask the agent to preserve the source text in a hierarchical
chapter/section/subsection/paragraph JSON structure and add the resulting document to
the shared collection. Copy `.env.example` to
`.env` and set:

    GEMINI_API_KEY=your_google_ai_studio_key
    GEMINI_MODEL=gemini-3.1-flash-lite

If the key is missing or Gemini returns an error, the app reports the issue and uses the
existing local parser for structured ingestion; normal local ingestion continues to work.

## What is inside
- ir_core/analyzer.py   lowercase, stop-words, light stemming, legal citations kept whole (153(1)(b) -> also 153(1), 153)
- ir_core/documents.py  file -> Document tree (sections, subsections, paragraphs, provisos, markdown headings)
- ir_core/index.py      inverted index with positions, one per field (title / meta / body or heading / text)
- ir_core/boolean.py    Boolean parser (AND OR NOT, parentheses, "phrases", field:word) + set evaluation
- ir_core/ranking.py    BIM, VSM (TF-IDF + cosine), BM25 (fielded)
- ir_core/structured.py overlapped lists, non-overlapped lists, proximal nodes
- ir_core/engine.py     search / compare / evaluate / plain-English explanation

## Week 6 - fuzzy and extended Boolean retrieval
- **Fuzzy model** gives each query term a membership score from 0 to 1, supports partial/typo-tolerant matching, and combines expressions with `min` (AND), `max` (OR), and `1 - membership` (NOT).
- **Extended Boolean** provides explicit set-based include/exclude retrieval using `AND`, `OR`, `NOT`, and parentheses.
- Both models are available in Search and Compare. The evaluation view reports **Precision, Recall, F1 Measure, and Accuracy** for every model using the built-in relevance judgments.
- app.py + static/index.html   the web app

## Week 5 browsing
- `/api/browse` lists source documents, child nodes, or BM25 matches from the browse search box (`limit`/`offset` paginate results).
- Browse filters include legal item type and exact values from source metadata (for example Law, Source Type, Tax Year, Effective Date, Author, or Language when supplied). Available metadata filters are derived from the loaded collection; fields absent from a source are not offered.
- `/api/browse/filters` returns the available metadata fields and values; `/api/browse` and `/api/browse/node/{id}` apply the selected metadata filter before pagination.
- `/api/browse/load-next` loads the next 50 PDF pages; `/api/browse/status` reports indexed-page progress and failures.
- `/api/browse/document/{id}` opens a document's top-level categories.
- `/api/browse/node/{id}` returns a preview, breadcrumb path, children, and related section links.
- `/api/browse/source.pdf` opens the supplied source PDF; section previews include PDF page locations.
- The starter parser is tailored to the supplied FBR consolidated Income Tax Ordinance PDF. Other PDFs still use
  the general text parser and may need structure cleanup before their hierarchy is dependable.

## Query syntax
withholding tax contractor      bare words (Boolean ANDs them; ranked models OR them)
contractor AND NOT rent         operators must be UPPERCASE
"higher rate"                   phrase       title:salary   field search       153(1)(b)   citation

## Assignment mapping (10 marks)
Connection to FYP (2): the sample is a legal-chunk collection with the proposal's metadata (Law, Section, Source Type, Tax Year, Effective Date). BM25 here is the keyword half of the proposal's hybrid RAG.
Implementation (3): Boolean, BIM, VSM (TF-IDF + cosine), BM25, overlapped / non-overlapped / proximal.
Demonstration (2): the 7 sample queries in sample_data.py (use 3-5). Each search shows an explanation panel.
Comparison (2): "Compare all models" and the P/R/F1/Accuracy evaluation button. Try Fuzzy vs Extended Boolean, Boolean vs BIM, BIM vs BM25, overlapped vs BM25.
Feynman (1): the panel gives What goes in / What happens inside / What comes out / Why it came first.

## Presentation order (8 items)
FYP problem -> retrieval requirement -> selected models -> implementation (index, models) -> example queries -> results -> comparison -> conclusion.

## Honest notes
- Sample data is MOCK: illustrative paraphrases, no real rates or legal text. Load the real Ordinance/SRO text for the final demo.
- The stemmer is deliberately light. Swap light_stem() for Porter/Snowball if needed.
- BM25 scores are raw scores, not calibrated probabilities.
