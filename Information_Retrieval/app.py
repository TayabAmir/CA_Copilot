"""IR Lab web app.  Run:  python app.py   then open http://127.0.0.1:8000"""
from pathlib import Path
import re
from threading import Thread, Lock
from copy import deepcopy
import io
from typing import Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

try:
    from .ir_core import Engine
    from .ir_core.engine import MODELS, format_evaluation
    from .ir_core.sample_data import SAMPLE_JUDGMENTS
    from .ir_core.gemini_agent import analyze_source, structure_source
except ImportError:
    from ir_core import Engine
    from ir_core.engine import MODELS, format_evaluation
    from ir_core.sample_data import SAMPLE_JUDGMENTS
    from ir_core.gemini_agent import analyze_source, structure_source

app = FastAPI(title="IR Lab")
engine = Engine()
ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")
ORDINANCE = ROOT.parent / "Material" / "2026724177725705IncomeTaxOrdinanace2001.pdf"


CHUNK_PAGES = 50
SOURCE_STATUS = "loading" if ORDINANCE.exists() else "missing"
SOURCE_ERROR = None
SOURCE_LOADED_PAGES = 0
SOURCE_TOTAL_PAGES = 0
SOURCE_LOCK = Lock()


def _merge_partial(partial):
    """Append a parsed page range, merging nodes that span chunk boundaries."""
    if "ito-2001" not in engine.docs:
        engine.add_document(partial)
        return
    doc = engine.docs["ito-2001"]
    existing_root = doc.nodes[0]
    parsed_root = partial.nodes[0]
    existing_root.page_end = max(existing_root.page_end or 0, parsed_root.page_end or 0)
    mapped = {parsed_root.id: existing_root}
    def key(n):
        return (n.type, n.citation or n.heading)
    for src in partial.nodes[1:]:
        parent = mapped.get(src.parent)
        if parent is None:
            parent = existing_root
        dst = next((engine.nodes[c] for c in parent.children
                    if key(engine.nodes[c]) == key(src)), None)
        if dst is None:
            dst = deepcopy(src)
            dst.id = f"ito-2001#{len(doc.nodes)}"
            dst.doc_id = "ito-2001"
            dst.parent = parent.id
            dst.children = []
            parent.children.append(dst.id)
            doc.nodes.append(dst)
            engine.nodes[dst.id] = dst
        else:
            if src.text and (not dst.page_end or (src.page_start or 0) > dst.page_end):
                dst.text = (dst.text + " " + src.text).strip()
            dst.page_start = min(x for x in (dst.page_start, src.page_start) if x is not None) if dst.page_start is not None or src.page_start is not None else None
            dst.page_end = max(x for x in (dst.page_end, src.page_end) if x is not None) if dst.page_end is not None or src.page_end is not None else None
        mapped[src.id] = dst
    engine._dirty = True


def _load_chunk(start_page):
    global SOURCE_STATUS, SOURCE_ERROR, SOURCE_LOADED_PAGES
    try:
        from pypdf import PdfReader
        data = ORDINANCE.read_bytes()
        reader = PdfReader(io.BytesIO(data))
        end_page = min(start_page + CHUNK_PAGES - 1, SOURCE_TOTAL_PAGES)
        try:
            from .ir_core.documents import _parse_income_tax_ordinance
        except ImportError:
            from ir_core.documents import _parse_income_tax_ordinance
        previous_chapter = None
        previous_section = None
        if start_page > 1:
            chapters = [n for n in engine.nodes.values() if n.type == "chapter" and (n.page_end or 0) < start_page]
            if chapters:
                previous_chapter = max(chapters, key=lambda n: n.page_end or 0).heading
            sections = [n for n in engine.nodes.values() if n.type == "section" and (n.page_end or 0) < start_page]
            if sections:
                latest = max(sections, key=lambda n: n.page_end or 0)
                previous_section = (latest.citation, latest.heading)
        partial = _parse_income_tax_ordinance(data, ORDINANCE.name, start_page, end_page,
                                              previous_chapter, previous_section)
        with SOURCE_LOCK:
            _merge_partial(partial)
            SOURCE_LOADED_PAGES = end_page
            SOURCE_STATUS = "ready" if end_page >= SOURCE_TOTAL_PAGES else "partial"
            SOURCE_ERROR = None
    except Exception as exc:
        SOURCE_ERROR = str(exc)
        SOURCE_STATUS = "error"


def _load_ordinance_background():
    try:
        from pypdf import PdfReader
        global SOURCE_TOTAL_PAGES
        SOURCE_TOTAL_PAGES = len(PdfReader(str(ORDINANCE)).pages)
        _load_chunk(1)
    except Exception as exc:
        global SOURCE_ERROR, SOURCE_STATUS
        SOURCE_ERROR = str(exc)
        SOURCE_STATUS = "error"


if ORDINANCE.exists():
    Thread(target=_load_ordinance_background, name="ordinance-loader", daemon=True).start()


def source_status():
    return {"status": SOURCE_STATUS, "error": SOURCE_ERROR,
            "documents": len(engine.docs), "loaded_pages": SOURCE_LOADED_PAGES,
            "total_pages": SOURCE_TOTAL_PAGES, "chunk_pages": CHUNK_PAGES}


class SearchReq(BaseModel):
    query: str
    model: str = "bm25"
    scope: str = "document"          # document | region
    structure: str = "plain"         # plain | overlapped | non_overlapped | proximal (region scope only)
    fields: Optional[Dict[str, float]] = None
    k: int = 10
    k1: float = 1.2
    b: float = 0.75
    relevant: List[str] = []
    hops: int = 2
    decay: float = 0.5
    minimal: bool = True
    sublinear: bool = False


class TextReq(BaseModel):
    title: str = ""
    text: str


class ConfigReq(BaseModel):
    stem: bool = True
    stopwords: bool = True


def state(errors=None):
    return {"stats": engine.stats(),
            "docs": [{"id": d.id, "title": d.title, "regions": len(d.nodes), "meta": d.meta} for d in engine.docs.values()],
            "source_status": SOURCE_STATUS,
            "source_progress": source_status(),
            "errors": errors or ([SOURCE_ERROR] if SOURCE_ERROR else [])}


def node_item(n):
    return {"id": n.id, "title": n.heading or n.text[:100], "type": n.type,
            "citation": n.citation, "path": n.path, "page_start": n.page_start,
            "page_end": n.page_end, "children": len(n.children), "doc_id": n.doc_id}


def ancestors(node_id):
    chain = []
    node = engine.nodes.get(node_id)
    while node:
        chain.append(node)
        node = engine.nodes.get(node.parent) if node.parent else None
    return [node_item(n) for n in reversed(chain)]


def browse_metadata_facets():
    """Return exact metadata values currently available in the loaded collection."""
    facets = {}
    for doc in engine.docs.values():
        for key, value in doc.meta.items():
            value = str(value).strip()
            if value:
                facets.setdefault(str(key), set()).add(value)
    return {key: sorted(values, key=str.casefold) for key, values in
            sorted(facets.items(), key=lambda pair: pair[0].casefold())}


def matches_browse_metadata(doc_id, field, value):
    if not field or not value:
        return True
    doc = engine.docs.get(doc_id)
    if not doc:
        return False
    wanted_field, wanted_value = field.strip().casefold(), value.strip().casefold()
    return any(str(key).strip().casefold() == wanted_field and
               str(actual).strip().casefold() == wanted_value
               for key, actual in doc.meta.items())


def node_full_text(node):
    """Return a node's own text plus its structured descendants in reading order."""
    parts = [node.text.strip()] if node.text.strip() else []
    for child_id in node.children:
        child = engine.nodes[child_id]
        body = node_full_text(child)
        label = child.heading or child.citation
        piece = f"{label} {body}".strip() if label else body
        if piece:
            parts.append(piece)
    return "\n\n".join(parts)


def related_nodes(node_id, limit=8):
    node = engine.nodes[node_id]
    while node.type != "section" and node.parent:
        node = engine.nodes[node.parent]
    if node.type != "section":
        return []

    def section_text(section):
        chunks = [section.text]
        stack = list(section.children)
        while stack:
            child = engine.nodes[stack.pop()]
            chunks.append(child.text)
            stack.extend(child.children)
        return " ".join(chunks)

    source_text = section_text(node)
    sections = {n.citation.lower(): n for n in engine.nodes.values() if n.type == "section" and n.citation}
    refs = re.findall(r"\bsections?\s+(\d+[a-z]{0,3})\b", source_text, re.I)
    related, seen = [], {node.id}
    for ref in refs:
        hit = sections.get(ref.lower())
        if hit and hit.id not in seen:
            seen.add(hit.id)
            related.append(hit)
    # Include incoming cross-references when the section itself has few outgoing links.
    for other in sections.values():
        if len(related) >= limit:
            break
        if other.id in seen:
            continue
        if re.search(rf"\bsections?\s+{re.escape(node.citation)}\b", section_text(other), re.I):
            seen.add(other.id)
            related.append(other)
    if not related:
        parent = engine.nodes.get(node.parent)
        if parent:
            related = [engine.nodes[c] for c in parent.children
                       if c != node.id and engine.nodes[c].type == "section"][:limit]
    return [node_item(n) for n in related[:limit]]


def kwargs(r: SearchReq, drop=("query", "k", "model")):
    return {k: v for k, v in r.dict().items() if k not in drop}


@app.get("/")
def home():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/state")
def get_state():
    return state()


@app.post("/api/browse/load-ordinance")
def browse_load_ordinance():
    global SOURCE_STATUS, SOURCE_ERROR, SOURCE_TOTAL_PAGES
    if not ORDINANCE.exists():
        raise HTTPException(404, "The supplied Income Tax Ordinance PDF was not found in Material/.")
    if SOURCE_STATUS in ("loading", "partial", "ready"):
        return source_status()
    SOURCE_STATUS, SOURCE_ERROR = "loading", None
    Thread(target=_load_ordinance_background, name="ordinance-reloader", daemon=True).start()
    return source_status()


@app.post("/api/browse/load-next")
def browse_load_next():
    global SOURCE_STATUS
    if SOURCE_STATUS == "loading":
        return source_status()
    if SOURCE_STATUS == "ready":
        return source_status()
    if SOURCE_STATUS == "error" and SOURCE_LOADED_PAGES >= SOURCE_TOTAL_PAGES:
        raise HTTPException(500, SOURCE_ERROR or "The Ordinance could not be loaded.")
    SOURCE_STATUS = "loading"
    Thread(target=_load_chunk, args=(SOURCE_LOADED_PAGES + 1,), name="ordinance-chunk-loader", daemon=True).start()
    return source_status()


@app.get("/api/browse/status")
def browse_status():
    return source_status()


@app.get("/api/browse/filters")
def browse_filters():
    need_docs()
    return {"fields": browse_metadata_facets()}


@app.get("/api/browse")
def browse(parent_id: Optional[str] = None, q: str = "", kind: Optional[str] = None,
           meta_field: Optional[str] = None, meta_value: Optional[str] = None,
           limit: int = 50, offset: int = 0):
    if SOURCE_STATUS == "loading" and not engine.docs:
        raise HTTPException(503, "The Ordinance is still loading.")
    if SOURCE_STATUS == "error" and not engine.docs:
        raise HTTPException(503, SOURCE_ERROR or "The Ordinance could not be read.")
    need_docs()
    limit = max(1, min(limit, 100))
    offset = max(0, offset)
    if q.strip():
        # Filter before pagination so matches on later pages are not lost.
        ranked = engine._run(q.strip(), model="bm25", scope="region")
        items = []
        for uid in ranked["order"]:
            hit = engine._item(ranked, uid)
            node = engine.nodes.get(uid)
            if (node and (not kind or node.type == kind) and
                    matches_browse_metadata(node.doc_id, meta_field, meta_value)):
                item = node_item(node)
                item["snippet"] = hit.get("snippet", "")
                item["score"] = hit.get("score")
                item["breadcrumb"] = ancestors(node.id)
                items.append(item)
        total = len(items)
        return {"mode": "search", "query": q, "items": items[offset:offset + limit], "total": total,
                "offset": offset, "limit": limit, "breadcrumb": [], "documents": []}

    if parent_id:
        parent = engine.nodes.get(parent_id)
        if not parent:
            raise HTTPException(404, "Browse location not found.")
        items = [node_item(engine.nodes[c]) for c in parent.children
                 if matches_browse_metadata(engine.nodes[c].doc_id, meta_field, meta_value)]
        breadcrumb = ancestors(parent_id)
        documents = []
    else:
        if kind and kind != "document":
            items = [node_item(n) for n in engine.nodes.values() if n.type == kind]
        else:
            items = [{"id": d.id, "title": d.title, "type": "document", "citation": "",
                      "path": d.title, "page_start": 1, "page_end": None, "children": len(d.nodes),
                      "doc_id": d.id} for d in engine.docs.values()]
        breadcrumb, documents = [], items
        if kind:
            items = [item for item in items if item["type"] == kind]
        if meta_field and meta_value:
            items = [item for item in items if matches_browse_metadata(item["doc_id"], meta_field, meta_value)]
    total = len(items)
    return {"mode": "browse", "query": "", "items": items[offset:offset + limit], "total": total,
            "offset": offset, "limit": limit,
            "breadcrumb": breadcrumb, "documents": documents}


@app.get("/api/browse/document/{doc_id}")
def browse_document(doc_id: str):
    doc = engine.docs.get(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found.")
    root = next((n for n in doc.nodes if n.type == "document"), None)
    children = [node_item(engine.nodes[c]) for c in root.children] if root else []
    return {"document": {"id": doc.id, "title": doc.title, "meta": doc.meta},
            "node": node_item(root) if root else None, "children": children,
            "breadcrumb": [node_item(root)] if root else []}


@app.get("/api/browse/section/{citation}")
def browse_section(citation: str):
    wanted = citation.strip().lower()
    node = next((n for n in engine.nodes.values()
                 if n.type == "section" and n.citation.strip().lower() == wanted), None)
    if not node:
        raise HTTPException(404, f"Section {citation} is not in the currently loaded pages. Load more Ordinance pages, then try again.")
    return {"id": node.id}


@app.get("/api/browse/node/{node_id}")
def browse_node(node_id: str, offset: int = 0, limit: int = 50,
                meta_field: Optional[str] = None, meta_value: Optional[str] = None):
    node = engine.nodes.get(node_id)
    if not node:
        raise HTTPException(404, "Browse location not found.")
    children = [node_item(engine.nodes[c]) for c in node.children
                if matches_browse_metadata(engine.nodes[c].doc_id, meta_field, meta_value)]
    related = [item for item in related_nodes(node_id)
               if matches_browse_metadata(item["doc_id"], meta_field, meta_value)]
    offset, limit = max(0, offset), max(1, min(limit, 100))
    return {"node": node_item(node), "text": node_full_text(node),
            "children": children[offset:offset + limit], "children_total": len(children),
            "breadcrumb": ancestors(node_id), "related": related}


@app.get("/api/browse/source.pdf")
def browse_source_pdf():
    if not ORDINANCE.exists():
        raise HTTPException(404, "The supplied Income Tax Ordinance PDF was not found in Material/.")
    return FileResponse(ORDINANCE, media_type="application/pdf",
                        headers={"Content-Disposition": 'inline; filename="Income-Tax-Ordinance-2001.pdf"'})


@app.post("/api/upload")
async def upload(files: List[UploadFile] = File(...)):
    errors = []
    for f in files:
        try:
            if not engine.add_file(f.filename, await f.read()):
                errors.append(f"{f.filename}: no documents found")
        except Exception as e:                      # bad PDF, bad JSON, missing library ...
            errors.append(f"{f.filename}: {e}")
    return state(errors)


@app.post("/api/agent/analyze")
async def analyze_upload(file: UploadFile = File(...)):
    """Return advisory Gemini/local-fallback metadata without changing the corpus."""
    try:
        return {"filename": file.filename, "metadata": analyze_source(file.filename, await file.read())}
    except Exception as exc:
        raise HTTPException(422, f"Could not analyze {file.filename}: {exc}") from exc


@app.post("/api/agent/ingest")
async def ingest_with_agent(file: UploadFile = File(...)):
    """Structure a source with Gemini/local fallback and add it to the collection."""
    try:
        data = await file.read()
        documents, method = structure_source(file.filename, data)
        ids = [engine.add_document(document) for document in documents]
        return {"filename": file.filename, "method": method, "ids": ids, **state()}
    except Exception as exc:
        raise HTTPException(422, f"Could not structure {file.filename}: {exc}") from exc


@app.post("/api/text")
def add_text(r: TextReq):
    if not r.text.strip():
        raise HTTPException(400, "Paste some text first.")
    engine.add_text(r.text, r.title)
    return state()


@app.post("/api/sample")
def sample():
    engine.load_sample()
    return state()


@app.delete("/api/docs/{doc_id}")
def delete(doc_id: str):
    engine.remove(doc_id)
    return state()


@app.post("/api/clear")
def clear():
    engine.clear()
    return state()


@app.post("/api/config")
def config(r: ConfigReq):
    engine.configure(r.stem, r.stopwords)
    return state()


def need_docs():
    if not engine.docs:
        raise HTTPException(400, "The collection is empty. Load the sample data or add your own documents first.")


@app.post("/api/search")
def search(r: SearchReq):
    need_docs()
    if r.model not in MODELS:
        raise HTTPException(400, f"Unknown model {r.model}")
    return engine.search(r.query, k=r.k, model=r.model, **kwargs(r, ("query", "k", "model")))


@app.post("/api/compare")
def compare(r: SearchReq):
    need_docs()
    return engine.compare(r.query, k=min(r.k, 8), **kwargs(r, ("query", "k", "model", "relevant")))


@app.get("/api/evaluate")
def evaluate():
    need_docs()
    # Relevance judgments belong to the seven-document teaching corpus, not
    # the user's uploaded/Ordinance collection. Keep evaluation isolated so
    # it never changes the active search corpus.
    evaluation_engine = Engine(stem=engine.stem, stopwords=engine.stopwords)
    evaluation_engine.load_sample()
    ev = evaluation_engine.evaluate(SAMPLE_JUDGMENTS)
    return {"text": format_evaluation(ev), "detail": ev}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
