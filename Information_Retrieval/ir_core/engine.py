"""Engine: the facade the UI / CLI / tests talk to.

Lifecycle:  start EMPTY -> add documents at run time -> (re)build indexes lazily -> search / compare / evaluate.
Rebuilding is cheap and keeps IDF correct, because IDF depends on the WHOLE collection.
"""
import html, re, time
from difflib import SequenceMatcher
from . import structured as S
from .analyzer import Analyzer, TOKEN_RE
from .boolean import RE_QTOK, Parser, evaluate, has_ops, positive_terms, pretty
from .documents import parse_file, parse_text
from .index import Index
from .ranking import rank

MODELS = ("fuzzy", "extended_boolean", "boolean", "bim", "tfidf", "bm25")
DOC_FIELDS = {"title": 2.0, "meta": 1.5, "body": 1.0}
REGION_FIELDS = {"heading": 2.0, "text": 1.0}
FORMULA = {
    "fuzzy": "Fuzzy retrieval: each term gets a membership in [0, 1] from exact, partial, or typo similarity. AND = min, OR = max, NOT = 1 - membership.",
    "extended_boolean": "Extended Boolean: AND = intersection, OR = union, and NOT = set difference, with explicit include/exclude semantics.",
    "boolean": "match / no match: AND = intersection, OR = union, NOT = complement. No scores, so no ranking.",
    "bim": "BIM: sum over query terms PRESENT in the unit of log(1 + p/(1-p) * (1-u)/u). Term frequency is ignored; rare terms weigh more.",
    "tfidf": "VSM (TF-IDF + cosine): weight = tf * log10(N/df); score = (q . d) / (|q| |d|).",
    "bm25": "BM25: sum of idf * tf*(k1+1) / (tf + k1*(1-b+b*len/avglen)). Repeats saturate; long units are discounted.",
    
}


def _fuzzy_similarity(query_term, indexed_term):
    """Return a useful partial-match membership for two normalized terms."""
    if query_term == indexed_term:
        return 1.0
    if query_term in indexed_term or indexed_term in query_term:
        return min(0.95, min(len(query_term), len(indexed_term)) /
                   max(len(query_term), len(indexed_term)) + 0.15)
    return SequenceMatcher(None, query_term, indexed_term).ratio()


def snippet(an, text, qtokens, width=240):
    qs = set(qtokens)
    spans = [(m.start(), m.end()) for m in TOKEN_RE.finditer(text)
             if (lambda t: t and t[0] in qs)(an.terms(m.group(), query=True))]
    start = max(0, spans[0][0] - width // 3) if spans else 0
    end = min(len(text), start + width)
    out, pos = [], start
    for s, e in spans:
        if s >= start and e <= end:
            out += [html.escape(text[pos:s]), "<mark>" + html.escape(text[s:e]) + "</mark>"]
            pos = e
    out.append(html.escape(text[pos:end]))
    return ("…" if start > 0 else "") + "".join(out) + ("…" if end < len(text) else "")


class Engine:
    def __init__(self, stem=True, stopwords=True):
        self.stem, self.stopwords = stem, stopwords
        self.docs, self.nodes, self._dirty = {}, {}, True

    # ---------- corpus (starts empty) ----------
    @property
    def an(self):
        return Analyzer(self.stem, self.stopwords)

    def configure(self, stem=None, stopwords=None):
        self.stem = self.stem if stem is None else bool(stem)
        self.stopwords = self.stopwords if stopwords is None else bool(stopwords)
        self._dirty = True

    def _unique_id(self, hint, title):
        base = re.sub(r"[^a-z0-9]+", "-", (hint or title).lower()).strip("-")[:40] or "doc"
        cand, i = base, 1
        while cand in self.docs:
            i += 1
            cand = f"{base}-{i}"
        return cand

    def add_document(self, doc):
        doc.assign(self._unique_id(doc.hint, doc.title))
        self.docs[doc.id] = doc
        self.nodes.update({n.id: n for n in doc.nodes})
        self._dirty = True
        return doc.id

    def add_file(self, filename, data):
        return [self.add_document(d) for d in parse_file(filename, data)]

    def add_text(self, text, title=""):
        return self.add_document(parse_text(text, title or "Pasted text"))

    def load_sample(self):
        from .sample_data import SAMPLE_DOCS
        ids = []
        for t in SAMPLE_DOCS:
            d = parse_text(t, "sample")
            if d.hint not in self.docs:
                ids.append(self.add_document(d))
        return ids

    def remove(self, doc_id):
        d = self.docs.pop(doc_id, None)
        for n in (d.nodes if d else []):
            self.nodes.pop(n.id, None)
        self._dirty = True

    def clear(self):
        self.docs, self.nodes, self._dirty = {}, {}, True

    # ---------- indexing ----------
    def _build(self):
        if not self._dirty:
            return
        an = self.an
        self.idx_doc = Index(["title", "meta", "body"])
        for d in self.docs.values():
            self.idx_doc.add(d.id, {"title": an.tokens(d.title), "meta": an.tokens(d.meta_text()),
                                    "body": an.tokens(d.body())})
        self.idx_own, self.idx_sub, self.sub_text = Index(["heading", "text"]), Index(["heading", "text"]), {}
        for n in reversed(list(self.nodes.values())):      # children before parents
            self.sub_text[n.id] = (n.text + " " + " ".join(
                self.nodes[c].heading + " " + self.sub_text[c] for c in n.children)).strip()
        for n in self.nodes.values():
            h = an.tokens(n.heading)
            self.idx_own.add(n.id, {"heading": h, "text": an.tokens(n.text)})
            self.idx_sub.add(n.id, {"heading": h, "text": an.tokens(self.sub_text[n.id])})
        self.adj = S.build_adjacency(self.nodes)
        self._dirty = False

    def stats(self):
        self._build()
        return {"documents": len(self.docs), "regions": len(self.nodes),
                "vocabulary": len(self.idx_doc.post["body"]), "stem": self.stem, "stopwords": self.stopwords}

    # ---------- search ----------
    def _run(self, query, model="bm25", scope="document", structure="plain", fields=None,
             k1=1.2, b=0.75, relevant=None, hops=2, decay=0.5, minimal=True, sublinear=False):
        self._build()
        an, notes = self.an, []
        if scope == "region":
            idx, base = (self.idx_sub if structure == "overlapped" else self.idx_own), REGION_FIELDS
        else:
            idx, base, structure = self.idx_doc, DOC_FIELDS, "plain"
        fw = {f: float(w) for f, w in {**base, **(fields or {})}.items() if f in idx.fields and float(w) > 0} or dict(base)
        names = list(fw)
        ast = Parser(query, an, idx.fields).parse()
        terms = positive_terms(ast)
        cand = (evaluate(ast, idx, names) or set()) if (has_ops(query) or model in ("boolean", "extended_boolean")) else None
        known = set(idx.uids)
        rel = [r for r in (relevant or []) if r in known]
        own, extra, lists = {}, {}, {}
        if model == "fuzzy":
            def membership(node, uid):
                kind = node[0]
                if kind == "skip":
                    return 0.0
                if kind == "term":
                    fs = [node[1]] if node[1] else names
                    best = 0.0
                    for f in fs:
                        for indexed in idx.post[f]:
                            best = max(best, _fuzzy_similarity(node[2], indexed)
                                       if uid in idx.post[f][indexed] else 0.0)
                    return best if best >= 0.45 else 0.0
                if kind == "phrase":
                    values = [membership(("term", node[1], t), uid) for t in node[2]]
                    return min(values) if values else 0.0
                if kind == "not":
                    return 1.0 - membership(node[1], uid)
                values = [membership(child, uid) for child in node[1]]
                if not values:
                    return 0.0
                return min(values) if kind == "and" else max(values)

            fuzzy_scores = {u: membership(ast, u) for u in idx.uids}
            scores = {u: s for u, s in fuzzy_scores.items() if s > 0}
            parts = {}
            for u in scores:
                parts[u] = {t: round(membership(("term", None, t), u), 4)
                            for t in dict.fromkeys(terms)}
        elif model in ("boolean", "extended_boolean"):
            matched = cand
            if scope == "region" and structure == "non_overlapped":
                matched, lists = S.non_overlapped_lists(idx, terms, names)
            elif scope == "region" and structure == "overlapped":
                matched = S.prune_minimal(cand, self.nodes) if minimal else cand
            elif scope == "region" and structure == "proximal":
                notes.append("Proximal nodes needs scores - use BIM / VSM (TF-IDF + cosine) / BM25. Showing plain Boolean matches.")
            scores, parts = {u: 1.0 for u in idx.uids if u in matched}, {}
        else:
            if scope == "region" and structure == "overlapped":
                c2 = S.overlapped_candidates(idx, self.nodes, terms, names, minimal)
                cand = c2 if cand is None else c2 & cand
            elif scope == "region" and structure == "non_overlapped":
                union, lists = S.non_overlapped_lists(idx, terms, names)
                cand = union if cand is None else union & cand
            scores, parts = rank(idx, model, terms, fw, k1, b, cand, rel, sublinear)
            if scope == "region" and structure == "proximal":
                own = dict(scores)
                scores, extra = S.proximal(own, self.adj, hops, decay)
        pos = {u: i for i, u in enumerate(idx.uids)}
        order = list(scores) if model in ("boolean", "extended_boolean") else sorted(scores, key=lambda u: (-scores[u], pos[u]))
        return dict(idx=idx, names=names, fw=fw, ast=ast, terms=terms, scores=scores, parts=parts, order=order,
                    own=own, extra=extra, lists=lists, notes=notes, scope=scope, structure=structure,
                    model=model, k1=k1, b=b, query=query, an=an)

    def _item(self, r, uid):
        idx, an, terms = r["idx"], r["an"], r["terms"]
        qset, uniq = set(terms), list(dict.fromkeys(terms))
        matched = [t for t in uniq if any(idx.counts[f][uid].get(t) for f in r["names"])]
        item = {"id": uid, "score": round(r["scores"][uid], 4), "matched": matched,
                "breakdown": {t: round(c, 4) for t, c in r["parts"].get(uid, {}).items()}}
        if r["scope"] == "document":
            d = self.docs[uid]
            best = max(d.nodes, key=lambda n: len(qset & set(an.terms(n.heading + " " + n.text)))) if d.nodes else None
            txt = (best.heading + ". " + best.text) if best else ""
            item.update(kind="document", title=d.title, meta=d.meta, snippet=snippet(an, txt, terms),
                        where=best.path if best and best.path else "document")
        else:
            n = self.nodes[uid]
            d = self.docs[n.doc_id]
            txt = (n.heading + ". " + (self.sub_text[uid] if r["structure"] == "overlapped" else n.text)).strip()
            item.update(kind="region", doc_id=d.id, title=d.title, type=n.type, path=n.path or "document root",
                        meta=d.meta, snippet=snippet(an, txt, terms))
            if r["structure"] == "proximal":
                item["own_score"] = round(r["own"].get(uid, 0.0), 4)
                via = sorted(r["extra"].get(uid, []), key=lambda x: -x[2])[:3]
                item["via"] = [{"id": m, "path": self.nodes[m].path or "document root", "doc": self.docs[self.nodes[m].doc_id].title,
                                "hops": h, "contribution": round(c, 4)} for m, h, c in via]
        return item

    def search(self, query, k=10, **kw):
        t0 = time.perf_counter()
        r = self._run(query, **kw)
        items = [self._item(r, u) for u in r["order"][:k]]
        idx, an = r["idx"], r["an"]
        words = [w for w in RE_QTOK.findall(query) if w not in ("AND", "OR", "NOT", "(", ")")]
        trace = {
            "query": query, "ast": pretty(r["ast"]), "terms": r["terms"], "model": r["model"], "scope": r["scope"],
            "structure": r["structure"], "N": idx.N, "fields": r["fw"], "avgdl": {f: round(idx.avgdl(f), 2) for f in r["names"]},
            "analysis": [{"word": w, "tokens": an.terms(w.split(":", 1)[-1].strip('"'), query=True)} for w in words],
            "term_stats": [{"term": t, "units": len(idx.docs_with(t, r["names"])),
                            "df": {f: idx.df(t, f) for f in r["names"]}} for t in dict.fromkeys(r["terms"])],
            "candidates": len(r["scores"]), "formula": FORMULA[r["model"]], "notes": r["notes"],
            "params": {"k1": r["k1"], "b": r["b"]},
            "lists": {t: sorted(self.nodes[u].path or "document root" for u in v)[:12] for t, v in r["lists"].items()},
        }
        return {"results": items, "trace": trace, "explain": self._explain(trace, items), "total": len(r["scores"]),
                "took_ms": round((time.perf_counter() - t0) * 1000, 2)}

    # ---------- the Feynman explanation ----------
    def _explain(self, tr, items):
        label = lambda it: it["title"] if it["kind"] == "document" else f'{it["title"]} › {it["path"]}'
        terms = list(dict.fromkeys(tr["terms"]))
        dropped = [a["word"] for a in tr["analysis"] if not a["tokens"]]
        inp = f'You typed "{tr["query"]}". The system keeps these index terms: {terms or "none"}.'
        if dropped:
            inp += f" It dropped {dropped} (stop-words carry no meaning)."
        inside = ["Cleaned the query with the same analyzer used on the documents (lowercase, stop-words, light stemming).",
                  "Looked each term up in the inverted index (no document is read): " +
                  "; ".join(f'"{s["term"]}" occurs in {s["units"]} of {tr["N"]} units' for s in tr["term_stats"]) + "."]
        if tr["structure"] != "plain":
            inside.append({"overlapped": "Overlapped lists: kept only regions that contain ALL terms inside the same region (most specific first).",
                           "non_overlapped": "Non-overlapped lists: each term has its own list of regions; a region qualifies if it is in ANY list.",
                           "proximal": "Proximal nodes: each region's score = its own score + discounted scores of connected nearby regions."}[tr["structure"]])
        inside.append(tr["formula"])
        inside += tr["notes"]
        out = (f'{len(items)} of {tr["candidates"]} matching units shown.' if items else
               "No unit matched. Try fewer terms, OR instead of AND, or check spelling.")
        if not items:
            why = "Nothing to rank."
        elif tr["model"] in ("boolean", "extended_boolean"):
            why = "Every result satisfies the Boolean condition and the model gives no scores, so none is 'more relevant' - that is exactly the Boolean limitation."
        else:
            top = items[0]
            why = f'"{label(top)}" is first with score {top["score"]}: it matches {len(top["matched"])} of {len(terms)} query terms'
            if top["breakdown"]:
                bt, bc = max(top["breakdown"].items(), key=lambda kv: kv[1])
                u = next((s["units"] for s in tr["term_stats"] if s["term"] == bt), "?")
                why += f', and the biggest contribution comes from "{bt}" ({bc}), found in only {u} of {tr["N"]} units - rarer terms weigh more.'
            if len(items) > 1:
                sec = items[1]
                miss = [t for t in top["matched"] if t not in sec["matched"]]
                why += (f' The runner-up scores {sec["score"]} and lacks {miss}.' if miss else
                        f' The runner-up scores {sec["score"]}: it matches the same terms but with weaker evidence (fewer repeats or a longer unit that dilutes them).')
            if top.get("via"):
                why += " It also got a boost from connected nodes: " + ", ".join(v["path"] for v in top["via"]) + "."
        return {"input": inp, "inside": inside, "output": out, "why_first": why}

    # ---------- compare + evaluate ----------
    def compare(self, query, models=MODELS, k=5, **kw):
        res = {m: self.search(query, model=m, k=k, **kw) for m in models}
        ids = {m: [x["id"] for x in res[m]["results"]] for m in models}
        pairs = [("fuzzy", "extended_boolean"), ("extended_boolean", "boolean"),
                 ("boolean", "bim"), ("bim", "bm25"), ("tfidf", "bm25"), ("bim", "tfidf")]
        return {"models": {m: {"results": res[m]["results"], "total": res[m]["total"]} for m in models},
                "agreement": [{"a": a, "b": b, "overlap": len(set(ids[a]) & set(ids[b])),
                               "same_top": bool(ids[a] and ids[b] and ids[a][0] == ids[b][0])}
                              for a, b in pairs if a in ids and b in ids]}

    def evaluate(self, judged, models=MODELS, k=3):
        table = {}
        for m in models:
            rows = []
            for q, rel in judged.items():
                rel, r = set(rel), self._run(q, model=m)
                ids = r["order"]
                top = ids if m in ("boolean", "extended_boolean") else ids[:k]
                hit = len(set(top) & rel)
                predicted, universe = set(top), set(self._run(q, model=m)["idx"].uids)
                tp = len(predicted & rel)
                fp = len(predicted - rel)
                fn = len(rel - predicted)
                tn = max(0, len(universe) - tp - fp - fn)
                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / (tp + fn) if tp + fn else 0.0
                f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
                ap, seen = 0.0, 0
                for i, u in enumerate(ids, 1):
                    if u in rel:
                        seen += 1
                        ap += seen / i
                rows.append({"query": q, "P": precision, "R": recall, "F1": f1,
                             "Accuracy": (tp + tn) / len(universe) if universe else 0.0,
                             "AP": None if m in ("boolean", "extended_boolean") else ap / len(rel) if rel else 0.0,
                             "returned": len(top)})
            mean = lambda key: (sum(x[key] for x in rows) / len(rows)) if rows and rows[0][key] is not None else None
            table[m] = {"rows": rows, "P": mean("P"), "R": mean("R"), "F1": mean("F1"),
                        "Accuracy": mean("Accuracy"), "MAP": mean("AP")}
        return {"k": k, "table": table}


def format_evaluation(ev):
    f = lambda x: "  -  " if x is None else f"{x:5.2f}"
    lines = [f'Classification metrics at k={ev["k"]} (Boolean models use their complete returned set)',
             f'{"model":16} {"P":>6} {"R":>6} {"F1":>6} {"Acc":>6}']
    for m, t in ev["table"].items():
        lines.append(f'{m:16} {f(t["P"])} {f(t["R"])} {f(t["F1"])} {f(t["Accuracy"])}')
    return "\n".join(lines)
