"""Ranking models. All three score a unit from the index postings of the query terms.

  bim   : Binary Independence Model.  score = sum of term weights for terms PRESENT (tf ignored).
          weight = log(1 + odds)  with odds = p/(1-p) * (1-u)/u,
          p = P(term | relevant), u = P(term | non-relevant). With no feedback p = 0.5, so the
          weight is a smoothed IDF; with relevance feedback p and u are re-estimated.
  tfidf : Vector Space Model. cosine( query vector , doc vector ) with  w = tf * log10(N/df).
  bm25  : score = sum idf * tf*(k1+1) / (tf + k1*(1-b+b*len/avglen)).

Fielded scoring: each field (title/meta/body...) is scored on its own statistics, then the
field scores are combined with the field weights (BM25F-lite).
"""
import math
from collections import Counter, defaultdict


def idf_prob(N, n):
    return math.log(1 + (N - n + 0.5) / (n + 0.5))


def _doc_norm(index, f, uid, sub):
    key = (f, uid, sub)
    if key not in index._norm:
        s = 0.0
        for t, c in index.counts[f][uid].items():
            idf = math.log10(index.N / index.df(t, f))
            s += (((1 + math.log10(c)) if sub else c) * idf) ** 2
        index._norm[key] = math.sqrt(s) or 1.0
    return index._norm[key]


def rank(index, model, terms, fields, k1=1.2, b=0.75, candidates=None, relevant=None, sublinear=False):
    """Return (scores {uid: score}, parts {uid: {term: contribution}})."""
    N, qtf = index.N, Counter(terms)
    scores, parts = defaultdict(float), defaultdict(lambda: defaultdict(float))
    rel = set(relevant or ())
    for f, w in fields.items():
        avg = index.avgdl(f) or 1.0
        qnorm = 1.0
        if model == "tfidf":
            qnorm = math.sqrt(sum((c * math.log10(N / index.df(t, f))) ** 2
                                  for t, c in qtf.items() if index.df(t, f))) or 1.0
        fs, fp = defaultdict(float), defaultdict(lambda: defaultdict(float))
        for t, qc in qtf.items():
            plist = index.post[f].get(t)
            if not plist:
                continue
            n = len(plist)
            if model == "bim":
                R, r = len(rel), sum(1 for u in rel if u in plist)
                p, u_ = (r + 0.5) / (R + 1), (n - r + 0.5) / (N - R + 1)
                wt = math.log(1 + (p / (1 - p)) * ((1 - u_) / u_))
            elif model == "bm25":
                idf = idf_prob(N, n)
            else:
                idf = math.log10(N / n)
            for uid, pos in plist.items():
                if candidates is not None and uid not in candidates:
                    continue
                tf = len(pos)
                if model == "bim":
                    c = wt
                elif model == "bm25":
                    c = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * index.length[f][uid] / avg))
                else:
                    tfw = (1 + math.log10(tf)) if sublinear else tf
                    c = (tfw * idf) * (qc * idf) / (qnorm * _doc_norm(index, f, uid, sublinear))
                fs[uid] += c
                fp[uid][t] += c
        for uid, s in fs.items():
            scores[uid] += w * s
            for t, c in fp[uid].items():
                parts[uid][t] += w * c
    return dict(scores), parts
