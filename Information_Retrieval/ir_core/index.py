"""The inverted index: term -> {unit_id: [positions]}, kept separately for every FIELD (title, body, ...).
A "unit" is whatever we retrieve: a whole document, or a single region (node) of a document."""
from collections import Counter, defaultdict


class Index:
    def __init__(self, fields):
        self.fields = list(fields)
        self.post = {f: defaultdict(dict) for f in self.fields}   # field -> term -> {uid: [positions]}
        self.counts = {f: {} for f in self.fields}                # field -> uid -> Counter(term)
        self.length = {f: {} for f in self.fields}                # field -> uid -> length in tokens
        self.uids = []
        self._norm = {}

    def add(self, uid, field_tokens):
        self.uids.append(uid)
        for f in self.fields:
            c, n = Counter(), 0
            for t, p in field_tokens.get(f, []):
                self.post[f][t].setdefault(uid, []).append(p)
                c[t] += 1
                n = max(n, p + 1)
            self.counts[f][uid], self.length[f][uid] = c, n

    @property
    def N(self):
        return len(self.uids)

    def avgdl(self, f):
        v = self.length[f].values()
        return sum(v) / len(v) if v else 0.0

    def df(self, term, f):
        return len(self.post[f].get(term, ()))

    def docs_with(self, term, fields):
        out = set()
        for f in fields:
            out |= set(self.post[f].get(term, ()))
        return out

    def phrase_docs(self, tokens, fields):
        """Units where the tokens occur consecutively (needs the stored positions)."""
        out = set()
        for f in fields:
            pls = [self.post[f].get(t) for t in tokens]
            if any(p is None for p in pls):
                continue
            for uid in set(pls[0]).intersection(*pls[1:]):
                starts = set(pls[0][uid])
                for i, pl in enumerate(pls[1:], 1):
                    nxt = set(pl[uid])
                    starts = {s for s in starts if s + i in nxt}
                    if not starts:
                        break
                if starts:
                    out.add(uid)
        return out
