"""Analyzer: raw text -> index terms.

RULE #1 of search: the SAME analyzer must process documents and queries,
otherwise "running" is indexed as "run" but searched as "running" and nothing matches.
"""
import re

STOPWORDS = frozenset("""a an and are as at be been but by for from had has have he her his if in into is it its of on or
our she that the their them then there these they this to was we were what when where which who will with you your
shall may would should could than so such also any upon thereof herein hereof""".split())
# "not", "no", "nor" are deliberately KEPT: negation matters in legal text.

# 153(1)(b) stays ONE token (legal citation) | plain words | numbers
TOKEN_RE = re.compile(r"\d+[A-Za-z]?(?:\([A-Za-z0-9]+\))+|[A-Za-z]+|\d+(?:,\d{3})*(?:\.\d+)?")
VOWEL = re.compile(r"[aeiouy]")


def light_stem(w: str) -> str:
    """A tiny rule-based stemmer (plurals, -ing, -ed, trailing -e). Swap for Porter/Snowball if you like."""
    if len(w) <= 3 or not w.isalpha():
        return w
    if w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith(("xes", "ches", "shes", "zes")):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    for suf in ("ing", "ed"):
        if w.endswith(suf) and len(w) - len(suf) >= 3 and VOWEL.search(w[:-len(suf)]):
            w = w[:-len(suf)]
            if len(w) > 2 and w[-1] == w[-2] and w[-1] not in "lsz":
                w = w[:-1]
            break
    if w.endswith("e") and len(w) > 4:
        w = w[:-1]
    return w


class Analyzer:
    def __init__(self, stem: bool = True, stopwords: bool = True):
        self.stem, self.stopwords = stem, stopwords

    def tokens(self, text: str, query: bool = False):
        """Return [(token, position)]. For documents, a citation like 153(1)(b) also emits
        its prefixes 153(1) and 153 at the same position, so a search for "153" finds it."""
        out, pos = [], 0
        for m in TOKEN_RE.finditer(text or ""):
            raw = m.group().lower().replace(",", "")
            if raw.isalpha():
                if len(raw) < 2 or (self.stopwords and raw in STOPWORDS):
                    continue
                out.append((light_stem(raw) if self.stem else raw, pos))
            elif "(" in raw:
                out.append((raw, pos))
                if not query:
                    base = raw
                    while "(" in base:
                        base = base[:base.rindex("(")]
                        out.append((base, pos))
            else:
                out.append((raw, pos))
            pos += 1
        return out

    def terms(self, text: str, query: bool = False):
        return [t for t, _ in self.tokens(text, query)]
