"""Boolean model: query parser + set-algebra evaluator.

Grammar (NOT binds tightest, then AND, then OR; bare neighbours mean AND):
    expr  := and ('OR' and)*
    and   := not (['AND'] not)*
    not   := 'NOT' not | atom
    atom  := '(' expr ')' | "quoted phrase" | [field:]word
AND = intersection, OR = union, NOT = universe minus set.
"""
import re

OPS = ("AND", "OR", "NOT")
CITE = r"\d+[A-Za-z]?(?:\([A-Za-z0-9]+\))+"          # 153(1)(b) is a citation, not parentheses
RE_OPS = re.compile(r'\b(AND|OR|NOT)\b|[()"]|\b[A-Za-z]+:\S')
RE_QTOK = re.compile(r'"[^"]*"|' + CITE + r'|\(|\)|[^\s()"]+')


def has_ops(q):
    return bool(RE_OPS.search(re.sub(CITE, " ", q)))


class Parser:
    def __init__(self, query, analyzer, fields=()):
        self.t = RE_QTOK.findall(query)
        self.i, self.an, self.fields = 0, analyzer, {f.lower() for f in fields}

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def eat(self):
        self.i += 1
        return self.t[self.i - 1]

    def parse(self):
        node = self.or_()
        while self.peek() is not None:          # stray ")" etc.
            self.eat()
            node = self.or_() if self.peek() is not None else node
        return node

    def or_(self):
        items = [self.and_()]
        while self.peek() == "OR":
            self.eat()
            items.append(self.and_())
        return ("or", items) if len(items) > 1 else items[0]

    def and_(self):
        items = [self.not_()]
        while self.peek() not in (None, "OR", ")"):
            if self.peek() == "AND":
                self.eat()
            items.append(self.not_())
        return ("and", items) if len(items) > 1 else items[0]

    def not_(self):
        if self.peek() == "NOT":
            self.eat()
            return ("not", self.not_())
        return self.atom()

    def atom(self):
        tok = self.peek()
        if tok is None or tok == ")":
            return ("skip",)
        if tok == "(":
            self.eat()
            n = self.or_()
            if self.peek() == ")":
                self.eat()
            return n
        self.eat()
        if tok in ("AND", "OR"):                # stray operator -> ignore it
            return self.atom()
        if tok.startswith('"'):
            toks = self.an.terms(tok.strip('"'), query=True)
            return ("phrase", None, toks) if len(toks) > 1 else (("term", None, toks[0]) if toks else ("skip",))
        field, word = None, tok
        if ":" in tok and tok.split(":", 1)[0].lower() in self.fields:
            field, word = tok.split(":", 1)[0].lower(), tok.split(":", 1)[1]
        toks = self.an.terms(word, query=True)
        if not toks:
            return ("skip",)                    # e.g. a stop-word
        return ("term", field, toks[0]) if len(toks) == 1 else ("phrase", field, toks)


def evaluate(node, index, fields):
    """Return the set of unit ids satisfying the expression (None = nothing to evaluate)."""
    k = node[0]
    if k == "skip":
        return None
    if k == "term":
        return index.docs_with(node[2], [node[1]] if node[1] else fields)
    if k == "phrase":
        return index.phrase_docs(node[2], [node[1]] if node[1] else fields)
    if k == "not":
        s = evaluate(node[1], index, fields)
        return None if s is None else set(index.uids) - s
    parts = [p for p in (evaluate(c, index, fields) for c in node[1]) if p is not None]
    if not parts:
        return None
    return set.intersection(*parts) if k == "and" else set.union(*parts)


def positive_terms(node, neg=False):
    """Terms that should drive ranking (everything not under a NOT)."""
    k = node[0]
    if k == "skip":
        return []
    if k == "term":
        return [] if neg else [node[2]]
    if k == "phrase":
        return [] if neg else list(node[2])
    if k == "not":
        return positive_terms(node[1], not neg)
    return [t for c in node[1] for t in positive_terms(c, neg)]


def pretty(node):
    k = node[0]
    if k == "skip":
        return "∅"
    if k == "term":
        return (node[1] + ":" if node[1] else "") + node[2]
    if k == "phrase":
        return '"' + " ".join(node[2]) + '"'
    if k == "not":
        return "NOT " + pretty(node[1])
    return "(" + f" {k.upper()} ".join(pretty(c) for c in node[1]) + ")"
