"""Structured text retrieval: regions and nodes.

overlapped      : regions may nest (a section CONTAINS its subsections). A region qualifies only if ALL query
                  terms occur INSIDE it ("same room"); we keep the most specific (smallest) such regions.
non_overlapped  : text is cut into disjoint regions (each node's OWN text). Each term has its own list; a region
                  qualifies if it is in ANY list ("same building"). Coverage up, precision down - ranking repairs it.
proximal        : a node's score = its own score + discounted scores of NEARBY nodes (parent/child, adjacent siblings,
                  cross-references like "see section 165"), within `hops` links, each hop multiplying by `decay`.
"""
from collections import defaultdict
from .documents import RE_REF


def prune_minimal(cands, nodes):
    keep = set(cands)
    for nid in cands:
        p = nodes[nid].parent
        while p:
            keep.discard(p)
            p = nodes[p].parent
    return keep


def overlapped_candidates(idx_sub, nodes, terms, fields, minimal=True):
    sets = [idx_sub.docs_with(t, fields) for t in dict.fromkeys(terms)]
    cands = set.intersection(*sets) if sets else set()
    return prune_minimal(cands, nodes) if minimal else cands


def non_overlapped_lists(idx_own, terms, fields):
    lists = {t: idx_own.docs_with(t, fields) for t in dict.fromkeys(terms)}
    return (set().union(*lists.values()) if lists else set()), lists


def build_adjacency(nodes):
    adj = defaultdict(set)

    def link(a, b):
        if a != b:
            adj[a].add(b)
            adj[b].add(a)
    sec = {}
    for n in nodes.values():
        if n.parent:
            link(n.id, n.parent)
        if n.type == "section" and n.citation:
            sec.setdefault(n.citation.lower(), n.id)
    for n in nodes.values():
        for a, b in zip(n.children, n.children[1:]):
            link(a, b)
    for n in nodes.values():
        for ref in RE_REF.findall(n.text):
            if ref.lower() in sec:
                link(n.id, sec[ref.lower()])
    return adj


def proximal(own, adj, hops=2, decay=0.5):
    extra = defaultdict(list)
    for m, s in own.items():
        if s <= 0:
            continue
        seen, frontier = {m}, [m]
        for d in range(1, hops + 1):
            nxt = []
            for x in frontier:
                for y in adj.get(x, ()):
                    if y not in seen:
                        seen.add(y)
                        nxt.append(y)
                        extra[y].append((m, d, s * decay ** d))
            frontier = nxt
    ids = set(own) | set(extra)
    return {n: own.get(n, 0.0) + sum(c for _, _, c in extra.get(n, [])) for n in ids}, extra
