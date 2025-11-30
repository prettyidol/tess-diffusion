"""Utilities to parse one-line TESS sequences into KG triples for ranking eval.

Each line in oneline files is a sequence of quads (h, r, t, time) joined with
" ||| ". For KG ranking, we ignore time and treat each quad as a triple.
"""
from __future__ import annotations

from typing import Dict, List, Tuple


Triple = Tuple[str, str, str]
Quad = Tuple[str, str, str, str]


def parse_oneline_triples(path: str) -> List[Triple]:
    triples: List[Triple] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            segments = [seg.strip() for seg in line.split("|||")]
            for seg in segments:
                parts = seg.split("\t")
                if len(parts) < 3:
                    continue
                h, r, t = parts[0], parts[1], parts[2]
                triples.append((h, r, t))
    return triples


def parse_oneline_quads(path: str) -> List[Quad]:
    """Parse oneline file into (h, r, t, time) quads.

    Notes:
      - If a line contains a single quad (no '|||'), it's handled as well.
      - If time is missing, use empty string "".
    """
    quads: List[Quad] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            segments = [seg.strip() for seg in line.split("|||")]
            for seg in segments:
                parts = seg.split("\t")
                if len(parts) < 3:
                    continue
                h, r, t = parts[0], parts[1], parts[2]
                time = parts[3] if len(parts) > 3 else ""
                quads.append((h, r, t, time))
    return quads


def build_id_mappings(triples: List[Triple]):
    """Build entity/relation id maps from observed triples."""
    ent2id: Dict[str, int] = {}
    rel2id: Dict[str, int] = {}
    for h, r, t in triples:
        if h not in ent2id:
            ent2id[h] = len(ent2id)
        if t not in ent2id:
            ent2id[t] = len(ent2id)
        if r not in rel2id:
            rel2id[r] = len(rel2id)
    return ent2id, rel2id
