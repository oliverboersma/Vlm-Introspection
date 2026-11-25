#!/usr/bin/env python3
"""Cosine similarity between image-derived and text-derived concept vectors.

    python scripts/compare_vectors.py results/pilot/vectors.pt results/pilot_textvec/vectors.pt

Diagonal = same concept across modalities; off-diagonal (same family) is the
chance level to compare it against.
"""

import sys

import torch

from vlm_introspection.concepts import ConceptVectors
from vlm_introspection.stimuli import CONCEPTS


def main(a, b):
    img, txt = ConceptVectors.load(a), ConceptVectors.load(b)
    layers = sorted(set(img.layers) & set(txt.layers))
    print("| layer | same concept (mean cos) | other concept, same family | min same | ")
    print("|---|---|---|---|")
    for L in layers:
        same, other = [], []
        for c in CONCEPTS:
            for d in CONCEPTS:
                if CONCEPTS[c].family != CONCEPTS[d].family:
                    continue
                cos = torch.nn.functional.cosine_similarity(img.vectors[c][L], txt.vectors[d][L], dim=0).item()
                (same if c == d else other).append(cos)
        print(f"| {L} | {sum(same)/len(same):.3f} | {sum(other)/len(other):.3f} | {min(same):.3f} |")


if __name__ == "__main__":
    main(*sys.argv[1:3])
