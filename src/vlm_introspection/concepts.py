"""Concept vectors: mean-difference directions in the residual stream.

For concept c in family F at layer L:

    v_c = mean_{x in images(c)} a_L(x) − mean_{x in images(F \\ {c})} a_L(x)

where a_L(x) is the residual activation at layer L averaged over the image-token
positions. Contrasting against the *other members of the same family* removes
what all color (or all shape) images share, leaving what is specific to c.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .stimuli import CONCEPTS, concept_image, family_members


@dataclass
class ConceptVectors:
    layers: list[int]
    vectors: dict[str, dict[int, torch.Tensor]] = field(default_factory=dict)  # concept -> L -> (d,)

    def unit(self, concept: str, layer: int) -> torch.Tensor:
        v = self.vectors[concept][layer]
        return v / v.norm()

    def save(self, path) -> None:
        torch.save({"layers": self.layers, "vectors": self.vectors}, path)

    @classmethod
    def load(cls, path) -> "ConceptVectors":
        d = torch.load(path)
        return cls(d["layers"], d["vectors"])


def mean_difference(acts: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """``acts[c]`` is (N_c, d); returns c -> mean(acts[c]) − mean(all other concepts' acts)."""
    means = {c: a.mean(0) for c, a in acts.items()}
    out = {}
    for c in acts:
        others = torch.cat([a for k, a in acts.items() if k != c])
        out[c] = means[c] - others.mean(0)
    return out


def extract(vlm, layer_ids: list[int], n_images: int = 16, seed_offset: int = 0,
            concepts: list[str] | None = None) -> ConceptVectors:
    """Extract vectors for every concept, contrasting within each family.

    Seeds ``seed_offset .. seed_offset + n_images − 1`` are used; keep evaluation
    images on a disjoint seed range.
    """
    concepts = concepts or list(CONCEPTS)
    cv = ConceptVectors(layer_ids)
    for family in sorted({CONCEPTS[c].family for c in concepts}):
        members = family_members(family)
        acts_by_layer: dict[int, dict[str, torch.Tensor]] = {L: {} for L in layer_ids}
        for c in members:
            imgs = [concept_image(c, seed_offset + i) for i in range(n_images)]
            acts = vlm.mean_acts(imgs, layer_ids)
            for L in layer_ids:
                acts_by_layer[L][c] = acts[L]
        for L in layer_ids:
            for c, v in mean_difference(acts_by_layer[L]).items():
                if c in concepts:
                    cv.vectors.setdefault(c, {})[L] = v
    return cv


def random_like(v: torch.Tensor, seed: int) -> torch.Tensor:
    """Norm-matched random direction (control for 'any perturbation does this')."""
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(v.shape, generator=g)
    return r / r.norm() * v.norm()
