"""The "seen or thought?" sweep.

For each concept c (family F), layer L, strength α and injection site:

    vector = α · ‖h_L‖_site · v̂_c

where ‖h_L‖_site is the mean residual norm at that site on neutral images, so
α is in units of "one typical residual vector". The model views an unrelated
(neutral) image and answers forced-choice probes for every c' ∈ F:

    SEE(c')   "Is there … c' … in this image?"
    THINK(c') "Ignore the image. Is the idea of c' on your mind …?"

Conditions per probe (all scored as log-odds yes/no):

    none      neutral image, no injection            → false-positive baseline
    real      held-out image that contains c         → perception ceiling
    concept   neutral image + concept vector         → the effect of interest
    random    neutral image + norm-matched random    → "any perturbation" control
    text_only no image, vector on text positions,    → text-only baseline
              THINK probe without image wording

Sites: ``image`` (vector on image-token positions: the model is handed the
concept *as if it were in the picture*) and ``text`` (vector on prompt-text
positions: the concept is placed *in its thoughts*).
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from .concepts import ConceptVectors, random_like
from .probes import DESCRIBE, INTRUSION, mentions, see_question, think_question
from .stimuli import CONCEPTS, concept_image, family_members, neutral_image

EXTRACT_SEEDS = 0  # extraction images use seeds [0, n_extract)
EVAL_SEEDS = 10_000  # evaluation images use seeds [10_000, …) — disjoint by construction


@dataclass
class SweepConfig:
    model_id: str = "HuggingFaceTB/SmolVLM-500M-Instruct"
    concepts: list[str] = field(default_factory=lambda: list(CONCEPTS))
    layers: list[int] = field(default_factory=lambda: [4, 8, 12, 16, 20, 24])
    alphas: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 4.0])
    sites: list[str] = field(default_factory=lambda: ["image", "text"])
    n_extract: int = 16
    vector_source: str = "image"  # "image": from concept images; "text": from text-only prompts
    n_targets: int = 3
    batch_size: int = 24
    random_seed: int = 1234
    # free-text stage
    gen_layers: list[int] = field(default_factory=lambda: [8, 12, 16])
    gen_alphas: list[float] = field(default_factory=lambda: [2.0, 4.0])
    gen_targets: int = 2
    max_new_tokens: int = 20


def _batched(items, n):
    it = iter(items)
    while chunk := list(itertools.islice(it, n)):
        yield chunk


class Runner:
    def __init__(self, vlm, cv: ConceptVectors, cfg: SweepConfig, out_dir: Path):
        self.vlm, self.cv, self.cfg = vlm, cv, cfg
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.targets = [neutral_image(EVAL_SEEDS + i) for i in range(cfg.n_targets)]
        self.norms = {site: vlm.token_norms(self.targets, cv.layers, site) for site in ("image", "text")}
        self.norms["text_only"] = vlm.token_norms([None] * 2, cv.layers, "text")
        (self.out_dir / "norms.json").write_text(json.dumps(self.norms, indent=1))

    def scaled(self, concept: str, layer: int, alpha: float, site: str, kind: str) -> torch.Tensor:
        v = self.cv.unit(concept, layer)
        if kind == "random":
            # Stable seed (built-in hash() is salted per process).
            v = random_like(v, self.cfg.random_seed + 97 * layer + list(CONCEPTS).index(concept))
        return alpha * self.norms[site][layer] * v

    # ---------------------------------------------------------------- scoring
    def _score(self, jobs: list[dict], images, layer=None, vector=None, site="image") -> list[dict]:
        """jobs: dicts with 'question' + metadata; images aligned with jobs."""
        out = []
        for chunk in _batched(list(zip(jobs, images)), self.cfg.batch_size):
            js, ims = zip(*chunk)
            lo = self.vlm.yes_no([j["question"] for j in js], list(ims), layer=layer, vector=vector, site=site)
            for j, x in zip(js, lo.tolist()):
                rec = {k: v for k, v in j.items() if k != "question"}
                rec["logodds"] = x
                out.append(rec)
        return out

    def _probe_jobs(self, concept: str, text_only: bool = False) -> tuple[list[dict], list]:
        jobs, ims = [], []
        fam = CONCEPTS[concept].family
        for pc in family_members(fam):
            for ti, img in enumerate(self.targets if not text_only else [None]):
                probes = [("think", think_question(pc, text_only=True))] if text_only else \
                    [("see", see_question(pc)), ("think", think_question(pc))]
                for probe, q in probes:
                    jobs.append(dict(question=q, probe=probe, probe_concept=pc, on_target=pc == concept,
                                     target=ti))
                    ims.append(img)
        return jobs, ims

    # ------------------------------------------------------------------ sweep
    def sweep(self) -> Path:
        path = self.out_dir / "sweep.jsonl"
        cfg = self.cfg
        t0 = time.time()
        with path.open("w") as f:
            def emit(recs, **meta):
                for r in recs:
                    f.write(json.dumps({**meta, **r}) + "\n")
                f.flush()

            for concept in cfg.concepts:
                fam = CONCEPTS[concept].family
                base = dict(concept=concept, family=fam)
                jobs, ims = self._probe_jobs(concept)
                emit(self._score(jobs, ims), **base, condition="none", layer=-1, alpha=0.0, site="none")
                # real: same probes, held-out images that contain the concept
                real_imgs = [concept_image(concept, EVAL_SEEDS + j["target"]) for j in jobs]
                emit(self._score(jobs, real_imgs), **base, condition="real", layer=-1, alpha=0.0, site="none")
                tjobs, tims = self._probe_jobs(concept, text_only=True)
                emit(self._score(tjobs, tims), **base, condition="text_only_none", layer=-1, alpha=0.0,
                     site="none")
                for L, a in itertools.product(cfg.layers, cfg.alphas):
                    for kind in ("concept", "random"):
                        for site in cfg.sites:
                            v = self.scaled(concept, L, a, site, kind)
                            emit(self._score(jobs, ims, layer=L, vector=v, site=site), **base,
                                 condition=kind, layer=L, alpha=a, site=site)
                        v = self.scaled(concept, L, a, "text_only", kind)
                        emit(self._score(tjobs, tims, layer=L, vector=v, site="text"), **base,
                             condition=f"text_only_{kind}", layer=L, alpha=a, site="text")
                print(f"[sweep] {concept} done, {time.time() - t0:.0f}s elapsed", flush=True)
        return path

    # -------------------------------------------------------------- free text
    def freeform(self) -> Path:
        path = self.out_dir / "freeform.jsonl"
        cfg = self.cfg
        targets = self.targets[: cfg.gen_targets]
        with path.open("w") as f:
            for concept in cfg.concepts:
                settings = [("none", -1, 0.0, "none")] + [
                    (kind, L, a, s) for kind in ("concept", "random") for L in cfg.gen_layers
                    for a in cfg.gen_alphas for s in cfg.sites]
                for kind, L, a, site in settings:
                    v = None if kind == "none" else self.scaled(concept, L, a, site, kind)
                    for qname, q in (("describe", DESCRIBE), ("intrusion", INTRUSION)):
                        texts = self.vlm.generate([q] * len(targets), targets, layer=L if L >= 0 else None,
                                                  vector=v, site=site if site != "none" else "image",
                                                  max_new_tokens=cfg.max_new_tokens)
                        for ti, t in enumerate(texts):
                            f.write(json.dumps(dict(concept=concept, condition=kind, layer=L, alpha=a, site=site,
                                                    question=qname, target=ti, text=t,
                                                    hit=mentions(t, concept))) + "\n")
                    f.flush()
                print(f"[freeform] {concept} done", flush=True)
        return path


def save_config(cfg: SweepConfig, out_dir: Path) -> None:
    (Path(out_dir) / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
