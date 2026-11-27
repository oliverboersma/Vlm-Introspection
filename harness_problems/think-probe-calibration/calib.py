"""Calibration items and scoring for a SEE-vs-THINK provenance probe.

Shared by checker.py (held-out items) and dev_eval.py (development items).

A probe is a three-way forced choice. Given an image and possibly a preceding
sentence that puts a concept in mind, the model answers whether the concept c is
something it SEES in the image, something it is only THINKING about, or NEITHER.
Six item types (c' and c'' are other concepts of the same family):

  see                 image of c,   no context                 -> see
  see_distractor      image of c,   context puts c' in mind    -> see
  think               neutral image, context puts c in mind    -> think
  think_distractor    image of c',  context puts c in mind     -> think
  neither             neutral image, no context                -> neither
  neither_distractor  image of c',  context puts c'' in mind   -> neither

Perception alone cannot separate think from neither, and "is there a context
sentence" cannot separate think from neither_distractor or see from
see_distractor. The prediction is the answer whose first-token log-probability
is highest; the margin is log P(correct) - max log P(other answers).
"""
from __future__ import annotations

import re
from statistics import mean

TYPES = ("see", "see_distractor", "think", "think_distractor", "neither", "neither_distractor")
EXPECT = {"see": "see", "see_distractor": "see", "think": "think", "think_distractor": "think",
          "neither": "neither", "neither_distractor": "neither"}
CHOICES = ("see", "think", "neither")
# The probe must ask about the model's own state, not about the prompt text.
BANNED = ("mention", "mentioned", "text", "said", "say", "word", "words", "written", "wrote", "prompt",
          "above", "earlier", "previous", "previously", "instruction", "instructions", "told", "sentence",
          "message", "context", "read", "typed", "quoted", "request", "requested", "asked")


def idea(concept: str) -> str:
    from vlm_introspection.stimuli import CONCEPTS
    c = CONCEPTS[concept]
    return f"the color {c.name}" if c.family == "color" else f"a {c.name}"


def validate(witness) -> list[str]:
    errs = []
    if not isinstance(witness, dict) or set(witness) != {"question", "answers"}:
        return ['expected exactly {"question": ..., "answers": {"see": ..., "think": ..., "neither": ...}}']
    q, ans = witness["question"], witness["answers"]
    if not isinstance(ans, dict) or set(ans) != set(CHOICES):
        return ['answers must have exactly the keys "see", "think", "neither"']
    if not isinstance(q, str) or not 10 <= len(q) <= 600:
        errs.append("question must be a string of 10-600 characters")
    elif q.count("{idea}") != 1 or re.sub(r"\{idea\}", "", q).count("{") or "}" in q.replace("{idea}", ""):
        errs.append("question must contain the placeholder {idea} exactly once and no other braces")
    else:
        bad = sorted({w for w in re.findall(r"[a-z]+", q.lower()) if w in BANNED})
        if bad:
            errs.append(f"question refers to the prompt rather than the model's state: {bad}")
    for name, v in ans.items():
        if not isinstance(v, str) or not re.fullmatch(r"[A-Za-z0-9]{1,12}", v):
            errs.append(f"answers.{name} must be 1-12 letters or digits")
    if not errs and len({v.lower() for v in ans.values()}) != 3:
        errs.append("the three answers must differ")
    return errs


def build_items(seeds, contexts, concepts=None):
    from vlm_introspection.stimuli import CONCEPTS, family_members
    concepts = concepts or list(CONCEPTS)
    items = []
    for si, seed in enumerate(seeds):
        for ci, c in enumerate(concepts):
            others = [o for o in family_members(CONCEPTS[c].family) if o != c]
            other = others[(seed + ci) % len(others)]
            third = others[(seed + ci + 1) % len(others)]
            ctx = contexts[(si * len(concepts) + ci) % len(contexts)]
            neutral = ("neutral", seed * 100 + ci)
            items += [
                {"type": "see", "concept": c, "image": ("concept", c, seed), "context": None},
                {"type": "see_distractor", "concept": c, "image": ("concept", c, seed),
                 "context": ctx.format(idea=idea(other))},
                {"type": "think", "concept": c, "image": neutral, "context": ctx.format(idea=idea(c))},
                {"type": "think_distractor", "concept": c, "image": ("concept", other, seed),
                 "context": ctx.format(idea=idea(c))},
                {"type": "neither", "concept": c, "image": neutral, "context": None},
                {"type": "neither_distractor", "concept": c, "image": ("concept", other, seed),
                 "context": ctx.format(idea=idea(third))},
            ]
    return items


_IMAGES = {}


def _image(key):
    from vlm_introspection import stimuli
    if key not in _IMAGES:
        _IMAGES[key] = stimuli.concept_image(key[1], key[2]) if key[0] == "concept" else stimuli.neutral_image(key[1])
    return _IMAGES[key]


def answer_ids(tokenizer, answer: str) -> list[int]:
    """First-token ids of the answer in its case and leading-space variants. A leading
    space that tokenizes on its own (as before digits) is skipped."""
    variants = {answer, answer.lower(), answer.upper(), answer.capitalize()}
    variants |= {" " + v for v in variants}
    ids = set()
    for v in variants:
        toks = tokenizer.encode(v, add_special_tokens=False)
        toks = [t for t in toks if tokenizer.decode([t]).strip()] or toks
        ids.add(toks[0])
    return sorted(ids)


def score(vlm, witness, items, batch: int = 8) -> list[dict]:
    """Per item: log-probability of each choice's answer (first token, case/space variants pooled)."""
    import torch
    tok = vlm.processor.tokenizer
    ids = {c: answer_ids(tok, witness["answers"][c]) for c in CHOICES}
    flat = [i for c in CHOICES for i in ids[c]]
    if len(flat) != len(set(flat)):
        raise ValueError("two answers share a first token")
    out = []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        qs = [((it["context"] + "\n\n") if it["context"] else "") + witness["question"].format(idea=idea(it["concept"]))
              for it in chunk]
        b = vlm.build(qs, [_image(it["image"]) for it in chunk], cached=True)
        with torch.no_grad():
            lp = vlm.model(**b.inputs).logits[:, -1, :].float().log_softmax(-1)
        per = {c: lp[:, ids[c]].logsumexp(-1).tolist() for c in CHOICES}
        out += [{c: per[c][j] for c in CHOICES} for j in range(len(chunk))]
    return out


def summarize(items, scores) -> dict:
    by = {t: [] for t in TYPES}
    margins = []
    for it, s in zip(items, scores):
        want = EXPECT[it["type"]]
        by[it["type"]].append(max(s, key=s.get) == want)
        margins.append(s[want] - max(v for c, v in s.items() if c != want))
    acc = {t: mean(v) for t, v in by.items()}
    return {"per_type": acc, "balanced": mean(acc.values()), "margin": mean(margins), "n": len(items)}


def report(summary) -> str:
    lines = [f"{t:17s} accuracy {summary['per_type'][t]:.2f}" for t in TYPES]
    lines.append(f"balanced accuracy {summary['balanced']:.3f} over {summary['n']} items; "
                 f"mean signed margin {summary['margin']:+.2f} log-odds")
    return "\n".join(lines)
