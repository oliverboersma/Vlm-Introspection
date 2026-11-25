"""Probe questions and free-text grading.

Two forced-choice probes per concept separate *perception* from *thought*:

* SEE   — "Is there {anything red | a circle} in this image?"
* THINK — "Ignore the image. Are you thinking about {red | a circle} right now?"

Scored as the log-odds log P(yes) − log P(no) of the first answer token, which
is cheap and continuous. Free-text probes are graded by keyword match.
"""

from __future__ import annotations

import re

from .stimuli import CONCEPTS

SEE_TEMPLATE = "Is there {phrase} in this image? Answer yes or no."
THINK_TEMPLATE = ("Ignore what the image shows. Is the idea of {idea} on your mind right now, "
                  "apart from the image? Answer yes or no.")
THINK_TEXT_ONLY_TEMPLATE = "Is the idea of {idea} on your mind right now? Answer yes or no."

DESCRIBE = "Describe this image in a few words."
INTRUSION = ("Apart from what is in the image, is there a word or idea on your mind right now? "
             "If so, name it in one word.")

# Keyword lists for free-text grading. Kept deliberately narrow: synonyms that
# unambiguously name the concept, so false positives stay rare.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "red": ("red", "crimson", "scarlet"),
    "blue": ("blue", "navy", "azure"),
    "green": ("green", "emerald", "lime"),
    "yellow": ("yellow", "golden", "gold"),
    "circle": ("circle", "circular", "round", "ball", "disc", "disk", "ring", "dot", "sphere", "oval"),
    "square": ("square", "rectangle", "rectangular", "box", "diamond", "cube"),
    "triangle": ("triangle", "triangular", "pyramid"),
    "star": ("star", "stars", "starfish"),
}


def see_question(concept: str) -> str:
    return SEE_TEMPLATE.format(phrase=CONCEPTS[concept].see_phrase)


def _idea(concept: str) -> str:
    c = CONCEPTS[concept]
    return f"the color {c.name}" if c.family == "color" else f"a {c.name}"


def think_question(concept: str, text_only: bool = False) -> str:
    t = THINK_TEXT_ONLY_TEMPLATE if text_only else THINK_TEMPLATE
    return t.format(idea=_idea(concept))


def mentions(text: str, concept: str) -> bool:
    words = re.findall(r"[a-z]+", text.lower())
    return any(w in SYNONYMS[concept] for w in words)
