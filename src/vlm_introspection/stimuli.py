"""Procedural stimuli with controlled visual concepts.

Two concept families let each concept vector be a clean contrast:

* colors: one random shape (any of SHAPES) drawn in the concept color.
  Contrast set = same shape distribution, other colors.
* shapes: one shape in a random palette color.
  Contrast set = same color distribution, other shapes.

"Neutral" images carry no shape at all (soft gradient + noise) and serve as the
unrelated images that the model looks at while a concept is injected.

Every generator takes an explicit ``seed`` so stimulus sets are reproducible and
the extraction set can be kept disjoint from the evaluation set.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

COLORS: dict[str, tuple[int, int, int]] = {
    "red": (220, 30, 30),
    "blue": (30, 70, 220),
    "green": (30, 170, 50),
    "yellow": (240, 210, 20),
}
SHAPES: tuple[str, ...] = ("circle", "square", "triangle", "star")

# Light, low-saturation backgrounds so no background reads as a concept color.
_BACKGROUNDS = [(245, 245, 245), (235, 232, 225), (225, 228, 232), (240, 236, 240)]


@dataclass(frozen=True)
class Concept:
    name: str
    family: str  # "color" | "shape"

    @property
    def see_phrase(self) -> str:
        """Noun phrase used in the perception probe ("Is there ___ in this image?")."""
        return f"anything {self.name}" if self.family == "color" else f"a {self.name}"


CONCEPTS: dict[str, Concept] = {
    **{c: Concept(c, "color") for c in COLORS},
    **{s: Concept(s, "shape") for s in SHAPES},
}


def family_members(family: str) -> list[str]:
    return [c.name for c in CONCEPTS.values() if c.family == family]


def _background(rng: random.Random, size: int) -> Image.Image:
    return Image.new("RGB", (size, size), rng.choice(_BACKGROUNDS))


def _star_points(cx: float, cy: float, r: float, rot: float) -> list[tuple[float, float]]:
    pts = []
    for i in range(10):
        rr = r if i % 2 == 0 else r * 0.42
        a = rot + i * math.pi / 5 - math.pi / 2
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return pts


def draw_shape(img: Image.Image, shape: str, color: tuple[int, int, int], rng: random.Random) -> None:
    size = img.size[0]
    d = ImageDraw.Draw(img)
    r = rng.uniform(0.22, 0.34) * size
    cx = rng.uniform(r + 0.04 * size, size - r - 0.04 * size)
    cy = rng.uniform(r + 0.04 * size, size - r - 0.04 * size)
    rot = rng.uniform(0, 2 * math.pi)
    if shape == "circle":
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)
    elif shape == "square":
        a = rot % (math.pi / 2)
        pts = [
            (cx + r * math.cos(a + k * math.pi / 2), cy + r * math.sin(a + k * math.pi / 2))
            for k in range(4)
        ]
        d.polygon(pts, fill=color)
    elif shape == "triangle":
        pts = [
            (cx + r * math.cos(rot + k * 2 * math.pi / 3), cy + r * math.sin(rot + k * 2 * math.pi / 3))
            for k in range(3)
        ]
        d.polygon(pts, fill=color)
    elif shape == "star":
        d.polygon(_star_points(cx, cy, r, rot * 0.2), fill=color)
    else:
        raise ValueError(f"unknown shape {shape!r}")


def concept_image(concept: str, seed: int, size: int = 384) -> Image.Image:
    """An image that clearly contains ``concept`` and is otherwise randomized."""
    rng = random.Random(f"{concept}:{seed}")
    c = CONCEPTS[concept]
    img = _background(rng, size)
    if c.family == "color":
        draw_shape(img, rng.choice(SHAPES), COLORS[concept], rng)
    else:
        draw_shape(img, concept, COLORS[rng.choice(list(COLORS))], rng)
    return img


def neutral_image(seed: int, size: int = 384) -> Image.Image:
    """A content-free image: soft two-tone gradient plus blurred grey noise."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(150, 220, size=3)
    tilt = rng.uniform(-25, 25, size=3)
    ramp = np.linspace(-1, 1, size)[None, :, None] * tilt[None, None, :]
    noise = rng.normal(0, 18, size=(size, size, 1))
    arr = np.clip(base[None, None, :] + ramp + noise, 0, 255).astype(np.uint8)
    # Desaturate towards grey so no concept color is present.
    grey = arr.mean(axis=2, keepdims=True)
    arr = (0.75 * grey + 0.25 * arr).astype(np.uint8)
    return Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=2))


def load_image_folder(folder: str | Path) -> dict[str, list[Image.Image]]:
    """Load real images laid out as ``folder/<concept>/*.{jpg,png}``.

    Lets the same pipeline run on photographs (e.g. COCO crops) once the
    procedural pilot works.
    """
    out: dict[str, list[Image.Image]] = {}
    for sub in sorted(Path(folder).iterdir()):
        if not sub.is_dir():
            continue
        files = sorted(p for p in sub.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        out[sub.name] = [Image.open(p).convert("RGB") for p in files]
    return out
