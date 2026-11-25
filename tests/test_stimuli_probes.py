import numpy as np
import torch

from vlm_introspection.concepts import mean_difference, random_like
from vlm_introspection.probes import mentions, see_question, think_question
from vlm_introspection.stimuli import COLORS, CONCEPTS, concept_image, family_members, neutral_image


def test_images_deterministic_and_distinct():
    a = np.asarray(concept_image("red", 3))
    assert np.array_equal(a, np.asarray(concept_image("red", 3)))
    assert not np.array_equal(a, np.asarray(concept_image("red", 4)))
    assert concept_image("star", 0).size == (384, 384)


def test_color_images_contain_their_color():
    for c, rgb in COLORS.items():
        arr = np.asarray(concept_image(c, 1)).reshape(-1, 3).astype(int)
        close = (np.abs(arr - np.array(rgb)).sum(1) < 30).mean()
        assert close > 0.05, c


def test_neutral_images_have_no_saturated_color():
    arr = np.asarray(neutral_image(0)).astype(int)
    sat = arr.max(-1) - arr.min(-1)
    assert sat.max() < 60


def test_families():
    assert family_members("color") == list(COLORS)
    assert len(family_members("shape")) == 4
    assert set(CONCEPTS) == set(family_members("color")) | set(family_members("shape"))


def test_questions():
    assert see_question("red") == "Is there anything red in this image? Answer yes or no."
    assert "a circle" in see_question("circle")
    assert "the color blue" in think_question("blue")
    assert "image" not in think_question("star", text_only=True)


def test_mentions():
    assert mentions("A bright red circle.", "red")
    assert mentions("A round blob", "circle")
    assert not mentions("A reddish blob", "red")  # whole-word only
    assert not mentions("a grey background", "blue")


def test_mean_difference():
    acts = {"a": torch.tensor([[1.0, 0.0], [3.0, 0.0]]), "b": torch.tensor([[0.0, 2.0]]), "c": torch.tensor([[0.0, 4.0]])}
    d = mean_difference(acts)
    assert torch.allclose(d["a"], torch.tensor([2.0, -3.0]))
    assert torch.allclose(d["b"], torch.tensor([-4.0 / 3, 2 - 4.0 / 3]))


def test_random_like_norm():
    v = torch.randn(50) * 3
    r = random_like(v, 0)
    assert torch.isclose(r.norm(), v.norm())
    assert torch.equal(r, random_like(v, 0))
