"""End-to-end checks on a real VLM. Run with: pytest -m slow"""

import pytest
import torch

from vlm_introspection.stimuli import concept_image, neutral_image
from vlm_introspection.probes import see_question

pytestmark = pytest.mark.slow
MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"


@pytest.fixture(scope="module")
def vlm():
    from vlm_introspection.model import VLM

    return VLM(MODEL)


def test_feature_cache_is_exact(vlm):
    imgs = [concept_image("red", 1), neutral_image(1)]
    qs = [see_question("red")] * 2
    with torch.no_grad():
        a = vlm.model(**vlm.build(qs, imgs).inputs).logits[:, -1]
        b = vlm.model(**vlm.build(qs, imgs, cached=True).inputs).logits[:, -1]
    assert torch.allclose(a, b, atol=1e-4)


def test_see_probe_discriminates(vlm):
    lo = vlm.yes_no([see_question("red")] * 2, [concept_image("red", 2), neutral_image(2)])
    assert lo[0] > 2 and lo[1] < 0


def test_text_mask_excludes_sink(vlm):
    b = vlm.build(["Describe this image."], [neutral_image(0)])
    first = b.inputs["attention_mask"][0].nonzero()[0].item()
    assert not b.text_mask[0, first]
    assert b.text_mask[0].sum() > 5 and b.image_mask[0].sum() == 64
