import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from vlm_introspection.hooks import Injection, capture, inject


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64)
    m = LlamaForCausalLM(cfg).eval()
    return m, m.model.layers


def test_capture_shapes(tiny):
    m, layers = tiny
    ids = torch.randint(0, 64, (2, 7))
    with capture(layers, [0, 2]) as st:
        m(input_ids=ids)
    assert set(st) == {0, 2}
    assert st[0].shape == (2, 7, 32)


def test_injection_adds_exactly_at_masked_positions(tiny):
    m, layers = tiny
    ids = torch.randint(0, 64, (2, 7))
    v = torch.randn(32)
    mask = torch.zeros(2, 7, dtype=torch.bool)
    mask[0, 3] = True
    mask[1, 5:] = True
    with capture(layers, [1]) as base:
        m(input_ids=ids)
    # Capture hook registered *after* the injection hook sees the injected output.
    with inject(layers, 1, Injection(v, mask)), capture(layers, [1]) as st:
        m(input_ids=ids)
    diff = st[1] - base[1]
    assert torch.allclose(diff[mask], v.expand(int(mask.sum()), -1), atol=1e-5)
    assert diff[~mask].abs().max() < 1e-6


def test_injection_respects_causality(tiny):
    m, layers = tiny
    ids = torch.randint(0, 64, (1, 8))
    mask = torch.zeros(1, 8, dtype=torch.bool)
    mask[0, 4] = True
    with capture(layers, [3]) as base:
        m(input_ids=ids)
    with inject(layers, 1, Injection(torch.randn(32) * 5, mask)), capture(layers, [3]) as st:
        m(input_ids=ids)
    d = (st[3] - base[3]).abs().amax(-1)[0]
    assert d[:4].max() < 1e-5  # earlier positions untouched downstream
    assert d[4:].min() > 1e-4  # injected and later positions affected


def test_hooks_removed(tiny):
    m, layers = tiny
    ids = torch.randint(0, 64, (1, 5))
    out0 = m(input_ids=ids).logits
    with inject(layers, 0, Injection(torch.randn(32) * 10, torch.ones(1, 5, dtype=torch.bool))):
        out1 = m(input_ids=ids).logits
    out2 = m(input_ids=ids).logits
    assert not torch.allclose(out0, out1)
    assert torch.allclose(out0, out2)


def test_generation_on_generated_flag(tiny):
    m, layers = tiny
    ids = torch.randint(0, 64, (1, 5))
    v = torch.randn(32) * 20
    no_mask = torch.zeros(1, 5, dtype=torch.bool)
    gen = lambda: m.generate(input_ids=ids, max_new_tokens=6, do_sample=False, pad_token_id=0)
    base = gen()
    with inject(layers, 1, Injection(v, no_mask, on_generated=False)):
        same = gen()
    with inject(layers, 1, Injection(v, no_mask, on_generated=True)):
        steered = gen()
    assert torch.equal(base, same)
    assert not torch.equal(base, steered)


def test_none_injection_is_noop(tiny):
    m, layers = tiny
    ids = torch.randint(0, 64, (1, 5))
    with inject(layers, 0, None):
        a = m(input_ids=ids).logits
    assert torch.allclose(a, m(input_ids=ids).logits)


def test_mask_shape_checked(tiny):
    m, layers = tiny
    with pytest.raises(ValueError):
        with inject(layers, 0, Injection(torch.zeros(32), torch.ones(1, 3, dtype=torch.bool))):
            m(input_ids=torch.randint(0, 64, (1, 5)))
