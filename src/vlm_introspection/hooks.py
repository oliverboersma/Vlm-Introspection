"""Residual-stream capture and injection via forward hooks on decoder layers.

Model-agnostic: everything operates on a ``torch.nn.ModuleList`` of decoder
layers, so the same code is unit-tested on a tiny randomly initialized Llama and
used on real VLM language backbones.

Convention: "layer L" means the *output* of decoder layer ``L`` (0-indexed),
i.e. the residual stream after block L has written to it.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch


def _hidden(output):
    return output[0] if isinstance(output, tuple) else output


def _replace_hidden(output, new):
    if isinstance(output, tuple):
        return (new, *output[1:])
    return new


@contextmanager
def capture(layers: torch.nn.ModuleList, layer_ids: list[int]) -> Iterator[dict[int, torch.Tensor]]:
    """Record the residual stream output of each layer in ``layer_ids``.

    Yields a dict that is filled during the forward pass: ``{L: (B, T, d)}``.
    Only the first (prefill) call per layer is kept.
    """
    store: dict[int, torch.Tensor] = {}
    handles = []
    for L in layer_ids:
        def hook(_mod, _inp, out, L=L):
            if L not in store:
                store[L] = _hidden(out).detach()
        handles.append(layers[L].register_forward_hook(hook))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


class Injection:
    """Add ``vector`` to the residual stream at one layer.

    Args:
        vector: (d,) direction already scaled to the desired magnitude, or
            (B, d) for a per-example vector.
        prefill_mask: (B, T) bool — which prompt positions receive the vector.
        on_generated: also add the vector at every newly generated token
            (decode steps, where the hook sees T == 1 with a KV cache).
    """

    def __init__(self, vector: torch.Tensor, prefill_mask: torch.Tensor, on_generated: bool = False):
        self.vector = vector
        self.prefill_mask = prefill_mask
        self.on_generated = on_generated
        self._prefilled = False

    def __call__(self, _mod, _inp, out):
        h = _hidden(out)
        v = self.vector.to(h.dtype).to(h.device)
        if v.dim() == 1:
            v = v[None, :]
        if not self._prefilled:
            self._prefilled = True
            mask = self.prefill_mask.to(h.device)
            if mask.shape != h.shape[:2]:
                raise ValueError(f"mask shape {tuple(mask.shape)} != hidden {tuple(h.shape[:2])}")
            h = h + mask[..., None].to(h.dtype) * v[:, None, :]
        elif self.on_generated:
            h = h + v[:, None, :]
        return _replace_hidden(out, h)


@contextmanager
def inject(layers: torch.nn.ModuleList, layer_id: int, injection: Injection | None):
    """Install ``injection`` on ``layers[layer_id]`` for the duration of the block.

    ``injection=None`` is a no-op, which keeps call sites uniform for the
    no-injection control condition.
    """
    if injection is None:
        yield
        return
    handle = layers[layer_id].register_forward_hook(injection)
    try:
        yield
    finally:
        handle.remove()
