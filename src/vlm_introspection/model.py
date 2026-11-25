"""Thin wrapper around a Hugging Face VLM: batching, token masks, probes.

Tested with SmolVLM (Idefics3). Other chat VLMs that use an image placeholder
token (LLaVA, Qwen2-VL, ...) should work as long as ``find_decoder_layers`` and
``image_token_id`` resolve; see the asserts in ``VLM.__init__``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from PIL import Image

from .hooks import Injection, capture, inject

YES_VARIANTS = ("Yes", " Yes", "yes", " yes", "YES", " YES")
NO_VARIANTS = ("No", " No", "no", " no", "NO", " NO")


def find_decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """Locate the language model's decoder ``ModuleList`` (not the vision tower's)."""
    candidates = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and name.endswith("layers") and "vision" not in name:
            candidates.append((name, mod))
    if not candidates:
        raise ValueError("no decoder layer list found")
    preferred = [c for c in candidates if "text_model" in c[0] or "language_model" in c[0]]
    return max(preferred or candidates, key=lambda c: len(c[1]))[1]


def _image_token_id(model) -> int:
    cfg = model.config
    for key in ("image_token_id", "image_token_index"):
        if getattr(cfg, key, None) is not None:
            return int(getattr(cfg, key))
    raise ValueError("model config has no image token id")


@dataclass
class Batch:
    inputs: dict[str, torch.Tensor]
    image_mask: torch.Tensor  # (B, T) positions holding image tokens
    text_mask: torch.Tensor  # (B, T) non-pad, non-image prompt positions


class VLM:
    def __init__(self, model_id: str, device: str = "cpu", dtype: torch.dtype = torch.float32,
                 split_images: bool = False):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_id = model_id
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        ip = getattr(self.processor, "image_processor", None)
        if ip is not None and hasattr(ip, "do_image_splitting"):
            ip.do_image_splitting = split_images  # 64 image tokens instead of ~1k for SmolVLM
        self.processor.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype).to(device).eval()
        self.layers = find_decoder_layers(self.model)
        self.n_layers = len(self.layers)
        self.image_token_id = _image_token_id(self.model)
        tok = self.processor.tokenizer
        self.yes_ids = sorted({i for w in YES_VARIANTS for i in tok.encode(w, add_special_tokens=False)[:1]})
        self.no_ids = sorted({i for w in NO_VARIANTS for i in tok.encode(w, add_special_tokens=False)[:1]})
        assert not set(self.yes_ids) & set(self.no_ids)
        self._feat_cache: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------ inputs
    def build(self, questions: list[str], images: list[Image.Image | None], cached: bool = False) -> Batch:
        """Tokenize a batch. ``images[i] is None`` gives a text-only prompt.

        A batch must be all-image or all-text (processor limitation).
        ``cached=True`` reuses vision-tower outputs across calls (single forward
        passes only; ``generate`` always uses raw pixels).
        """
        assert len(questions) == len(images)
        has_img = [im is not None for im in images]
        assert all(has_img) or not any(has_img), "mix of image and text-only prompts in one batch"
        prompts = []
        for q, im in zip(questions, images):
            content = ([{"type": "image"}] if im is not None else []) + [{"type": "text", "text": q}]
            msgs = [{"role": "user", "content": content}]
            prompts.append(self.processor.apply_chat_template(msgs, add_generation_prompt=True))
        kwargs = dict(text=prompts, return_tensors="pt", padding=True)
        if all(has_img):
            kwargs["images"] = [[im] for im in images]
        inputs = dict(self.processor(**kwargs).to(self.device))
        ids, attn = inputs["input_ids"], inputs["attention_mask"].bool()
        image_mask = ids == self.image_token_id
        if cached and all(has_img) and self.supports_feature_cache:
            inputs = self._swap_in_cached_features(inputs, images)
        # The first real token of each row is an attention sink (residual norm
        # ~100x the others in SmolVLM); keep it out of the text injection site.
        first = attn.float().argmax(dim=1)
        sink = torch.zeros_like(attn)
        sink[torch.arange(len(first)), first] = True
        return Batch(inputs, image_mask, attn & ~image_mask & ~sink)

    # ----------------------------------------------------- vision feature cache
    @property
    def supports_feature_cache(self) -> bool:
        import inspect

        return "mm_encoder_outputs" in inspect.signature(self.model.forward).parameters and hasattr(
            self.model.model, "get_image_features")

    @torch.no_grad()
    def _swap_in_cached_features(self, inputs: dict, images: list[Image.Image]) -> dict:
        """Replace pixel inputs by cached post-connector image features.

        The vision tower is ~85% of a SmolVLM forward pass on CPU, and the probe
        sweep shows each image to the language model dozens of times, so this is
        the main speed lever. Injection happens in the language model, after the
        features are merged, so caching does not interact with it.
        """
        from transformers.modeling_outputs import BaseModelOutputWithPooling

        keys = [hashlib.sha1(im.tobytes()).hexdigest() + str(im.size) for im in images]
        missing = [i for i, k in enumerate(keys) if k not in self._feat_cache]
        if missing:
            pv = inputs["pixel_values"][missing]
            pam = inputs["pixel_attention_mask"][missing]
            feats = self.model.model.get_image_features(pv, pam, return_dict=True).pooler_output
            feats = torch.cat(list(feats)) if isinstance(feats, (list, tuple)) else feats
            per = feats.reshape(len(missing), -1, feats.shape[-1])
            for j, i in enumerate(missing):
                self._feat_cache[keys[i]] = per[j]
        out = {k: v for k, v in inputs.items() if k not in ("pixel_values", "pixel_attention_mask")}
        stacked = torch.cat([self._feat_cache[k] for k in keys])
        out["mm_encoder_outputs"] = {"image": BaseModelOutputWithPooling(pooler_output=stacked)}
        return out

    def site_mask(self, batch: Batch, site: str) -> torch.Tensor:
        if site == "image":
            if not batch.image_mask.any():
                raise ValueError("site='image' on a text-only batch")
            return batch.image_mask
        if site == "text":
            return batch.text_mask
        if site == "all":
            return batch.image_mask | batch.text_mask
        raise ValueError(site)

    # ------------------------------------------------------------- activations
    @torch.no_grad()
    def mean_acts(self, images: list[Image.Image], layer_ids: list[int], question: str = "Describe this image.",
                  pool: str = "image", batch_size: int = 8) -> dict[int, torch.Tensor]:
        """Per-image mean residual activation over ``pool`` positions → ``{L: (N, d)}``."""
        out: dict[int, list[torch.Tensor]] = {L: [] for L in layer_ids}
        for i in range(0, len(images), batch_size):
            chunk = images[i : i + batch_size]
            b = self.build([question] * len(chunk), chunk, cached=True)
            mask = self.site_mask(b, pool).float()
            with capture(self.layers, layer_ids) as store:
                self.model(**b.inputs)
            for L in layer_ids:
                h = store[L].float()
                out[L].append((h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True))
        return {L: torch.cat(v) for L, v in out.items()}

    @torch.no_grad()
    def token_norms(self, images: list[Image.Image | None], layer_ids: list[int], site: str,
                    question: str = "Describe this image.") -> dict[int, float]:
        """Median L2 norm of residual vectors at ``site`` positions, per layer (for scaling).

        Median, not mean: a handful of massive-activation tokens would otherwise
        set the scale (see ``build``).
        """
        b = self.build([question] * len(images), images, cached=True)
        mask = self.site_mask(b, site)
        with capture(self.layers, layer_ids) as store:
            self.model(**b.inputs)
        return {L: store[L].float().norm(dim=-1)[mask].median().item() for L in layer_ids}

    # ------------------------------------------------------------------ probes
    @torch.no_grad()
    def yes_no(self, questions: list[str], images: list[Image.Image | None], layer: int | None = None,
               vector: torch.Tensor | None = None, site: str = "image") -> torch.Tensor:
        """log P(yes) − log P(no) for the first answer token, optionally under injection."""
        b = self.build(questions, images, cached=True)
        inj = None
        if vector is not None:
            inj = Injection(vector, self.site_mask(b, site))
        with inject(self.layers, layer if layer is not None else 0, inj):
            logits = self.model(**b.inputs).logits[:, -1, :].float()
        lp = logits.log_softmax(-1)
        return (lp[:, self.yes_ids].logsumexp(-1) - lp[:, self.no_ids].logsumexp(-1)).cpu()

    @torch.no_grad()
    def generate(self, questions: list[str], images: list[Image.Image | None], layer: int | None = None,
                 vector: torch.Tensor | None = None, site: str = "image", max_new_tokens: int = 24,
                 on_generated: bool | None = None) -> list[str]:
        b = self.build(questions, images)
        inj = None
        if vector is not None:
            # Text-site injection persists into the answer, as in concept-injection work;
            # image-site injection only touches the image tokens.
            og = (site != "image") if on_generated is None else on_generated
            inj = Injection(vector, self.site_mask(b, site), on_generated=og)
        with inject(self.layers, layer if layer is not None else 0, inj):
            out = self.model.generate(**b.inputs, max_new_tokens=max_new_tokens, do_sample=False)
        n = b.inputs["input_ids"].shape[1]
        return [s.strip() for s in self.processor.batch_decode(out[:, n:], skip_special_tokens=True)]
