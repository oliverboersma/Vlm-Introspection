#!/usr/bin/env python3
"""Run the seen-or-thought pilot end to end.

    python scripts/run_pilot.py --out results/pilot              # full CPU pilot (~1 h on 4 cores)
    python scripts/run_pilot.py --out results/quick --quick      # calibration run (~3 min)
    python scripts/run_pilot.py --out results/pilot --stage freeform

Stages: extract (concept vectors) → sweep (forced-choice probes) → freeform
(generated text, keyword-graded). Vectors are cached in ``<out>/vectors.pt``.
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import torch

from vlm_introspection.concepts import ConceptVectors, extract, extract_text
from vlm_introspection.experiment import EXTRACT_SEEDS, Runner, SweepConfig, save_config
from vlm_introspection.model import VLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--vectors", choices=["image", "text"], default=None,
                    help="derive concept vectors from images (default) or text-only prompts")
    ap.add_argument("--quick", action="store_true", help="small calibration grid")
    ap.add_argument("--stage", choices=["all", "sweep", "freeform"], default="all")
    ap.add_argument("--config", help="JSON file overriding SweepConfig fields")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    cfg = SweepConfig()
    if args.quick:
        cfg = dataclasses.replace(cfg, concepts=["red", "circle"], layers=[8, 16, 24], alphas=[1.0, 2.0, 4.0, 8.0],
                                  n_targets=2, n_extract=8, gen_layers=[12], gen_alphas=[4.0], gen_targets=1)
    if args.config:
        cfg = dataclasses.replace(cfg, **json.loads(Path(args.config).read_text()))
    if args.vectors:
        cfg = dataclasses.replace(cfg, vector_source=args.vectors)
    if args.model:
        cfg = dataclasses.replace(cfg, model_id=args.model)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out)
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    vlm = VLM(cfg.model_id, device=args.device, dtype=dtype)
    print(f"{cfg.model_id}: {vlm.n_layers} layers", flush=True)

    vpath = out / "vectors.pt"
    if vpath.exists():
        cv = ConceptVectors.load(vpath)
        assert set(cfg.layers + cfg.gen_layers) <= set(cv.layers), "cached vectors miss some layers"
    else:
        t = time.time()
        layers = sorted(set(cfg.layers + cfg.gen_layers))
        if cfg.vector_source == "text":
            cv = extract_text(vlm, layers, concepts=cfg.concepts)
        else:
            cv = extract(vlm, layers, n_images=cfg.n_extract, seed_offset=EXTRACT_SEEDS, concepts=cfg.concepts)
        cv.save(vpath)
        print(f"[extract] {len(cv.vectors)} concepts × {len(layers)} layers in {time.time() - t:.0f}s", flush=True)

    runner = Runner(vlm, cv, cfg, out)
    if args.stage in ("all", "sweep"):
        runner.sweep()
    if args.stage in ("all", "freeform"):
        runner.freeform()


if __name__ == "__main__":
    main()
