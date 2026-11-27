"""Development evaluation for the agent: same items and scoring as the checker, but
different seeds and context phrasings (the checker's are held out).

    python dev_eval.py witness.json [--seeds 1 2 3] [--concepts red circle ...] [--json]
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calib  # noqa: E402

DEV_CONTEXTS = ("Think about {idea}.", "Imagine {idea}.", "Keep {idea} in mind.")
_VLM = None


def get_vlm():
    global _VLM
    if _VLM is None:
        import torch
        from vlm_introspection.model import VLM
        gpu = torch.cuda.is_available()
        _VLM = VLM(os.environ.get("CALIB_MODEL", "HuggingFaceTB/SmolVLM-500M-Instruct"),
                   device="cuda" if gpu else "cpu", dtype=torch.bfloat16 if gpu else torch.float32)
    return _VLM


def evaluate(witness, seeds=(1, 2), concepts=None, contexts=DEV_CONTEXTS):
    errs = calib.validate(witness)
    if errs:
        raise ValueError("; ".join(errs))
    items = calib.build_items(seeds, contexts, concepts)
    scores = calib.score(get_vlm(), witness, items)
    return calib.summarize(items, scores), items, scores


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("witness")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--concepts", nargs="+")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    s, items, scores = evaluate(json.load(open(a.witness)), a.seeds, a.concepts)
    if a.json:
        print(json.dumps({"summary": s, "items": [{**it, "score": sc} for it, sc in zip(items, scores)]}))
    else:
        print(calib.report(s))
