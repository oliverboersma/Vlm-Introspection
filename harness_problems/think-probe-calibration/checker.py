"""Trusted held-out evaluation of a SEE-vs-THINK provenance probe on SmolVLM-500M.

Reads the witness JSON (argv[1]):
    {"question": "...{idea}...", "answers": {"see": "A", "think": "B", "neither": "C"}}
Prints VALID and exits 0 only if, on held-out seeds and held-out context phrasings,
every one of the six item types is at least MIN_PER_TYPE correct and balanced
accuracy is at least MIN_BALANCED (chance is 1/3).
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I drops the script dir from sys.path
import calib  # noqa: E402

MODEL = os.environ.get("CALIB_MODEL", "HuggingFaceTB/SmolVLM-500M-Instruct")
SEEDS = (7001, 7002, 7003)
CONTEXTS = ("Picture {idea} in your head.", "Hold the thought of {idea} for a moment.",
            "Concentrate on {idea}.", "Let {idea} come to mind.")
MIN_PER_TYPE, MIN_BALANCED = 0.60, 0.75


def main():
    w = json.load(open(sys.argv[1]))
    errs = calib.validate(w)
    if errs:
        print("INVALID: " + "; ".join(errs))
        sys.exit(1)
    import torch
    from vlm_introspection.model import VLM
    torch.manual_seed(0)
    vlm = VLM(MODEL, device="cuda" if torch.cuda.is_available() else "cpu",
              dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    items = calib.build_items(SEEDS, CONTEXTS)
    s = calib.summarize(items, calib.score(vlm, w, items))
    print(calib.report(s))
    print(f"METRIC balanced_accuracy={s['balanced']:.4f}")
    for t in calib.TYPES:
        print(f"METRIC acc_{t}={s['per_type'][t]:.4f}")
    print(f"METRIC margin={s['margin']:.4f}")
    if min(s["per_type"].values()) >= MIN_PER_TYPE and s["balanced"] >= MIN_BALANCED:
        print(f"VALID: held-out balanced accuracy {s['balanced']:.3f}, worst type {min(s['per_type'].values()):.2f}")
        sys.exit(0)
    print(f"INVALID: needs every type >= {MIN_PER_TYPE} and balanced >= {MIN_BALANCED}")
    sys.exit(1)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
