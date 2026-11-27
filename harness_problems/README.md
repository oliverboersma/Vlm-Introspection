# Harness problems

Problems for the agentic harness in
[claudetest/harness](https://github.com/oliverboersma/claudetest/tree/claude/harness-recent-x-bookmark-5t83e2/harness).
Each one has a trusted, held-out evaluation. A result counts only when the
checker accepts it; whatever an attempt claims in its report doesn't count.

| Problem | Goal | Evaluation | Measured so far |
|---|---|---|---|
| [`think-probe-calibration`](think-probe-calibration/) | Next step 2: a forced-choice SEE / THINK / NEITHER probe for SmolVLM-500M that classifies calibration items of known provenance | 144 held-out items (3 seeds x 8 concepts x 6 types), held-out context phrasings; pass if every type >= 60 % and balanced >= 75 % (chance 33 %); ~2.5 min on 4 CPU cores | three hand-written wordings: 0.33-0.44 balanced on dev items; the 1/2/3 wording gets 0.43 held out |
| [`think-probe-calibration-2b`](think-probe-calibration-2b/) | the same on SmolVLM-2.2B (next step 1) | same checker, model from `CALIB_MODEL` | not measured (GPU) |

The six item types (c', c'' are other concepts of the same family):

| Type | Image | Sentence before the question | Correct |
|---|---|---|---|
| see | c | none | see |
| see_distractor | c | puts c' in mind | see |
| think | neutral | puts c in mind | think |
| think_distractor | c' | puts c in mind | think |
| neither | neutral | none | neither |
| neither_distractor | c' | puts c'' in mind | neither |

A probe that only checks perception can't separate think from neither. A probe
that only checks "is there a sentence about c" can't separate see from
see_distractor. Questions that refer to the prompt ("mentioned", "text",
"above", ...) are rejected, so the probe has to ask about the model's own state.

The measured wordings all separate "in the image" from "not in the image" well
(see 0.88-1.00). None of them separates "thinking about c" from "not": neither
0.00, think 0.25-0.38. That matches the README's finding 7. Whether any wording
can do it at 500M is the open question an agent run would test; a careful
negative result is evidence for moving to a larger model.

## Running

```bash
pip install -e .          # into the main site-packages: checkers run under python -I, which skips --user installs
python harness_problems/think-probe-calibration/dev_eval.py candidate.json --seeds 1 2   # dev score, ~1.5 min
python harness_problems/think-probe-calibration/checker.py candidate.json                # held-out score, ~2.5 min

cd ../claudetest/harness   # agent runs (needs ANTHROPIC_API_KEY)
python -m harness run ../../Vlm-Introspection/harness_problems/think-probe-calibration --config configs/default.toml
```

`candidate.json` looks like:

```json
{"question": "Look at the image carefully. Is {idea} visible in it? If yes, answer 1. If it is not visible but you were thinking about it, answer 2. If it is not visible and you were not thinking about it, answer 3.",
 "answers": {"see": "1", "think": "2", "neither": "3"}}
```

The held-out seeds and phrasings are in `checker.py`. The harness sandbox
limits time and memory but doesn't hide files, so an agent determined to read
them could. Treat a pass as a candidate and re-check it on fresh seeds before
relying on it.
