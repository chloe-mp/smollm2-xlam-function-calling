# Function calling: fine-tuning SmolLM2, and auditing the benchmark that scores it

Fine-tuning SmolLM2 on [`Salesforce/xlam-function-calling-60k`](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) — full fine-tune of a 135M, then LoRA and QLoRA on a 1.7B — with an evaluation harness built to be defensible, and an LLM-as-judge audit of the evaluation itself.

The headline is not the score. It is that **20.5% of the benchmark's failures turned out to have ground truth no model could have produced.**

## Results

Exact match on 2,395 held-out examples, after removing contaminated rows. 95% Wilson intervals.

| Model | Method | Exact match | 95% CI | Peak GPU | Train time |
|---|---|---|---|---|---|
| SmolLM2-1.7B | LoRA | **75.11%** | [73.34, 76.81] | 9.48 GB | 1h45 |
| SmolLM2-1.7B | QLoRA (NF4) | 72.61% | [70.79, 74.36] | 5.87 GB | 1h36 |
| SmolLM2-135M | Full fine-tune | 21.80% | [20.19, 23.49] | — | — |
| SmolLM2-135M-Instruct | Best prompting | 1.60% | — | — | — |
| — | Always first tool, no args | 0.20% | — | — | — |

The LoRA/QLoRA gap is significant under a paired McNemar test (136 vs 76 discordant pairs, χ² = 16.42, p = 5.1e-05). QLoRA trades 2.5 points for 38% less memory — and finished 9 minutes *faster* on the same A100, because 4-bit weights mean fewer bytes moved per token.

The 135M run had no generalisation gap at all (train and eval loss within 0.02), which is what motivated scaling rather than collecting more data.

## Evaluation

Six nested levels, each strictly harder than the last, so a failure is located rather than just counted:

| Level | LoRA 1.7B | QLoRA 1.7B |
|---|---|---|
| 1. Valid JSON | 99.21% | 99.46% |
| 2. Schema `[{name, arguments}]` | 99.21% | 99.46% |
| 3. No hallucinated tool | 98.87% | 99.21% |
| 4. Correct call count | 97.04% | 97.08% |
| 5. Correct tool names | 95.62% | 95.78% |
| 6. Correct argument values (= exact match) | 75.11% | 72.61% |

Everything structural is above 95%. The whole gap is argument values.

What the harness does beyond scoring:

- **Decontamination.** `train_test_split` partitions rows, not content: 93 queries appeared on both sides, spanning 105 rows. They are removed (2,500 → 2,395).
- **Baselines.** Base model, Instruct, few-shot prompting, raw completion without a chat template, and a trivial baseline (always the first tool, empty arguments) — so the fine-tuned number has something to mean something *against*.
- **Pinned revisions.** Scores attach to a commit SHA, never to `main`, so republishing a model cannot silently change what was measured.
- **Calibrated generation.** `max_new_tokens=512` derived from the longest reference in the eval set (410 tokens), not guessed. Left padding, greedy decoding, truncation flagged per example.
- **Per-example persistence.** Every prediction and verdict is pushed to the Hub, so results can be re-analysed without re-running generation.

## The judge

The scorer compares against the reference and has no way to know whether the reference is usable. From the held-out set:

```
query:  "the weather forecast for the next 5 days and upcoming sports
         events in a specific location"
gold:   [{"name": "local_weather_api",
          "arguments": {"q": "London,uk", "num_of_days": 5}}, ...]
```

"London" appears nowhere in the request. No model could produce this, and the scorer counts it as a model failure.

`judge_xlam_job.py` quantifies how often that happens. Protocol:

- **Cross-family judge** (Qwen3-235B-A22B-Instruct) so it cannot prefer its own output style, at temperature 0, with the inference provider pinned — providers serve different quantisations, so a verdict should not depend on routing.
- **Blind**: the judge never sees which model produced a prediction, nor whether it passed exact match.
- **Two independent questions**, not one: *could any model derive this reference?* and *is this prediction correct?* In a single-field version, a "when in doubt, say incorrect" tie-break silently swallowed the unusable-reference signal.
- **Validated on known-correct cases.** All 4,790 predictions are judged, including the 3,538 that already matched exactly. Those are correct by construction, so any the judge calls wrong are measured judge errors.

**Judge agreement on the 3,538 exact matches: 99.89%** (4 disagreements, all objecting to the *reference* rather than favouring the model; 5 API/parse errors out of 4,790). Two full passes produced identical verdicts.

Of the 1,252 exact-match failures:

| | Count | Share |
|---|---|---|
| Genuine model errors | 960 | 76.7% |
| Judged correct anyway | 287 | 22.9% |
| **References not derivable from the request** | **257** | **20.5%** |

The unusable-reference rate is nearly identical for both models (5.3% and 5.6% of their examples), as it should be for a property of the dataset rather than the model.

Excluding unusable references, judged accuracy is **84.55%** (LoRA) and **82.29%** (QLoRA). Exact match remains the headline metric — it is deterministic, reproduces to the digit, and it was not badly wrong: 16 of the 25 missing points are genuine model errors.

Breakdown of the 960 real errors: wrong argument value 708, missing argument 89, wrong tool 81, wrong call count 74, malformed output 6.

## Repository

| File | What it does |
|---|---|
| `train_xlam_job.py` | Full fine-tune of SmolLM2-135M (HF Jobs, PEP 723 inline deps) |
| `train_xlam_peft_job.py` | LoRA / QLoRA on SmolLM2-1.7B, resumable from Hub checkpoints |
| `eval_xlam_job.py` | 6-level scorer, 11 model conditions plus a trivial baseline, decontamination, results pushed to the Hub |
| `judge_xlam_job.py` | LLM-as-judge audit, checkpointed and resumable |
| `inspect_xlam.py` | Single-example inspection — a debugging tool, never a source of numbers |
| `judge_metrics.json` | Aggregate judge results |
| `training_curve.svg`, `convergence_2395.svg` | Training curves; the second shows the task metric flat from step 2000 |

Scripts declare their dependencies inline (PEP 723), so `uv` installs them on the remote machine.

```bash
# Train (one job per method)
hf jobs uv run --flavor a100-large --timeout 3h --secrets HF_TOKEN \
  train_xlam_peft_job.py --method lora

# Evaluate
hf jobs uv run --flavor a100-large --timeout 2h --secrets HF_TOKEN eval_xlam_job.py

# Judge (network-bound, no GPU needed)
uv run judge_xlam_job.py --limit 20   # smoke test and cost estimate
uv run judge_xlam_job.py              # all 4,790 predictions
```

## Artifacts

**Models** — [smollm2-1.7b-xlam-lora](https://huggingface.co/Chloemp/smollm2-1.7b-xlam-lora) · [smollm2-1.7b-xlam-qlora](https://huggingface.co/Chloemp/smollm2-1.7b-xlam-qlora) · [smollm2-135m-xlam-fullft](https://huggingface.co/Chloemp/smollm2-135m-xlam-fullft)

**Datasets** — [eval results](https://huggingface.co/datasets/Chloemp/xlam-smollm2-eval-results) (17,370 scored predictions across 12 conditions) · [judge verdicts](https://huggingface.co/datasets/Chloemp/xlam-smollm2-judge-results) (4,790 verdicts)

## Setup

SmolLM2-1.7B, LoRA r=16 α=32 on all linear layers, lr 2e-4, 3% warmup, 1 epoch (3,563 steps), effective batch 16, bf16, single A100. QLoRA identical but with an NF4 double-quantised base and bf16 compute. Data split with `test_size=0.05, seed=42`; the first 500 held-out examples are excluded because they were used for eval during training.

QLoRA adapters are evaluated on a 4-bit base with the training quantisation config, not merged into bf16: the adapter learned to correct a dequantised NF4 model, so grading it on the bf16 base would measure a different model.

## Limitations

- **The split does not test unseen tools.** A random split of xLAM means nearly every tool in the eval set also appears in training. These numbers measure reliable calling of a known toolset on new phrasing, not generalisation to new tools.
- **The judge has no human-labelled calibration.** Agreement is measured against exact matches, which are easy cases; there is no Cohen's κ against hand-labelled hard cases, which is the usual standard.
- **One configuration.** One epoch, one LoRA rank, one learning rate. The LoRA/QLoRA comparison holds at exactly one point in that space.
- **One dataset, single-turn.** No external benchmark, so nothing here is directly comparable to BFCL or similar leaderboards.

Code comments are in French.
