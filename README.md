# Function calling: fine-tuning SmolLM2, and auditing the benchmark that scores it

Fine-tuning SmolLM2 on [`Salesforce/xlam-function-calling-60k`](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) — full fine-tune of a 135M, then LoRA and QLoRA on a 1.7B — with an evaluation harness built to be defensible, and an LLM-as-judge audit of the evaluation itself.

The headline is not the score. It is that **20.5% of the benchmark's failures turned out to have ground truth no model could have produced.**

## Results

Exact match on 2,395 held-out examples, after removing contaminated rows. 95% Wilson intervals.

| Model | Method | Exact match | 95% CI | Peak GPU | Train time |
|---|---|---|---|---|---|
| SmolLM2-1.7B-Instruct | LoRA | **83.47%** | [81.92, 84.90] | 9.48 GB | ~1h45 |
| SmolLM2-1.7B | LoRA | 75.11% | [73.34, 76.81] | 9.48 GB | 1h45 |
| SmolLM2-1.7B | QLoRA (NF4) | 72.61% | [70.79, 74.36] | 5.87 GB | 1h36 |
| SmolLM2-1.7B-Instruct | Few-shot prompting, no fine-tune | 39.58% | [37.64, 41.56] | — | — |
| SmolLM2-1.7B-Instruct | Training prompt, no fine-tune | 0.13% | [0.04, 0.37] | — | — |
| SmolLM2-135M | Full fine-tune | 21.80% | [20.19, 23.49] | — | — |
| SmolLM2-135M-Instruct | Best prompting | 1.60% | — | — | — |
| — | Always first tool, no args | 0.20% | — | — | — |

The LoRA/QLoRA gap is significant under a paired McNemar test (136 vs 76 discordant pairs, χ² = 16.42, p = 5.1e-05). QLoRA trades 2.5 points for 38% less memory — and finished 9 minutes *faster* on the same A100, because 4-bit weights mean fewer bytes moved per token.

Starting the same LoRA from the Instruct checkpoint instead of the base adds 8.4 points (234 vs 34 discordant pairs, exact McNemar p ≈ 7e-38). Why that experiment was run is below, in [Generalisation to unseen tools](#generalisation-to-unseen-tools).

The 135M run had no generalisation gap at all (train and eval loss within 0.02), which is what motivated scaling rather than collecting more data.

## Evaluation

Six nested levels, each strictly harder than the last, so a failure is located rather than just counted:

| Level | LoRA 1.7B | QLoRA 1.7B | LoRA 1.7B-Instruct |
|---|---|---|---|
| 1. Valid JSON | 99.21% | 99.46% | 99.75% |
| 2. Schema `[{name, arguments}]` | 99.21% | 99.46% | 99.75% |
| 3. No hallucinated tool | 98.87% | 99.21% | 99.71% |
| 4. Correct call count | 97.04% | 97.08% | 98.83% |
| 5. Correct tool names | 95.62% | 95.78% | 98.29% |
| 6. Correct argument values (= exact match) | 75.11% | 72.61% | 83.47% |

Everything structural is above 95%. The whole gap is argument values — for the fine-tuned models. Prompted models fail much earlier, at the schema, which is the subject of the next section.

What the harness does beyond scoring:

- **Decontamination.** `train_test_split` partitions rows, not content: 93 queries appeared on both sides, spanning 105 rows. They are removed (2,500 → 2,395).
- **Baselines.** Base model, Instruct, few-shot prompting, raw completion without a chat template, and a trivial baseline (always the first tool, empty arguments) — so the fine-tuned number has something to mean something *against*.
- **Pinned revisions.** Scores attach to a commit SHA, never to `main`, so republishing a model cannot silently change what was measured.
- **Calibrated generation.** `max_new_tokens=512` derived from the longest reference in the eval set (410 tokens), not guessed. Left padding, greedy decoding, truncation flagged per example.
- **Per-example persistence.** Every prediction and verdict is pushed to the Hub, so results can be re-analysed without re-running generation.

## Was fine-tuning necessary?

Two comparisons answer it, and they do not point the same way.

**At equal size, fine-tuning wins by a wide margin.** Prompting SmolLM2-1.7B-Instruct with the output format spelled out and three examples reaches 39.58%, against 75.11% for the same base fine-tuned with LoRA and 83.47% fine-tuned from Instruct. With the plain training prompt — which lists the tools but never states the output format — prompting collapses to 0.13%.

The level breakdown says where it breaks. The prompted model produces a valid schema only **52.65%** of the time: half its outputs are not the right shape at all, despite explicit instructions and three examples. It is not losing on argument values, it is losing on format compliance. That is what the fine-tune buys — a convention held in the weights instead of restated in every prompt.

**Against a model 140× larger, it is a draw.** `eval_api_job.py` evaluates Qwen3-235B-A22B-Instruct through HF Inference Providers on a fixed 300-example subset of the same held-out set, with the same scorer and the same prompts.

| Same 300 examples | Exact match | 95% CI |
|---|---|---|
| **LoRA from Instruct, 1.7B** | **80.00%** | [75.1, 84.1] |
| Qwen3-235B, format prompt | 78.33% | [73.3, 82.6] |
| LoRA from base, 1.7B | 72.67% | [67.4, 77.4] |
| QLoRA from base, 1.7B | 70.00% | [64.6, 74.9] |
| Qwen3-235B, training prompt | 0.00% | [0.0, 1.3] |

The 1.7B fine-tune and the 235B generalist are statistically indistinguishable (paired McNemar, 20 vs 15 discordant pairs, p = 0.50). The 235B does beat the *base*-model LoRA (30 vs 13, p = 0.015).

So fine-tuning was not necessary to reach this quality — a large general model reaches it by prompting alone. It was necessary to reach it **at 1.7B**: 3.4 GB running locally, no per-call cost, nothing leaving the machine. And the last row carries the same lesson as at 1.7B: with the format unstated, scale does not save you either.

Caveats: n = 300 for the API conditions, so ±5 points. Qwen3-235B is also the judge used in the next section — here it is evaluated, and it never judges its own outputs.

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

The judge was run on the LoRA and QLoRA from the base model only, not on the Instruct-based LoRA.

### Checking the judge by hand (in progress)

The 257 flagged failures are 147 distinct examples (both models fail on many of the same ones). For 114 of them, both verdicts say "not derivable"; for 33, one verdict says "not derivable" and the other says "derivable". Since derivability is a property of the reference, not of the prediction, those 33 are judge self-contradictions.

`audit_xlam.py` maps each flagged example back to its original xLAM `id` and samples cases for blind human review (`audit_xlam/verifier.py` serves a local review page that hides the judge's verdict until the reviewer has answered).

All 33 contradictory cases have been reviewed: **9 references are truly not derivable, 15 are derivable, 9 are uncertain.** When the judge contradicts itself, it is usually wrong to call the reference impossible. The confirmed defects fall into a few kinds: opaque IDs with no documented mapping (a league or sport ID copied from the parameter example), unreplaced placeholders (`"<token_value>"`, credentials `"mysecret"` / `"mytoken"`), identifiers that are close but wrong (BART station `CIV` for `CIVC`), a Python expression instead of a value (`"math.pi"`), and a query word put in the wrong parameter (`datum: "tokyo"`). A random sample of 50 of the 114 consistent flags is next; it will give the judge's precision, and a corrected estimate of the unusable-reference rate. The 20.5% above should be read as an upper bound until then.

## Generalisation to unseen tools

The random split tests new phrasings of known tools, never new tools. `test_outils_inconnus.py` probes that directly: 20 hand-written queries against the 4 tools of my own MCP server ([mcp-veille-nlp](https://huggingface.co/spaces/Chloemp/mcp-veille-nlp): trending Hub repos, GitHub repos, arXiv papers, release notes), none of which appear in xLAM. 15 queries need one call, 5 need several. Run locally (Apple M4 Pro, MPS, greedy); the base LoRA reproduces 28 of 30 recorded A100 predictions on held-out xLAM, so the local setup matches the evaluation.

| Model | Exact match (defaults normalised) | Strict |
|---|---|---|
| LoRA from base | 60% | 45% |
| LoRA from Instruct | 70% | 55% |
| SmolLM2-1.7B-Instruct, format prompt, no fine-tune | **80%** | 60% |

The base LoRA gains 35 points over prompted Instruct on known tools, and loses 20 on new ones. Its errors are xLAM habits rather than misreadings: it never once chose `search_trending_repos` (0 of 6 queries that needed it), passed a list parameter as a string, and copied the example value from a parameter description into a call nobody asked for.

That left two explanations. The LoRA starts from the base model, which never had general post-training, so the comparison might just reflect the starting point; or fine-tuning on xLAM erodes a general skill. Retraining with everything identical except the starting checkpoint (`train_xlam_peft_job.py --base instruct`) separates them:

- The **starting point explains most of it**: +8.4 points on known tools (above) and +10 on unseen ones.
- **Some specialisation cost may remain**: the Instruct-based LoRA is still 2 queries below the un-tuned Instruct on unseen tools, and one of its errors is new, an invented tool name (`get_trending_ds`). Two queries out of 20 is not significant (exact McNemar p = 0.5); settling it needs a larger unseen-tool set.

Caveats: 20 queries, written and labelled by one person; tool descriptions translated to English; the four failures shared by all three models come from an ambiguity between "papers" and "repos" in my own descriptions.

## Repository

| File | What it does |
|---|---|
| `train_xlam_job.py` | Full fine-tune of SmolLM2-135M (HF Jobs, PEP 723 inline deps) |
| `train_xlam_peft_job.py` | LoRA / QLoRA on SmolLM2-1.7B (`--base instruct` to start from the Instruct checkpoint), resumable from Hub checkpoints |
| `eval_xlam_job.py` | 6-level scorer, 12 model conditions plus a trivial baseline, decontamination, results pushed to the Hub |
| `judge_xlam_job.py` | LLM-as-judge audit, checkpointed and resumable |
| `eval_api_job.py` | Same eval set and scorer against a large model via HF Inference Providers, no GPU |
| `test_outils_inconnus.py` | Local probe on 20 queries against 4 tools absent from xLAM |
| `audit_xlam.py`, `audit_xlam/` | Maps flagged references back to xLAM ids; local blind review page |
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

# Large-model baseline through an API (no GPU either)
uv run eval_api_job.py --limit 10     # smoke test and cost estimate
uv run eval_api_job.py                # 300 examples, 2 prompt conditions
```

## Artifacts

**Models** — [smollm2-1.7b-instruct-xlam-lora](https://huggingface.co/Chloemp/smollm2-1.7b-instruct-xlam-lora) · [smollm2-1.7b-xlam-lora](https://huggingface.co/Chloemp/smollm2-1.7b-xlam-lora) · [smollm2-1.7b-xlam-qlora](https://huggingface.co/Chloemp/smollm2-1.7b-xlam-qlora) · [smollm2-135m-xlam-fullft](https://huggingface.co/Chloemp/smollm2-135m-xlam-fullft)

**Datasets** — [eval results](https://huggingface.co/datasets/Chloemp/xlam-smollm2-eval-results) (25,155 scored predictions across 17 conditions) · [judge verdicts](https://huggingface.co/datasets/Chloemp/xlam-smollm2-judge-results) (4,790 verdicts)

## Setup

SmolLM2-1.7B, LoRA r=16 α=32 on all linear layers, lr 2e-4, 3% warmup, 1 epoch (3,563 steps), effective batch 16, bf16, single A100. QLoRA identical but with an NF4 double-quantised base and bf16 compute. The Instruct-based LoRA is identical to the base LoRA apart from the starting checkpoint; the only side difference is the tokenizer's eos (`<|im_end|>` instead of `<|endoftext|>`), which matches the end-of-turn token of the shared chat template. It was trained in two jobs (interrupted at step 1500, resumed from the full Hub checkpoint). Data split with `test_size=0.05, seed=42`; the first 500 held-out examples are excluded because they were used for eval during training.

QLoRA adapters are evaluated on a 4-bit base with the training quantisation config, not merged into bf16: the adapter learned to correct a dequantised NF4 model, so grading it on the bf16 base would measure a different model.

## Limitations

- **The split does not test unseen tools.** A random split of xLAM means nearly every tool in the eval set also appears in training. The headline numbers measure reliable calling of a known toolset on new phrasing; the unseen-tool probe above is only 20 queries.
- **The judge's calibration is partial.** Human review so far covers only examples the judge flagged, which measures its precision but not what it misses; a Cohen's κ needs a sample of unflagged failures too.
- **One configuration.** One epoch, one LoRA rank, one learning rate. The LoRA/QLoRA comparison holds at exactly one point in that space.
- **One dataset, single-turn.** No external benchmark, so nothing here is directly comparable to BFCL or similar leaderboards.
- **The large-model comparison is a 300-example subset.** Enough to show a draw rather than a gap, not enough to resolve a few points either way.

Code comments are in French.
