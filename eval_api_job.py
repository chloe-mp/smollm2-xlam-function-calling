# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub>=0.34",
#     "transformers>=4.46",
#     "datasets",
#     "pandas",
#     "pyarrow",
#     "torch",
#     "peft",
# ]
# ///
"""
Baseline « gros modèle généraliste, sans fine-tuning » sur le même held-out.

Question à laquelle ce script répond : le fine-tuning était-il nécessaire, ou
un modèle généraliste suffisamment gros fait-il la tâche en prompting ?

Les baselines de prompting existantes (`base`, `instruct`, `base-fewshot`...)
sont toutes sur le 135M. Elles ne répondent pas à cette question : elles
mesurent un petit modèle, pas l'alternative réaliste au fine-tuning.

Ce script importe `eval_xlam_job` pour réutiliser À L'IDENTIQUE le jeu
d'évaluation (même split, même décontamination), le scorer à 6 niveaux et la
persistance sur le Hub. Un scorer dupliqué serait un scorer qui diverge.

Le modèle est appelé via HF Inference Providers : pas de GPU, ça tourne en
local. Échantillon de 300 par défaut (IC 95 % ≈ ±5 pts) : assez pour trancher
un écart de plusieurs dizaines de points, pas pour départager 2 pts.

    uv run eval_api_job.py --limit 10        # smoke test + coût réel
    uv run eval_api_job.py                   # 300 exemples, 2 conditions
    uv run eval_api_job.py --n 2395          # le held-out complet (plus cher)
"""

import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from huggingface_hub import InferenceClient

import eval_xlam_job as E

# Le juge de judge_xlam_job.py et ce modèle sont le même : ici il est ÉVALUÉ,
# pas juge. Les deux rôles ne se mélangent pas dans un même chiffre — aucune
# des conditions jugées par le juge n'est celle-ci.
API_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
PROVIDER = "scaleway"  # épinglé : débit et reproductibilité (cf. judge_xlam_job.py)

# (clé de condition, mode de prompt)
#   plain   : le prompt exact vu par les modèles fine-tunés — comparaison
#             stricte « même entrée, modèle différent »
#   fewshot : format explicite + 3 exemples. C'est la condition honnête pour
#             un généraliste : personne ne déploierait sans dire le format.
CONDITIONS = [("qwen3-235b", "plain"), ("qwen3-235b-fewshot", "fewshot")]

SAMPLE_N = 300
SEED = 42
MAX_WORKERS = 8
MAX_RETRIES = 3
MAX_TOKENS = E.MAX_NEW_TOKENS  # même plafond que la génération locale


def build_messages(ex: dict, mode: str, fewshot_block: str) -> list[dict]:
    """Mêmes textes système que le harnais local, sans chat template.

    L'API applique son propre template côté serveur : on lui passe des
    messages, pas une chaîne déjà rendue.
    """
    if mode == "plain":
        system = E.SYSTEM_PROMPT.format(tools=ex["tools"])
    else:
        system = E.SYSTEM_PROMPT_FEWSHOT.format(tools=ex["tools"], examples=fewshot_block)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": ex["query"]},
    ]


class Usage:
    def __init__(self):
        self.lock = threading.Lock()
        self.prompt = self.completion = self.calls = 0

    def add(self, u) -> None:
        with self.lock:
            self.calls += 1
            self.prompt += getattr(u, "prompt_tokens", 0) or 0
            self.completion += getattr(u, "completion_tokens", 0) or 0


def generate_one(client, ex: dict, mode: str, fewshot_block: str, usage: Usage) -> dict:
    for attempt in range(MAX_RETRIES):
        try:
            r = client.chat.completions.create(
                model=API_MODEL,
                messages=build_messages(ex, mode, fewshot_block),
                temperature=0,  # déterministe, comme le greedy du harnais local
                max_tokens=MAX_TOKENS,
            )
            usage.add(r.usage)
            choice = r.choices[0]
            return {
                "pred": (choice.message.content or "").strip(),
                "n_gen_tokens": getattr(r.usage, "completion_tokens", -1) or -1,
                # Équivalent du drapeau local : la génération a-t-elle été coupée ?
                "truncated": choice.finish_reason == "length",
            }
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                return {"pred": "", "n_gen_tokens": -1, "truncated": False, "error": f"{type(e).__name__}"}
            time.sleep(2**attempt)
    return {"pred": "", "n_gen_tokens": -1, "truncated": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=SAMPLE_N, help=f"taille d'échantillon (défaut {SAMPLE_N})")
    parser.add_argument("--limit", type=int, default=None, help="smoke test : ex. 10")
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    eval_ds, shots = E.get_clean_eval(token)
    fewshot_block = E.build_fewshot_block(shots)

    n = args.limit or min(args.n, len(eval_ds))
    # Échantillon FIXE (seed) : deux runs portent sur les mêmes exemples, et
    # les conditions se comparent ligne à ligne (McNemar possible).
    sample = eval_ds.shuffle(seed=SEED).select(range(n))
    print(f"Modèle : {API_MODEL} via {PROVIDER} | {n} exemples | conditions : {[c for c, _ in CONDITIONS]}")

    client = InferenceClient(provider=PROVIDER, token=token)
    usage = Usage()
    all_rows, metrics = [], {}
    t0 = time.time()

    for key, mode in CONDITIONS:
        print(f"\n===== {key} (mode={mode}) =====")
        rows = list(sample)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            # mode=mode : la lambda capturerait sinon la variable de boucle,
            # piège classique si l'exécution devient paresseuse.
            outs = list(pool.map(lambda ex, mode=mode: generate_one(client, ex, mode, fewshot_block, usage), rows))

        counts = {f"level_{i}": 0 for i in range(1, 7)}
        for ex, out in zip(rows, outs):
            tools_names = E.extract_tool_names(ex["tools"])
            scores = E.score_one(out["pred"], ex["answers"], tools_names)
            for k, v in scores.items():
                counts[k] += bool(v)
            all_rows.append({
                "model": key,
                "prompt_mode": mode,
                "query": ex["query"],
                "tools": ex["tools"],
                "gold": ex["answers"],
                "pred": out["pred"],
                "n_gen_tokens": out["n_gen_tokens"],
                "truncated": out["truncated"],
                **scores,
            })
        metrics[key] = {k: round(100.0 * v / len(rows), 2) for k, v in counts.items()}
        n_trunc = sum(o["truncated"] for o in outs)
        n_err = sum("error" in o for o in outs)
        print(f"  tronquées : {n_trunc}/{len(rows)} | erreurs API : {n_err}")
        for k, v in metrics[key].items():
            print(f"  {k}: {v}%")

    print(f"\n{usage.calls} appels en {(time.time() - t0) / 60:.1f} min | "
          f"{usage.prompt} tokens entrée + {usage.completion} sortie")

    pd.DataFrame(all_rows).to_parquet("api_eval_results.parquet", index=False)
    print("Écrit localement : api_eval_results.parquet")

    if args.no_push or args.limit:
        return
    # Même fusion et même push que le harnais : les conditions non rejouées
    # sont conservées, et la carte du dataset est réparée avant l'upload.
    run_keys = [k for k, _ in CONDITIONS]
    merged = E.merge_with_previous(all_rows, run_keys, token)
    E.push_results(merged, metrics, token)


if __name__ == "__main__":
    main()
