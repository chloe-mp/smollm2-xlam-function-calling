# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub>=0.34",
#     "pandas",
#     "pyarrow",
#     "datasets",
# ]
# ///
"""
LLM-as-a-judge sur les prédictions de l'éval xLAM (adaptateurs 1.7B).

Ce que le juge AJOUTE au scorer déterministe (eval_xlam_job.py) : le scorer
compare à la vérité terrain et n'a aucun moyen de savoir si cette vérité est
utilisable. Exemple réel du held-out :

    query : "the weather forecast ... in a specific location"
    gold  : [{"name": "local_weather_api", "arguments": {"q": "London,uk", ...}}]

"London" n'apparaît nulle part dans la query : le modèle ne POUVAIT pas le
deviner. Le scorer compte ça comme un échec du modèle. Le juge le compte comme
une référence inutilisable. L'exact-match reste la métrique principale — le
juge sert à savoir quelle part des 25 % d'échecs est réelle.

Protocole :
- le juge ne sait PAS quel modèle a produit la prédiction, ni si elle a passé
  l'exact-match (sinon il ancre son verdict dessus) ;
- il juge les 4790 prédictions, pas seulement les échecs : les 3538 qui sont
  exactes servent de VALIDATION du juge (il doit les dire correctes ; ses
  "incorrect" sur celles-là sont des erreurs de juge, mesurables sans annoter
  quoi que ce soit à la main) ;
- température 0, et chaque appel est indépendant.

Lancement local (pas de GPU, c'est de l'appel réseau) :
    uv run judge_xlam_job.py --limit 20        # smoke test + estimation du coût
    uv run judge_xlam_job.py                   # les 4790

Ou sur HF Jobs (machine la moins chère, aucun GPU nécessaire) :
    hf jobs uv run --flavor cpu-basic --timeout 3h --secrets HF_TOKEN judge_xlam_job.py
"""

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from huggingface_hub import HfApi, InferenceClient, hf_hub_download

RESULTS_DATASET = "Chloemp/xlam-smollm2-eval-results"
JUDGE_DATASET = "Chloemp/xlam-smollm2-judge-results"
JUDGED_MODELS = ["lora-1.7b", "qlora-1.7b"]

# Juge d'une AUTRE famille que les modèles jugés (SmolLM2) : un juge évalue
# plus favorablement ses propres sorties (self-preference bias). Ici le risque
# est faible (le juge ne compare pas deux candidats) mais c'est gratuit.
JUDGE_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
# Provider ÉPINGLÉ, pour deux raisons :
# 1. débit : "auto" re-route à chaque appel et s'effondre en charge soutenue
#    (mesuré : 0,06 appel/s après 2 h, contre 4,0/s sur scaleway épinglé) ;
# 2. reproductibilité : deux providers servent des quantifications différentes
#    du même modèle et peuvent rendre des verdicts différents. Un run de juge
#    doit venir d'une seule pile de service, sinon le verdict dépend du routage.
PROVIDER = "scaleway"

MAX_WORKERS = 8
MAX_RETRIES = 3
CHECKPOINT = "judge_checkpoint.jsonl"

SYSTEM = """You grade an AI model's function-calling output against a reference answer.

You will see: the user's request, the tools available, the reference answer, and the prediction.

Judge whether the PREDICTION correctly answers the REQUEST, using the TOOLS available.

Critical: the reference answer is not automatically correct. It was written by annotators and
is sometimes unusable — most often it contains argument values that appear nowhere in the
request and could not be derived from it (a city, an ID, a date the request never mentions).
When the reference could not have been produced from the request by any model, say so instead
of penalising the prediction.

Answer TWO independent questions. Do not let one influence the other.

Q1 reference_derivable: could ANY model produce the reference from the request and the tools
alone? "no" when the reference contains a value the request never supplies and that cannot be
looked up from it — a city, coordinates, an ID, a date that appears nowhere in the request.
Judge the reference here, not the prediction.

Q2 verdict: is the PREDICTION a correct response to the REQUEST? Answer this even when Q1 is
"no", judging the prediction against the request itself.

Reply with JSON only, no other text:
{"reference_derivable": "yes" | "no",
 "verdict": "correct" | "incorrect",
 "category": "<one category below>",
 "reason": "<one short sentence>"}

verdict:
- "correct": the prediction would satisfy the request. Be strict: the deterministic scorer that
  ran before you ALREADY treated 5 and "5", "true" and true, and key order as equal, so any
  difference you are looking at is a real difference in value or structure, not typing.
  Two calls are equivalent ONLY if sending both to the same API returns the same result.
  These are NOT equivalent, and are "incorrect":
    * different values of any kind, including coordinates, IDs, URLs, city names or dates,
      even when both plausibly refer to the same place or thing
    * an argument present in the reference and missing from the prediction, or vice versa,
      even when its value looks like a default
    * a different syntax for the same intent, e.g. {"$lt": 30} vs "<30"
  Trailing-slash-only URL differences and pure whitespace differences ARE equivalent.
- "incorrect": the prediction would not satisfy the request.

When you hesitate between "correct" and "incorrect", answer "incorrect". This tie-break applies
to Q2 only — never let it push reference_derivable to "yes".

category (pick exactly one):
- identical — prediction matches the reference in substance
- equivalent_form — same meaning, different typing or formatting
- missing_argument — prediction omits an argument the request asked for
- wrong_value — an argument value contradicts the request
- wrong_tool — wrong tool chosen, or a tool not in the list
- wrong_count — wrong number of calls for what the request asks
- malformed — not valid, usable JSON tool calls
- reference_not_derivable — reference needs facts absent from the request
- other"""

USER_TEMPLATE = """REQUEST:
{query}

TOOLS AVAILABLE:
{tools}

REFERENCE ANSWER:
{gold}

PREDICTION:
{pred}"""


def build_messages(row: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                query=row["query"], tools=row["tools"], gold=row["gold"], pred=row["pred"] or "(empty output)"
            ),
        },
    ]


def parse_verdict(text: str) -> dict | None:
    """Le modèle répond parfois avec une clôture markdown ou du texte autour."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.DOTALL)  # dernier recours : le 1er objet JSON
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict) or obj.get("verdict") not in {"correct", "incorrect"}:
        return None
    if obj.get("reference_derivable") not in {"yes", "no"}:
        return None
    return obj


class Usage:
    """Compteur de tokens partagé entre threads : le coût EST un résultat."""

    def __init__(self):
        self.lock = threading.Lock()
        self.prompt = 0
        self.completion = 0
        self.calls = 0

    def add(self, u) -> None:
        with self.lock:
            self.calls += 1
            self.prompt += getattr(u, "prompt_tokens", 0) or 0
            self.completion += getattr(u, "completion_tokens", 0) or 0


def judge_one(client: InferenceClient, row: dict, usage: Usage) -> dict:
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=build_messages(row),
                temperature=0,  # déterministe : deux runs doivent donner le même verdict
                max_tokens=200,
            )
            usage.add(resp.usage)
            parsed = parse_verdict(resp.choices[0].message.content or "")
            if parsed:
                return {**row_key(row), **parsed}
            # Réponse non parsable : on retente, le modèle diverge parfois une fois.
        except Exception as e:  # réseau, rate limit, provider indisponible
            if attempt == MAX_RETRIES - 1:
                return {**row_key(row), "verdict": "error", "category": "api_error", "reason": f"{type(e).__name__}: {e}"[:200]}
            time.sleep(2**attempt)
    return {**row_key(row), "verdict": "error", "category": "unparsable", "reason": "réponse non parsable après retries"}


def row_key(row: dict) -> dict:
    return {"model": row["model"], "idx": row["idx"], "query": row["query"]}


def load_rows(limit: int | None, seed: int) -> pd.DataFrame:
    path = hf_hub_download(
        RESULTS_DATASET, "data/train-00000-of-00001.parquet", repo_type="dataset"
    )
    df = pd.read_parquet(path)
    df = df[df.model.isin(JUDGED_MODELS)].copy()
    # idx stable par modèle : sert de clé de reprise, indépendante de l'ordre.
    df["idx"] = df.groupby("model").cumcount()
    if limit:
        # Échantillon stratifié : autant d'exact-match que d'échecs, pour que le
        # smoke test mesure AUSSI l'accord du juge sur des cas connus corrects.
        ok = df[df.level_6].sample(n=limit // 2, random_state=seed)
        ko = df[~df.level_6].sample(n=limit - limit // 2, random_state=seed)
        df = pd.concat([ok, ko])
    return df


def load_checkpoint() -> dict[tuple, dict]:
    if not os.path.exists(CHECKPOINT):
        return {}
    done = {}
    with open(CHECKPOINT) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # ligne tronquée par une interruption : on la refera
            if r.get("verdict") != "error":  # une erreur mérite d'être rejouée
                done[(r["model"], r["idx"])] = r
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="smoke test : ex. 20")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-push", action="store_true", help="n'écrit que le local")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    df = load_rows(args.limit, args.seed)
    done = load_checkpoint()
    todo = [r for r in df.to_dict("records") if (r["model"], r["idx"]) not in done]
    print(f"À juger : {len(todo)} | déjà fait (checkpoint) : {len(done)} | juge : {JUDGE_MODEL}")

    client = InferenceClient(provider=PROVIDER, token=token)
    usage = Usage()
    t0 = time.time()
    lock = threading.Lock()
    results = list(done.values())

    with open(CHECKPOINT, "a") as ckpt, ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for i, res in enumerate(pool.map(lambda r: judge_one(client, r, usage), todo), 1):
            with lock:
                ckpt.write(json.dumps(res, ensure_ascii=False) + "\n")
                ckpt.flush()  # une interruption ne doit pas coûter les appels déjà payés
                results.append(res)
            if i % 50 == 0 or i == len(todo):
                rate = i / (time.time() - t0)
                print(f"  {i}/{len(todo)} | {rate:.1f}/s | reste ~{(len(todo) - i) / rate / 60:.0f} min", flush=True)

    judged = pd.DataFrame(results).merge(
        df[["model", "idx", "level_6", "gold", "pred", "tools"]], on=["model", "idx"], how="left"
    )
    judged.to_parquet("judge_results.parquet", index=False)
    report(judged, usage, time.time() - t0)

    if not args.no_push and not args.limit:
        api = HfApi(token=token)
        # Le repo n'existe pas au premier run : le créer AVANT l'upload, sinon
        # on perd les verdicts déjà payés sur la dernière ligne du script.
        api.create_repo(JUDGE_DATASET, repo_type="dataset", exist_ok=True)
        api.upload_file(
            path_or_fileobj="judge_results.parquet",
            path_in_repo="data/train-00000-of-00001.parquet",
            repo_id=JUDGE_DATASET,
            repo_type="dataset",
            commit_message=f"Verdicts {JUDGE_MODEL} sur {len(judged)} prédictions",
        )
        print(f"\nPoussé → https://huggingface.co/datasets/{JUDGE_DATASET}")


def report(judged: pd.DataFrame, usage: Usage, elapsed: float) -> None:
    print(f"\n===== JUGE : {len(judged)} verdicts en {elapsed / 60:.1f} min =====")
    print(f"tokens : {usage.prompt} entrée + {usage.completion} sortie sur {usage.calls} appels")
    if usage.calls:
        per = (usage.prompt + usage.completion) / usage.calls
        print(f"         soit ~{per:.0f} tokens/appel -> {per * 4790 / 1e6:.2f} M pour les 4790")

    print("\nverdicts :")
    for v, n in judged.verdict.value_counts().items():
        print(f"  {v:20s} {n:5d}  ({100 * n / len(judged):.1f} %)")
    nd = (judged.reference_derivable == "no").sum()
    print(f"  références non dérivables : {nd} ({100 * nd / len(judged):.1f} %)")

    # VALIDATION DU JUGE : sur les exact-match, la bonne réponse est "correct".
    ok = judged[judged.level_6]
    if len(ok):
        agree = (ok.verdict == "correct").mean()
        print(f"\naccord sur les {len(ok)} exact-match (doit être ~100 %) : {100 * agree:.1f} %")
        bad = ok[ok.verdict != "correct"]
        if len(bad):
            print(f"  {len(bad)} désaccords = erreurs du juge, ex. :")
            for _, r in bad.head(3).iterrows():
                print(f"    [{r.verdict}/{r.category}] {str(r.reason)[:110]}")

    # Ce pour quoi on a lancé le juge : combien d'échecs ne sont pas du modèle ?
    ko = judged[~judged.level_6]
    if len(ko):
        print(f"\nsur les {len(ko)} échecs de l'exact-match :")
        for v, n in ko.verdict.value_counts().items():
            print(f"  {v:20s} {n:5d}  ({100 * n / len(ko):.1f} %)")
        nd = (ko.reference_derivable == "no").sum()
        print(f"  dont référence non dérivable : {nd} ({100 * nd / len(ko):.1f} %)")
        print("\n  catégories :")
        for c, n in ko.category.value_counts().head(8).items():
            print(f"    {c:26s} {n:5d}")
        for model in sorted(judged.model.unique()):
            m = judged[judged.model == model]
            em = 100 * m.level_6.mean()
            usable = m[m.reference_derivable == "yes"]
            corr = 100 * (usable.verdict == "correct").mean() if len(usable) else float("nan")
            print(f"\n  {model:12s} exact-match {em:.2f} %  ->  jugé correct hors références inutilisables : {corr:.2f} % (n={len(usable)})")


if __name__ == "__main__":
    main()
