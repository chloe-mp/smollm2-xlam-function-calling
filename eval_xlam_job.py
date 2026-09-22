# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "transformers>=4.46",
#     "datasets",
#     "pandas",
#     "torch",
#     "hf_transfer",
#     "accelerate",
#     "peft",
#     "bitsandbytes",
# ]
# ///

import json
import logging
import os
import re
import sys
import threading
from collections import Counter
from typing import Any

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from huggingface_hub import DatasetCard, HfApi, hf_hub_download, login
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
TEST_SIZE = 0.05
EVAL_SKIP = 500
N_EVAL = 2395  # jeu propre complet (2500 - 105 lignes contaminées)
MAX_NEW_TOKENS = 512
DTYPE = torch.bfloat16

MODEL_FT = "Chloemp/smollm2-135m-xlam-fullft"
# Révision figée : les scores publiés se rattachent à ce SHA, pas à "main".
# Sans ça, republier le modèle change silencieusement ce que l'éval mesure.
REVISION_FT = "050f71474648a88c470640c1b820aff9b8aa6113"

# Adaptateurs LoRA / QLoRA sur SmolLM2-1.7B (train_xlam_peft_job.py).
# Ces repos ne contiennent que l'adaptateur (+ tokenizer) : le modèle de base
# est lu dans adapter_config.json. Révisions à figer sur le commit
# "End of training" de chaque repo, comme REVISION_FT — tant qu'elles valent
# None, l'éval de ces conditions refuse de tourner (voir load_model).
MODEL_LORA = "Chloemp/smollm2-1.7b-xlam-lora"
MODEL_QLORA = "Chloemp/smollm2-1.7b-xlam-qlora"
REVISION_LORA = "8868bcd45fb4942552222602f539148799d6e593"
REVISION_QLORA = "a7cd2365348afefd29ad03fd11ce8d0235d8ae78"

# Conditions dont le modèle de base est chargé en 4 bits, avec EXACTEMENT la
# config de quantification de l'entraînement. L'adaptateur QLoRA a appris à
# corriger un base NF4 déquantifié, pas le base bf16 : l'évaluer sur le bf16
# mesurerait un autre modèle que celui entraîné.
EVAL_4BIT_BASE = {"qlora-1.7b"}
BNB_4BIT = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

BASE_NAME = "HuggingFaceTB/SmolLM2-135M"
INSTRUCT_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"

# (clé, modèle, révision, mode de prompt)
#   plain        : system minimal + query, via chat template  (condition d'origine)
#   fewshot      : system + format explicite + N exemples, via chat template
#   raw_fewshot  : même texte SANS chat template (complétion pure, pour un base)
MODELS_TO_EVAL = [
    ("ft", MODEL_FT, REVISION_FT, "plain"),
    ("base", BASE_NAME, None, "plain"),
    ("instruct", INSTRUCT_NAME, None, "plain"),
    ("instruct-fewshot", INSTRUCT_NAME, None, "fewshot"),
    ("base-fewshot", BASE_NAME, None, "raw_fewshot"),
    # Le smoke test montre qu'instruct ignore la consigne de format via le chat
    # template. Cette condition isole la question : est-ce le modèle, ou le
    # template qui l'enferme en mode "assistant" ? Supprime-la si tu veux
    # raccourcir le run.
    ("instruct-raw", INSTRUCT_NAME, None, "raw_fewshot"),
    # Checkpoints intermédiaires, récupérés dans l'historique git du repo modèle
    # (hub_strategy="every_save" a poussé un commit tous les 500 steps).
    # But : l'eval loss plateaute dès ~2250 — la métrique de tâche aussi ?
    ("ft-step1000", MODEL_FT, "fa451a32b95595afc1be9227ba789936e352d72b", "plain"),
    ("ft-step2000", MODEL_FT, "e4c3406570f61f8a383f5a52f7d16a60c8d707e8", "plain"),
    ("ft-step3000", MODEL_FT, "ff2ffbad2e81f882900dcdeb91a861e28b71f5ac", "plain"),
    # Modèle plus gros, fine-tuning paramètre-efficace. Même prompt "plain"
    # que `ft` : les deux adaptateurs ont été entraînés sur le même template.
    ("lora-1.7b", MODEL_LORA, REVISION_LORA, "plain"),
    ("qlora-1.7b", MODEL_QLORA, REVISION_QLORA, "plain"),
]

# Conditions à exécuter dans ce run. Les autres sont conservées telles quelles
# depuis le dataset de résultats existant sur le Hub (pas de réévaluation GPU).
# Mets None pour tout relancer.
# Les checkpoints restants, sur les 2395, pour que la courbe exact-match vs
# step soit mesurée à la même taille d'échantillon ET au même batch_size que
# `ft` et `ft-step3000` (un batch_size différent déplace le score de ~1 pt).
RUN_ONLY = ["lora-1.7b", "qlora-1.7b"]

N_SHOTS = 3  # exemples de format, tirés du TRAIN (jamais du held-out)
MAX_INPUT_LEN = 3072  # marge : le few-shot rallonge le prompt

# Le modèle *base* n'a pas de chat_template (il n'a jamais vu de format
# conversationnel). On lui prête celui de l'Instruct, qui est byte-identique
# à celui du fine-tune : les trois modèles voient donc exactement le même
# prompt, sinon la comparaison ne mesure plus le modèle mais le formatage.
TEMPLATE_SOURCE = "HuggingFaceTB/SmolLM2-135M-Instruct"

SYSTEM_PROMPT = "You are a function-calling assistant. Available tools:\n{tools}"

# Le prompt d'origine ne dit nulle part quel format produire : `ft` le sait par
# entraînement, base/instruct ne pouvaient pas le deviner. Ici on le spécifie.
SYSTEM_PROMPT_FEWSHOT = (
    "You are a function-calling assistant. Available tools:\n{tools}\n\n"
    "Reply with a JSON array of tool calls and nothing else.\n"
    'Format: [{{"name": "<tool_name>", "arguments": {{"<arg>": "<value>"}}}}]\n'
    "Use only the tools listed above. One object per required call. "
    "No explanation, no markdown fences, no code.\n\n"
    "{examples}"
)

# Marqueurs de fin pour la complétion brute : un modèle base ne produit pas
# d'EOS, il enchaînerait sur un faux exemple suivant. Équivalent d'une
# stop-sequence, donc partie intégrante de la stratégie de prompting.
RAW_STOP_MARKERS = ["\nQuery:", "\n\n"]

HUB_DATASET_ID = "Chloemp/xlam-smollm2-eval-results"  # change si tu veux
RESULTS_REPO_PRIVATE = False


# ---------------------------------------------------------------------------
# Helpers dataset
# ---------------------------------------------------------------------------
def to_messages(example: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.format(tools=example["tools"])},
            {"role": "user", "content": example["query"]},
        ],
        "tools": example["tools"],
        "query": example["query"],
        "answers": example["answers"],
    }


def get_clean_eval(token: str) -> tuple[Dataset, list[dict]]:
    """Retourne (held-out propre, exemples few-shot tirés du TRAIN).

    Les shots viennent du split train : le held-out est déjà filtré de toute
    query présente dans le train, donc ils ne peuvent pas fuiter dans l'éval.
    """
    raw = load_dataset("Salesforce/xlam-function-calling-60k", token=token)["train"]
    converted = raw.map(to_messages, remove_columns=raw.column_names)
    split = converted.train_test_split(test_size=TEST_SIZE, seed=SEED)

    train_queries = frozenset(split["train"]["query"])
    eval_ds = split["test"].select(range(EVAL_SKIP, len(split["test"])))

    # Anti-contamination
    clean = eval_ds.filter(lambda x: x["query"] not in train_queries)
    # On prend les N_EVAL premiers pour la Phase 2
    clean = clean.select(range(min(N_EVAL, len(clean))))

    # Shots : on privilégie des golds courts (1 ou 2 appels) pour ne pas
    # gonfler le prompt, et on fige la sélection avec SEED.
    shots_pool = split["train"].shuffle(seed=SEED).select(range(200))
    shots = []
    for ex in shots_pool:
        if len(ex["answers"]) > 220:
            continue
        shots.append({"query": ex["query"], "answers": ex["answers"]})
        if len(shots) == N_SHOTS:
            break
    return clean, shots


def build_fewshot_block(shots: list[dict]) -> str:
    """Bloc d'exemples : on montre le FORMAT, pas la sélection d'outil.

    On n'y répète pas la liste d'outils de chaque exemple : ça tripleraient la
    longueur du prompt pour rien, alors que ce qu'on veut enseigner ici c'est
    la forme de la sortie.
    """
    if not shots:
        return ""
    lines = ["Examples of the expected output format:", ""]
    for s in shots:
        lines.append(f"Query: {s['query']}")
        lines.append(f"Output: {s['answers']}")
        lines.append("")
    return "\n".join(lines)


def build_prompt(tokenizer, ex: dict, mode: str, fewshot_block: str) -> str:
    """Rend le prompt texte final pour un exemple, selon le mode."""
    if mode == "plain":
        return tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=True
        )

    system = SYSTEM_PROMPT_FEWSHOT.format(tools=ex["tools"], examples=fewshot_block)

    if mode == "fewshot":
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": ex["query"]},
        ]
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    if mode == "raw_fewshot":
        # Pas de chat template : un modèle base n'en a jamais vu. Le few-shot
        # est de la complétion de motif, ce qu'il sait faire nativement.
        return f"{system}\nQuery: {ex['query']}\nOutput:"

    raise ValueError(f"mode de prompt inconnu : {mode}")


def cut_at_stop(text: str) -> str:
    """Coupe la complétion brute au premier marqueur de fin."""
    cut = len(text)
    for marker in RAW_STOP_MARKERS:
        i = text.find(marker)
        if i != -1:
            cut = min(cut, i)
    return text[:cut].strip()


# ---------------------------------------------------------------------------
# Normalisation & scoring (7 niveaux)
# ---------------------------------------------------------------------------
def normalize_value(v: Any) -> Any:
    """Typage tolérant ("5" == 5) + retour toujours hashable.

    Les conteneurs deviennent des tuples : Counter() exige des éléments
    hashables, et un tuple ne l'est que si tout son contenu l'est.
    Choix assumé : l'ordre des listes est significatif (tuple, pas frozenset).
    """
    if isinstance(v, str):
        s = v.strip()
        low = s.lower()
        if low in {"true", "false"}:
            return low == "true"
        # Un zéro initial est porteur de sens (code postal, indicatif, ID) :
        # on ne le convertit pas, sinon "07300" == 7300 => faux positif.
        if s and not (len(s) > 1 and s[0] == "0" and s[1] != "."):
            try:
                if "." in s:
                    return float(s)
                return int(s)
            except ValueError:
                pass
        return v
    if isinstance(v, (bool, int, float)):
        return v
    if isinstance(v, list):
        return tuple(normalize_value(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, normalize_value(val)) for k, val in v.items()))
    return v


def normalize_call(call: dict) -> tuple[str, tuple]:
    """Retourne (name, tuple d'items normalisés) pour comparaison multiset."""
    name = call.get("name") or ""
    args = call.get("arguments", {})
    if not isinstance(args, dict):
        args = {}
    # tri + normalisation
    items = tuple(sorted((k, normalize_value(v)) for k, v in args.items()))
    return name, items


def parse_json_safe(text: str) -> Any | None:
    text = text.strip()
    # Parfois le modèle met un markdown fence
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def score_one(
    pred_text: str, gold_text: str, available_tools: list[str]
) -> dict[str, bool]:
    """
    Retourne un dict de 6 booléens (niveaux 1→6), cumulatifs.

    Le niveau 6 (arguments exacts) EST l'exact match : y parvenir suppose déjà
    le bon schéma, les bons outils, la bonne cardinalité et le bon multiset de
    noms. Une ancienne version exposait un niveau 7 séparé, strictement égal au
    6 sur les 12 580 lignes évaluées — il a été fusionné.
    """
    res = {f"level_{i}": False for i in range(1, 7)}

    # 1. JSON parsable
    pred = parse_json_safe(pred_text)
    gold = parse_json_safe(gold_text)
    if pred is None or gold is None:
        return res
    res["level_1"] = True

    # 2. Schéma [{name, arguments}]
    def is_valid_schema(obj):
        if not isinstance(obj, list):
            return False
        for item in obj:
            if not isinstance(item, dict):
                return False
            if "name" not in item or "arguments" not in item:
                return False
        return True

    if not (is_valid_schema(pred) and is_valid_schema(gold)):
        return res
    res["level_2"] = True

    # 3. Aucun outil halluciné
    tool_names = set(available_tools) if available_tools else set()
    pred_names = [c.get("name") for c in pred]
    if not all(n in tool_names for n in pred_names):
        return res
    res["level_3"] = True

    # 4. Bonne cardinalité
    if len(pred) != len(gold):
        return res
    res["level_4"] = True

    # 5. Multiset des names exact
    if Counter(pred_names) != Counter(c.get("name") for c in gold):
        return res
    res["level_5"] = True

    # 6. Arguments exacts = exact match complet
    pred_norm = Counter(normalize_call(c) for c in pred)
    gold_norm = Counter(normalize_call(c) for c in gold)
    if pred_norm != gold_norm:
        return res
    res["level_6"] = True
    return res


# ---------------------------------------------------------------------------
# Génération
# ---------------------------------------------------------------------------
def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    device: str,
) -> tuple[list[str], list[dict]]:
    """Retourne (textes générés, métadonnées par exemple).

    La métadonnée `truncated` distingue deux échecs que le score confond :
    une génération qui a déraillé sans jamais émettre d'EOS, et une réponse
    légitimement longue qu'on a coupée. Décision prise lors de la calibration
    de MAX_NEW_TOKENS (max réel observé 410, plafond 512), perdue lors d'une
    réécriture du script — rétablie ici.
    """
    # Troncature à droite = on couperait la query elle-même, en silence.
    # On compte les cas plutôt que de les découvrir dans les scores.
    n_trunc = sum(
        1 for p in prompts if len(tokenizer(p)["input_ids"]) > MAX_INPUT_LEN
    )
    if n_trunc:
        print(f"  !! {n_trunc}/{len(prompts)} prompts tronqués à {MAX_INPUT_LEN} tokens")

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_LEN,
    ).to(device)

    gen_config = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,  # greedy pour reproductibilité
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=gen_config)

    # On ne garde que les nouveaux tokens
    eos_id = tokenizer.eos_token_id
    generated, metas = [], []
    for i, out in enumerate(outputs):
        input_len = inputs["input_ids"][i].shape[0]
        gen_ids = out[input_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # Une séquence terminée contient son EOS (les rangs suivants sont du
        # padding). Une séquence qui a atteint le plafond n'en contient aucun.
        hits = (gen_ids == eos_id).nonzero()
        if len(hits):
            n_gen, truncated = int(hits[0][0]) + 1, False
        else:
            n_gen, truncated = int(gen_ids.shape[0]), True

        generated.append(text)
        metas.append({"n_gen_tokens": n_gen, "truncated": truncated})
    return generated, metas


# ---------------------------------------------------------------------------
# Baseline triviale
# ---------------------------------------------------------------------------
def baseline_trivial(tools_str: str) -> str:
    """Toujours le premier outil, arguments vides."""
    try:
        tools = json.loads(tools_str) if isinstance(tools_str, str) else tools_str
        if tools and isinstance(tools, list) and len(tools) > 0:
            name = tools[0].get("name", "unknown")
            return json.dumps([{"name": name, "arguments": {}}], ensure_ascii=False)
    except (json.JSONDecodeError, TypeError, AttributeError, KeyError) as exc:
        logger.warning(
            "Impossible de lire les outils pour le baseline trivial: %s", exc
        )
    return "[]"


# ---------------------------------------------------------------------------
# Chargement des modèles
# ---------------------------------------------------------------------------
def is_adapter_repo(model_name: str, revision: str | None, token: str) -> bool:
    return HfApi().file_exists(
        model_name, "adapter_config.json", revision=revision, token=token
    )


def load_model(model_key: str, model_name: str, revision: str | None, token: str):
    """Modèle complet, ou base + adaptateur PEFT si le repo est un adaptateur.

    On charge le base et l'adaptateur SÉPARÉMENT plutôt que de laisser
    AutoModelForCausalLM résoudre le repo adaptateur : `revision` doit
    s'appliquer au repo de l'adaptateur, pas au base (où ce SHA n'existe pas).
    """
    if not is_adapter_repo(model_name, revision, token):
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            dtype=DTYPE,
            device_map="auto",
            token=token,
        )

    if revision is None:
        raise ValueError(
            f"[{model_key}] adaptateur sans révision figée : renseigne le SHA du "
            f"commit 'End of training' de {model_name} avant de publier un score."
        )

    cfg_path = hf_hub_download(
        model_name, "adapter_config.json", revision=revision, token=token
    )
    with open(cfg_path) as f:
        adapter_cfg = json.load(f)
    base_name = adapter_cfg["base_model_name_or_path"]
    quantized = model_key in EVAL_4BIT_BASE
    print(f"  [{model_key}] base={base_name} | 4 bits={quantized} | adaptateur@{revision[:7]}")

    base = AutoModelForCausalLM.from_pretrained(
        base_name,
        dtype=DTYPE,
        device_map="auto",
        quantization_config=BNB_4BIT if quantized else None,
        token=token,
    )
    model = PeftModel.from_pretrained(base, model_name, revision=revision, token=token)

    if quantized:
        # Pas de fusion : fusionner dans des poids 4 bits requantifie le
        # résultat (perte), et c'est justement la condition d'entraînement
        # qu'on veut mesurer. On génère à travers l'adaptateur, plus lent.
        return model
    # LoRA sur base bf16 : fusion W + BA en bf16. Même modèle mathématique,
    # sans le surcoût de l'adaptateur à chaque token généré.
    return model.merge_and_unload()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    token = os.environ["HF_TOKEN"]
    login(token=token)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    eval_ds, shots = get_clean_eval(token)
    fewshot_block = build_fewshot_block(shots)
    print(f"Few-shot : {len(shots)} exemples tirés du train ({len(fewshot_block)} chars)")
    print(f"Éval clean size: {len(eval_ds)}")

    # On prépare les tools disponibles (liste de noms)
    def extract_tool_names(tools_field):
        try:
            tools = (
                json.loads(tools_field) if isinstance(tools_field, str) else tools_field
            )
            return [t["name"] for t in tools]
        except (json.JSONDecodeError, TypeError, KeyError):
            return []

    all_results = []
    metrics = {}

    to_run = [m for m in MODELS_TO_EVAL if RUN_ONLY is None or m[0] in RUN_ONLY]
    run_keys = [m[0] for m in to_run]
    if RUN_ONLY is not None and "baseline" in RUN_ONLY:
        run_keys.append("baseline")
    print(f"Conditions évaluées dans ce run : {run_keys}")

    for model_key, model_name, revision, prompt_mode in to_run:
        print(f"\n===== Evaluating {model_key} ({model_name}, mode={prompt_mode}) =====")
        tok = AutoTokenizer.from_pretrained(model_name, revision=revision, token=token)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        # Decoder-only : la génération part du dernier token. En right-padding
        # ce dernier token est <pad> pour tout prompt plus court que le batch.
        tok.padding_side = "left"
        if tok.chat_template is None:
            print(f"  [{model_key}] pas de chat_template -> emprunt à {TEMPLATE_SOURCE}")
            tok.chat_template = AutoTokenizer.from_pretrained(
                TEMPLATE_SOURCE, token=token
            ).chat_template

        model = load_model(model_key, model_name, revision, token)
        model.eval()

        preds, metas = [], []
        # 24 Go de VRAM pour 270 Mo de poids : à 8, le GPU attend plus qu'il
        # ne calcule. Gain sous-linéaire (un batch tourne jusqu'à ce que sa
        # séquence la plus longue finisse), mais net.
        # Gardé à 32 pour le 1.7B aussi : un autre batch_size déplace le score
        # de ~1 pt. Coût : KV cache jusqu'à ~13 Go (32 x ~2100 tokens x 197 Ko)
        # + 3,4 Go de poids -> lancer sur a100-large, pas a10g.
        batch_size = 32
        for i in range(0, len(eval_ds), batch_size):
            batch = eval_ds.select(range(i, min(i + batch_size, len(eval_ds))))
            prompts = [
                build_prompt(tok, ex, prompt_mode, fewshot_block) for ex in batch
            ]
            batch_preds, batch_meta = generate_batch(model, tok, prompts, device)
            if prompt_mode == "raw_fewshot":
                batch_preds = [cut_at_stop(t) for t in batch_preds]
            preds.extend(batch_preds)
            metas.extend(batch_meta)
            if (i // batch_size) % 10 == 0:
                print(f"  generated {i + len(batch_preds)}/{len(eval_ds)}")

        # Scoring
        level_counts = {f"level_{i}": 0 for i in range(1, 7)}
        detailed = []

        for idx, (ex, pred_text, meta) in enumerate(zip(eval_ds, preds, metas)):
            tools_names = extract_tool_names(ex["tools"])
            scores = score_one(pred_text, ex["answers"], tools_names)
            for k, v in scores.items():
                if v:
                    level_counts[k] += 1

            detailed.append(
                {
                    "model": model_key,
                    "prompt_mode": prompt_mode,
                    "query": ex["query"],
                    "tools": ex["tools"],
                    "gold": ex["answers"],
                    "pred": pred_text,
                    **meta,
                    **scores,
                }
            )

        n = len(eval_ds)
        model_metrics = {k: round(100.0 * v / n, 2) for k, v in level_counts.items()}
        metrics[model_key] = model_metrics
        all_results.extend(detailed)

        n_trunc = sum(m["truncated"] for m in metas)
        print(f"  générations tronquées à {MAX_NEW_TOKENS} tokens : {n_trunc}/{len(metas)}")
        print(f"Metrics {model_key}:")
        for k, v in model_metrics.items():
            print(f"  {k}: {v}%")

        # libère la mémoire
        del model
        torch.cuda.empty_cache()

    # ----- Baseline triviale (gratuite, mais on la refait seulement si demandée) -----
    if "baseline" in run_keys:
        print("\n===== Baseline triviale =====")
        run_baseline(eval_ds, extract_tool_names, all_results, metrics)

    # ----- Fusion avec les runs précédents + push -----
    all_results = merge_with_previous(all_results, run_keys, token)
    push_results(all_results, metrics, token)


def run_baseline(eval_ds, extract_tool_names, all_results, metrics):
    level_counts = {f"level_{i}": 0 for i in range(1, 7)}
    for ex in eval_ds:
        pred_text = baseline_trivial(ex["tools"])
        tools_names = extract_tool_names(ex["tools"])
        scores = score_one(pred_text, ex["answers"], tools_names)
        for k, v in scores.items():
            if v:
                level_counts[k] += 1
        all_results.append(
            {
                "model": "baseline",
                "prompt_mode": "n/a",
                "n_gen_tokens": 0,
                "truncated": False,
                "query": ex["query"],
                "tools": ex["tools"],
                "gold": ex["answers"],
                "pred": pred_text,
                **scores,
            }
        )
    n = len(eval_ds)
    metrics["baseline"] = {k: round(100.0 * v / n, 2) for k, v in level_counts.items()}
    print(metrics["baseline"])


def merge_with_previous(new_rows: list[dict], run_keys: list[str], token: str) -> list[dict]:
    """Conserve les conditions non rejouées depuis le dataset existant.

    push_to_hub écrase le split : sans ça, un run partiel effacerait les
    résultats des conditions qu'on vient de ne PAS réévaluer.
    """
    api = HfApi(token=token)
    try:
        api.dataset_info(HUB_DATASET_ID)
    except Exception:
        print("  dataset inexistant — premier run, rien à conserver")
        return new_rows

    # On lit le parquet directement plutôt que par load_dataset : ce dernier
    # valide contre le bloc `dataset_info` du README, qui se désynchronise dès
    # qu'on ajoute une colonne. Une carte périmée ne doit pas empêcher de
    # relire des données parfaitement saines.
    try:
        path = hf_hub_download(
            HUB_DATASET_ID,
            "data/train-00000-of-00001.parquet",
            repo_type="dataset",
            token=token,
        )
        old = pd.read_parquet(path).to_dict("records")
    except Exception as e:
        # Le dataset existe mais on n'a pas su le lire : pousser maintenant
        # effacerait toutes les conditions non rejouées. On s'arrête net.
        raise RuntimeError(
            f"{HUB_DATASET_ID} existe mais est illisible "
            f"({type(e).__name__}: {e}). Pousser écraserait les conditions "
            f"non rejouées ({sorted(set(run_keys))} seulement seraient gardées). "
            "Vérifie le repo, puis relance."
        ) from e

    kept = [dict(r) for r in old if r["model"] not in run_keys]
    for r in kept:
        r.setdefault("prompt_mode", "plain")  # runs antérieurs à cette colonne
        r.setdefault("n_gen_tokens", -1)  # -1 = non mesuré (run antérieur)
        r.setdefault("truncated", False)
        # level_7 était un doublon strict de level_6 (vérifié sur 12 580
        # lignes). Les snapshots antérieurs à la fusion le portent encore.
        r.pop("level_7", None)
    anciens = sorted({r["model"] for r in kept})
    print(f"  {len(kept)} lignes conservées des runs précédents : {anciens}")
    return kept + new_rows


def check_card_schema(results_ds: Dataset, token: str) -> None:
    """Vérifie que la carte du Hub déclare bien les colonnes du parquet.

    Quand on ajoute une colonne à un dataset déjà publié, push_to_hub rafraîchit
    les compteurs mais pas toujours le bloc `features`. Le parquet reste sain,
    mais load_dataset et le viewer lèvent un CastError. On le signale ici plutôt
    que de le découvrir sur la page trois jours plus tard.
    """
    try:
        card = DatasetCard.load(HUB_DATASET_ID, token=token)
        info = card.data.to_dict().get("dataset_info") or {}
        if isinstance(info, list):
            info = info[0] if info else {}
        declarees = [f["name"] for f in info.get("features", [])]
    except Exception as e:
        print(f"  (carte illisible : {type(e).__name__}, vérification ignorée)")
        return

    reelles = list(results_ds.column_names)
    if not declarees:
        print("  (carte sans bloc features, vérification ignorée)")
        return
    if set(declarees) == set(reelles):
        print("  carte cohérente avec le parquet")
        return

    print("  !! CARTE DÉSYNCHRONISÉE — le viewer et load_dataset vont échouer")
    manquantes = sorted(set(reelles) - set(declarees))
    en_trop = sorted(set(declarees) - set(reelles))
    if manquantes:
        print(f"     absentes de la carte      : {manquantes}")
    if en_trop:
        print(f"     déclarées mais inexistantes : {en_trop}")
    print(f"     → corrige le bloc dataset_info du README de {HUB_DATASET_ID}")


def make_card_pushable(token: str) -> None:
    """Retire un bloc `dataset_info` incomplet, qui ferait planter le push.

    datasets >= 4 met la carte à jour en faisant
    `repo_info.download_size -= deleted_size`. Si le bloc dataset_info du
    README a été écrit à la main sans `download_size` (c'est le cas ici :
    features + num_examples seulement), la soustraction lève
    `TypeError: NoneType -= int`. Le piège : ça arrive APRÈS l'upload des
    données mais AVANT le commit — 1 h 45 de GPU perdue, rien sur le Hub.
    Sans le bloc, push_to_hub le régénère complet.
    """
    try:
        card = DatasetCard.load(HUB_DATASET_ID, token=token)
    except Exception as e:
        print(f"  (carte illisible : {type(e).__name__}, vérification ignorée)")
        return
    info = card.data.to_dict().get("dataset_info")
    if isinstance(info, list):
        info = info[0] if info else None
    if not isinstance(info, dict) or info.get("download_size") is not None:
        return
    print("  carte sans download_size -> retrait du bloc dataset_info (régénéré au push)")
    card.data.dataset_info = None
    card.push_to_hub(HUB_DATASET_ID, repo_type="dataset", token=token)


def push_parquet_directly(results_ds: Dataset, token: str) -> None:
    """Repli : écrit le parquet à l'emplacement que merge_with_previous relit.

    Même chemin que celui produit par push_to_hub, donc le prochain run
    retrouve ses lignes. La carte reste périmée (check_card_schema le dira),
    mais les données, elles, sont sauvées.
    """
    local = "results.parquet"
    results_ds.to_parquet(local)
    HfApi(token=token).upload_file(
        path_or_fileobj=local,
        path_in_repo="data/train-00000-of-00001.parquet",
        repo_id=HUB_DATASET_ID,
        repo_type="dataset",
        commit_message="Résultats (repli : upload direct du parquet)",
        token=token,
    )
    print("  repli réussi : données poussées, carte à rafraîchir")


def push_results(all_results: list[dict], metrics: dict, token: str) -> None:
    print("\nPush des résultats sur le Hub...")
    results_ds = Dataset.from_list(all_results)
    make_card_pushable(token)
    try:
        results_ds.push_to_hub(HUB_DATASET_ID, private=RESULTS_REPO_PRIVATE, token=token)
    except Exception as e:
        # Ne jamais laisser un échec de mise à jour de carte emporter le run :
        # la génération coûte des heures de GPU, l'upload coûte 3 Mo.
        print(f"  !! push_to_hub a échoué ({type(e).__name__}: {e})")
        push_parquet_directly(results_ds, token)
    check_card_schema(results_ds, token)

    # Métriques agrégées
    api = HfApi(token=token)
    metrics_path = "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    api.upload_file(
        path_or_fileobj=metrics_path,
        path_in_repo="metrics.json",
        repo_id=HUB_DATASET_ID,
        repo_type="dataset",
        token=token,
    )

    print("\n===== RÉSUMÉ FINAL (conditions de ce run) =====")
    for model_key, m in metrics.items():
        print(f"{model_key:18s}  exact-match (L6) = {m['level_6']}%")
    print(f"\nRésultats détaillés → https://huggingface.co/datasets/{HUB_DATASET_ID}")
    print("Done.")


def exit_clean() -> None:
    """Sort explicitement du processus.

    Les runs précédents affichaient "Done." puis se terminaient en
    "Job timeout" : le script allait au bout mais l'interpréteur ne rendait
    pas la main, donc le job était facturé jusqu'au timeout. Un thread
    non-daemon encore vivant empêche Python de sortir — on l'identifie avant
    de forcer la sortie, pour savoir quoi corriger à la source.
    """
    restants = [
        t.name
        for t in threading.enumerate()
        if t is not threading.main_thread() and not t.daemon
    ]
    if restants:
        print(f"threads non-daemon encore actifs : {restants}")
    else:
        print("aucun thread bloquant — sortie immédiate")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
    exit_clean()
