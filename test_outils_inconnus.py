# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "transformers>=4.46", "peft", "accelerate", "pandas", "pyarrow"]
# ///
"""Test rapide en local (Mac, MPS) : le LoRA 1.7B généralise-t-il à des outils jamais vus ?

Outils inconnus = les 4 outils du serveur MCP mcp-veille-nlp (absents de xLAM),
décrits au format xLAM. 20 requêtes écrites à la main avec leur gold.
Contrôle : 30 exemples du held-out xLAM déjà jugés, pour vérifier que le
setup local reproduit les prédictions enregistrées.

Comparé à SmolLM2-1.7B-Instruct avec un prompt qui précise le format de sortie.
Exploratoire : 20 requêtes rédigées par une seule personne, pas un benchmark.
"""

import json
import sys
import time

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_LORA = "Chloemp/smollm2-1.7b-xlam-lora"
REVISION_LORA = "8868bcd45fb4942552222602f539148799d6e593"
INSTRUCT = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.bfloat16

SYSTEM_PROMPT = "You are a function-calling assistant. Available tools:\n{tools}"
SYSTEM_PROMPT_FORMAT = (
    "You are a function-calling assistant. Available tools:\n{tools}\n\n"
    "Reply with a JSON array of tool calls and nothing else.\n"
    'Format: [{{"name": "<tool_name>", "arguments": {{"<arg>": "<value>"}}}}]\n'
    "Use only the tools listed above. One object per required call. "
    "No explanation, no markdown fences, no code."
)

# --- Outils du serveur MCP, au format xLAM ------------------------------------
OUTILS = [
    {"name": "get_trending_hf",
     "description": "Get the currently trending repositories on the Hugging Face Hub.",
     "parameters": {
         "repo_type": {"description": "Type of repository: 'model', 'dataset' or 'space'.", "type": "str", "default": "model"},
         "limit": {"description": "Number of results to return.", "type": "int", "default": 10}}},
    {"name": "search_trending_repos",
     "description": "Search popular GitHub repositories on a topic, created recently, sorted by stars.",
     "parameters": {
         "topic": {"description": "GitHub topic tag, e.g. 'nlp', 'llm', 'large-language-models'.", "type": "str", "default": "llm"},
         "since_days": {"description": "Only keep repositories created within this many days.", "type": "int", "default": 30},
         "limit": {"description": "Number of results.", "type": "int", "default": 10}}},
    {"name": "search_recent_papers",
     "description": "Search the most recent arXiv papers in a category matching keywords.",
     "parameters": {
         "cat": {"description": "arXiv category, e.g. 'cs.CL' (Computation and Language).", "type": "str", "default": "cs.CL"},
         "keyword": {"description": "List of keywords to match in abstracts, e.g. ['LLM', 'RAG'].", "type": "List[str]", "default": ""},
         "since_days": {"description": "Only keep papers published within this many days.", "type": "int", "default": 7},
         "limit": {"description": "Number of results.", "type": "int", "default": 10}}},
    {"name": "get_release_notes",
     "description": "Get the latest release notes of a GitHub repository.",
     "parameters": {
         "repo": {"description": "Repository in 'owner/repo' format, e.g. 'huggingface/trl'.", "type": "str"},
         "limit": {"description": "Number of recent releases to return, newest first.", "type": "int", "default": 5}}},
]
DEFAUTS = {o["name"]: {k: v.get("default") for k, v in o["parameters"].items() if "default" in v} for o in OUTILS}
TOOLS_STR = json.dumps(OUTILS)


def c(name, **args):
    return {"name": name, "arguments": args}


# 20 requêtes : 15 à un appel, 5 à plusieurs appels.
REQUETES = [
    ("What are the 5 trending models on Hugging Face right now?", [c("get_trending_hf", limit=5)]),
    ("Show me the top 20 trending Hugging Face models.", [c("get_trending_hf", limit=20)]),
    ("Get the latest 3 releases of huggingface/transformers.", [c("get_release_notes", repo="huggingface/transformers", limit=3)]),
    ("What changed in the most recent release of vllm-project/vllm? Just the last one.", [c("get_release_notes", repo="vllm-project/vllm", limit=1)]),
    ("Find recent arXiv papers in cs.CL about RAG from the last 14 days.", [c("search_recent_papers", cat="cs.CL", keyword=["RAG"], since_days=14)]),
    ("List 5 new papers on quantization and distillation in cs.LG from the past 30 days.", [c("search_recent_papers", cat="cs.LG", keyword=["quantization", "distillation"], since_days=30, limit=5)]),
    ("Which GitHub repositories tagged 'rag' created in the last 7 days have the most stars?", [c("search_trending_repos", topic="rag", since_days=7)]),
    ("Give me the 15 most starred new repos on the topic 'agents' from the past 60 days.", [c("search_trending_repos", topic="agents", since_days=60, limit=15)]),
    ("Show release notes for huggingface/trl.", [c("get_release_notes", repo="huggingface/trl")]),
    ("What are the trending datasets on the Hub? Give me 10.", [c("get_trending_hf", repo_type="dataset", limit=10)]),
    ("Latest 10 releases of pytorch/pytorch.", [c("get_release_notes", repo="pytorch/pytorch", limit=10)]),
    ("Any new VLM papers in cs.CV in the last 7 days?", [c("search_recent_papers", cat="cs.CV", keyword=["VLM"], since_days=7)]),
    ("Top 3 new GitHub repos about 'nlp' from the last 14 days.", [c("search_trending_repos", topic="nlp", since_days=14, limit=3)]),
    ("Get 8 recent cs.CL papers mentioning SLM or RAG from the last 10 days.", [c("search_recent_papers", cat="cs.CL", keyword=["SLM", "RAG"], since_days=10, limit=8)]),
    ("Show me 25 trending GitHub repositories on large-language-models created in the last 90 days.", [c("search_trending_repos", topic="large-language-models", since_days=90, limit=25)]),
    ("Get the 5 trending Hugging Face models and the last 2 releases of langchain-ai/langgraph.",
     [c("get_trending_hf", limit=5), c("get_release_notes", repo="langchain-ai/langgraph", limit=2)]),
    ("Find cs.CL papers about agents from the last 3 days, and the new GitHub repos with topic 'llm' created in the past 3 days.",
     [c("search_recent_papers", cat="cs.CL", keyword=["agents"], since_days=3), c("search_trending_repos", topic="llm", since_days=3)]),
    ("Show just the most recent release of huggingface/peft and of huggingface/accelerate.",
     [c("get_release_notes", repo="huggingface/peft", limit=1), c("get_release_notes", repo="huggingface/accelerate", limit=1)]),
    ("Get the 10 trending spaces on Hugging Face and the 10 trending datasets.",
     [c("get_trending_hf", repo_type="space", limit=10), c("get_trending_hf", repo_type="dataset", limit=10)]),
    ("Weekly digest: 5 trending HF models, 5 new cs.CL papers about RAG from the last 7 days, and 5 new GitHub repos on 'llm' from the last 7 days.",
     [c("get_trending_hf", limit=5), c("search_recent_papers", cat="cs.CL", keyword=["RAG"], since_days=7, limit=5),
      c("search_trending_repos", topic="llm", since_days=7, limit=5)]),
]


# --- Scoring ------------------------------------------------------------------
def norm(v):
    if isinstance(v, str) and v.strip().lstrip("-").isdigit():
        return int(v)
    if isinstance(v, str) and "," in v:          # "RAG, LLM" ~ ["RAG", "LLM"]
        return sorted(x.strip() for x in v.split(","))
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, list):
        return sorted(norm(x) if not isinstance(x, str) else x.strip() for x in v)
    return v


def call_key(call, lenient):
    name, args = call.get("name"), dict(call.get("arguments") or {})
    if lenient:  # un argument explicite égal à sa valeur par défaut = équivalent à son omission
        for k, d in DEFAUTS.get(name, {}).items():
            if k in args and norm(args[k]) == norm(d):
                del args[k]
    if "keyword" in args and isinstance(args["keyword"], str):
        args["keyword"] = [args["keyword"]]
    return json.dumps([name, {k: norm(v) for k, v in sorted(args.items())}], sort_keys=True, default=str)


def score(pred_text, gold, lenient=True):
    try:
        pred = json.loads(pred_text)
        if isinstance(pred, dict):
            pred = [pred]
        assert isinstance(pred, list) and all(isinstance(p, dict) for p in pred)
    except Exception:
        return {"json": False, "noms": False, "exact": False}
    noms_ok = sorted(p.get("name", "") for p in pred) == sorted(g["name"] for g in gold)
    exact = sorted(call_key(p, lenient) for p in pred) == sorted(call_key(g, lenient) for g in gold)
    return {"json": True, "noms": noms_ok, "exact": exact}


# --- Modèles ------------------------------------------------------------------
def charger(nom, repo=MODEL_LORA, revision=REVISION_LORA):
    if nom != "instruct-1.7b":
        tok = AutoTokenizer.from_pretrained(repo, revision=revision)
        if tok.chat_template is None:
            tok.chat_template = AutoTokenizer.from_pretrained(INSTRUCT).chat_template
        cfg = json.load(open(hf_hub_download(repo, "adapter_config.json", revision=revision)))
        base = AutoModelForCausalLM.from_pretrained(cfg["base_model_name_or_path"], dtype=DTYPE).to(DEVICE)
        model = PeftModel.from_pretrained(base, repo, revision=revision).merge_and_unload()
        return tok, model, SYSTEM_PROMPT
    tok = AutoTokenizer.from_pretrained(INSTRUCT)
    model = AutoModelForCausalLM.from_pretrained(INSTRUCT, dtype=DTYPE).to(DEVICE)
    return tok, model, SYSTEM_PROMPT_FORMAT


@torch.no_grad()
def generer(tok, model, system, tools, query):
    msgs = [{"role": "system", "content": system.format(tools=tools)}, {"role": "user", "content": query}]
    # Comme l'éval : on rend le texte du template, puis on tokenise sans re-ajouter de tokens spéciaux.
    texte = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(texte, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
    # eos explicite, comme l'éval : sinon generation_config du base (<|endoftext|>) ne s'arrête pas à <|im_end|>.
    out = model.generate(ids, max_new_tokens=384, do_sample=False,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id, eos_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    import argparse
    p = argparse.ArgumentParser()
    # Autre adaptateur à tester, ex. le LoRA parti de l'Instruct :
    #   uv run test_outils_inconnus.py --adapter Chloemp/smollm2-1.7b-instruct-xlam-lora \
    #     --revision <SHA "End of training"> --nom lora-instruct-1.7b
    p.add_argument("--adapter", default=MODEL_LORA)
    p.add_argument("--revision", default=REVISION_LORA)
    p.add_argument("--nom", default="lora-1.7b")
    p.add_argument("--sortie", default="test_outils_inconnus.parquet")
    a = p.parse_args()

    controle = pd.read_parquet("judge_results.parquet")
    controle = controle[controle.model == "lora-1.7b"].head(30)
    lignes = []
    for nom in [a.nom, "instruct-1.7b"]:
        print(f"\n===== {nom} =====", flush=True)
        t0 = time.time()
        tok, model, system = charger(nom, a.adapter, a.revision)
        model.eval()
        print(f"chargé en {time.time() - t0:.0f}s sur {DEVICE}", flush=True)
        for i, (q, gold) in enumerate(REQUETES):
            t = time.time()
            pred = generer(tok, model, system, TOOLS_STR, q)
            s, s_strict = score(pred, gold), score(pred, gold, lenient=False)
            lignes.append({"modele": nom, "jeu": "mcp_inconnus", "i": i, "multi": len(gold) > 1, "requete": q,
                           "gold": json.dumps(gold), "pred": pred, **s, "exact_strict": s_strict["exact"],
                           "sec": round(time.time() - t, 1)})
            print(f"  [{i:2}] exact={s['exact']!s:5} noms={s['noms']!s:5} {time.time() - t:4.1f}s  {pred[:110]}", flush=True)
        if nom != "instruct-1.7b":
            # Contrôle : pour le LoRA d'origine, reproduit l'éval A100 ; pour un autre
            # adaptateur, donne seulement un aperçu du score sur des outils connus.
            for _, r in controle.iterrows():
                pred = generer(tok, model, system, r.tools, r.query)
                lignes.append({"modele": nom, "jeu": "controle_xlam", "i": int(r.idx), "requete": r.query,
                               "gold": r.gold, "pred": pred, "identique_a_l_eval": pred == r.pred,
                               **score(pred, json.loads(r.gold), lenient=False)})
            ctl = pd.DataFrame([l for l in lignes if l["jeu"] == "controle_xlam"])
            print(f"  contrôle xLAM : {ctl.identique_a_l_eval.sum()}/30 prédictions identiques au LoRA base (éval A100), "
                  f"exact {ctl.exact.sum()}/30", flush=True)
        del model
        torch.mps.empty_cache() if DEVICE == "mps" else None

    df = pd.DataFrame(lignes)
    df.to_parquet(a.sortie)
    m = df[df.jeu == "mcp_inconnus"]
    print("\n===== Résumé (outils MCP jamais vus, 20 requêtes) =====")
    print(m.groupby("modele")[["json", "noms", "exact", "exact_strict"]].mean().mul(100).round(0).to_string())
    print(m.groupby(["modele", "multi"]).exact.mean().mul(100).round(0).to_string())


if __name__ == "__main__":
    sys.exit(main())
