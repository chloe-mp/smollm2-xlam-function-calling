# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets", "pandas", "pyarrow"]
# ///
"""Prépare l'audit publiable des références non dérivables de xLAM-60k.

Entrée : judge_results.parquet (4 790 verdicts = 2 395 exemples x 2 modèles).
Sortie : audit_xlam/a_verifier.csv, une ligne par EXEMPLE signalé, avec :
  - xlam_id        : identifiant d'origine dans Salesforce/xlam-function-calling-60k
                     (l'`idx` du juge n'est que la position dans notre held-out)
  - signale_par    : 2 = les deux verdicts du juge (LoRA et QLoRA) disent "non dérivable",
                     1 = un seul -> le juge se contredit sur une propriété de la référence
  - a_verifier     : "ambigu" (tous les signale_par=1), "echantillon" (tirage fixe parmi
                     les signale_par=2), "" sinon
  - verdict_humain : vide, à remplir ("non_derivable" / "derivable" / "incertain")

Pourquoi joindre sur (query, answers) : to_messages() supprime la colonne `id`
(remove_columns), et une même query peut apparaître plusieurs fois dans xLAM
(ce sont justement les 93 fuites train/test). Query + answers identifient la ligne.
"""

import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset

SEED = 42
N_ECHANTILLON = 50  # parmi les exemples signalés par les deux verdicts
SORTIE = Path("audit_xlam")


def main():
    juge = pd.read_parquet("judge_results.parquet")
    signales = juge[juge.reference_derivable == "no"]

    # Une ligne par exemple : combien de verdicts le signalent, et les raisons de chacun.
    par_exemple = (
        signales.groupby("idx")
        .agg(signale_par=("model", "nunique"), modeles=("model", lambda s: ",".join(sorted(s))))
        .reset_index()
    )
    raisons = juge[juge.idx.isin(par_exemple.idx)].pivot_table(
        index="idx", columns="model", values="reason", aggfunc="first"
    )
    raisons.columns = [f"raison_{c}" for c in raisons.columns]
    derivable = juge[juge.idx.isin(par_exemple.idx)].pivot_table(
        index="idx", columns="model", values="reference_derivable", aggfunc="first"
    )
    derivable.columns = [f"derivable_{c}" for c in derivable.columns]
    contenu = juge[juge.model == "lora-1.7b"].set_index("idx")[["query", "gold", "tools"]]
    audit = par_exemple.join(contenu, on="idx").join(raisons, on="idx").join(derivable, on="idx")

    # --- Retrouver l'id d'origine xLAM -------------------------------------------
    xlam = load_dataset("Salesforce/xlam-function-calling-60k")["train"].to_pandas()
    xlam["cle"] = xlam["query"] + "\x00" + xlam["answers"]
    candidats = xlam.groupby("cle")["id"].apply(list)
    audit["cle"] = audit["query"] + "\x00" + audit["gold"]
    audit["xlam_ids"] = audit["cle"].map(candidats)
    sans_id = audit.xlam_ids.isna().sum()
    multiples = (audit.xlam_ids.dropna().map(len) > 1).sum()
    audit["xlam_id"] = audit.xlam_ids.map(lambda ids: ids[0] if isinstance(ids, list) and len(ids) == 1 else None)

    # --- Ce qu'il faut vérifier à la main ------------------------------------------
    audit["a_verifier"] = ""
    audit.loc[audit.signale_par == 1, "a_verifier"] = "ambigu"
    deux = audit[audit.signale_par == 2]
    tirage = deux.sample(n=min(N_ECHANTILLON, len(deux)), random_state=SEED).index
    audit.loc[tirage, "a_verifier"] = "echantillon"
    audit["verdict_humain"] = ""
    audit["commentaire"] = ""

    colonnes = ["xlam_id", "idx", "signale_par", "modeles", "a_verifier", "verdict_humain", "commentaire",
                "query", "gold", "tools"] + [c for c in audit.columns if c.startswith(("raison_", "derivable_"))]
    audit = audit.sort_values(["a_verifier", "signale_par", "idx"], ascending=[False, True, True])[colonnes]
    audit = audit.rename(columns={"idx": "eval_idx"})

    SORTIE.mkdir(exist_ok=True)
    audit.to_csv(SORTIE / "a_verifier.csv", index=False)
    resume = {
        "exemples_signales": int(len(audit)),
        "signales_par_les_deux_verdicts": int((audit.signale_par == 2).sum()),
        "signales_par_un_seul": int((audit.signale_par == 1).sum()),
        "a_verifier_ambigus": int((audit.a_verifier == "ambigu").sum()),
        "a_verifier_echantillon": int((audit.a_verifier == "echantillon").sum()),
        "sans_id_xlam": int(sans_id),
        "plusieurs_id_xlam": int(multiples),
        "echecs_signales_cote_predictions": int(
            ((juge.reference_derivable == "no") & ~juge.level_6.astype(bool)).sum()
        ),
        "taille_held_out": int(juge.idx.nunique()),
    }
    (SORTIE / "resume.json").write_text(json.dumps(resume, indent=2, ensure_ascii=False))
    print(json.dumps(resume, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
