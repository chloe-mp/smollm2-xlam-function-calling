# TODO — Évaluation de smollm2-135m-xlam-fullft

Modèle à évaluer : `Chloemp/smollm2-135m-xlam-fullft`
Révision finale : `050f71474648a88c470640c1b820aff9b8aa6113` ("End of training", step 3563)

> Règle : je code, Claude relit. Objectif = savoir tout refaire sans IA.

---

## Phase 0 — Nettoyage (5 min)

- [ ] `train_xlam_job.py` : corriger le commentaire de `hub_strategy="every_save"`
      → il pousse à CHAQUE save (10 commits sur le Hub le prouvent), pas une fois à la fin.
- [ ] `eval_xlam.py` : supprimer la variable `tools` (l.12-14), définie mais jamais utilisée.
- [ ] Renommer `eval_xlam.py` → `inspect_xlam.py`. C'est un outil d'inspection à 1 exemple,
      pas une éval. Le garder pour déboguer le format, ne jamais en tirer un chiffre.

## Phase 1 — L'éval minimale qui vaut quelque chose

### 1.1 Vérifier que le run est sain  ✅ FAIT (2026-08-19)
- [x] Courbes trackio consultées (run `fullft-2000`).
- [x] **Verdict : run sain.** eval_loss 0.925 -> 0.82, décroissance monotone,
      aucune remontée => pas d'overfitting. Le run n'a ni divergé ni été coupé trop tôt.
- [x] Section `train` consultée aussi (8 métriques).
- [x] **PAS d'overfitting, du tout.** train_loss finale ~0.84 vs eval_loss ~0.82.
      mean_token_accuracy : train ~0.81 vs eval ~0.812. Écart de généralisation ~nul.
      => Tu es limitée par la CAPACITÉ du modèle (135M), pas par les données.
      Conséquence : plus de données ou plus d'epochs n'apporteront presque rien.
      Le levier, c'est un modèle plus gros (360M / 1.7B).
- [x] Le plateau apparent en fin de run est CONFONDU avec le learning rate :
      scheduler linéaire 5e-5 -> ~0, donc le LR est quasi nul sur le dernier tiers.
      On ne peut PAS conclure "1 epoch suffisait" depuis ce run. Pour trancher,
      il faudrait un run 2 epochs avec le scheduler étalé sur les 2 epochs.
- [x] `grad_norm` sain : 0.8 -> 0.65, stable 0.6-0.75, petit pic ~1.0 en fin. Pas d'explosion.
- [x] Pas de warmup (warmup_ratio=0 par défaut). À tester sur un prochain run.
- [x] `epoch` 0->1.0 et `global_step` 0->3563 linéaires : checks de sanité OK.

**Piège repéré — axe X trompeur** : le "step" affiché n'est PAS le step d'entraînement.
Trackio compte les événements de log (71 logs train tous les 50 steps + 14 évals
tous les 250 = ~85). CONFIRMÉ par la courbe `train/global_step` : elle monte
linéairement de 0 à 3563 pendant que l'axe X va de 0 à 85.
Pour lire un vrai step, sélectionner `global_step` comme axe X dans la sidebar.

**Piège repéré — échelles d'axe Y** : `runtime` (8.439->8.449 s),
`samples_per_second` (59.18->59.25) et `steps_per_second` oscillent joliment...
sur 0.1% d'amplitude. C'est du jitter matériel, zéro information.
Toujours lire l'échelle Y avant d'interpréter une forme. À filtrer de la vue.

**Hypothèse à tester en Phase 1.4** : l'entropy chute (0.97 -> 0.87) puis plafonne
=> le modèle a appris le FORMAT (accolades, guillemets, clés "name"/"arguments"),
et la loss résiduelle est sur les tokens de CONTENU (valeurs d'arguments).
Prédiction : niveaux 1-2 très hauts, chute entre les niveaux 5 et 6.
Si l'éval confirme -> le problème n'est pas le fine-tuning mais la capacité du 135M.

### 1.2 Reconstruire le split held-out
- [ ] Nouveau fichier `eval_xlam_job.py`, en-tête PEP 723 comme `train_xlam_job.py`.
- [ ] Reproduire EXACTEMENT : `load_dataset` → `map(to_messages)` → `train_test_split(test_size=0.05, seed=42)`.
- [ ] Prendre `split["test"]` **à partir de l'indice 500** (les 500 premiers ont servi de suivi
      pendant l'entraînement → contaminés pour le choix de checkpoint).
- **Fini quand** : tu as ~2500 exemples, et tu as vérifié par un test explicite
      qu'aucune `query` de ton set d'éval n'apparaît dans `split["train"]`.

### 1.3 Calibrer la génération
- [ ] Calculer la longueur en tokens du plus long `answers` de ton set d'éval.
- [ ] En déduire `max_new_tokens` (cette valeur + marge). Sinon des réponses tronquées
      seront comptées comme fausses.
- [ ] Prompt bit-à-bit identique à l'entraînement : même SYSTEM_PROMPT, même chat template,
      `add_generation_prompt=True`.
- [ ] `do_sample=False` (déterminisme), batching avec padding À GAUCHE.
- **Fini quand** : sur 5 exemples, la sortie brute ressemble à du JSON d'appels de fonctions.

### 1.4 Écrire le scoring
- [ ] `json.loads` des DEUX côtés (`answers` de xLAM est une string JSON, pas un objet).
- [ ] Normaliser avant comparaison : tri des clés, comparaison des appels en multiset.
- [ ] Calculer les 7 niveaux, chacun en % :
      1. JSON parsable
      2. Schéma `[{name, arguments}]`
      3. Aucun outil halluciné (tous les `name` ∈ tools fournis)
      4. Bonne cardinalité (bon nombre d'appels)
      5. Multiset des `name` exact
      6. Arguments : précision/rappel sur les paires clé-valeur
      7. Exact match complet  ← métrique headline
- **Fini quand** : tu peux dire où se produit la chute entre les niveaux.

### 1.5 Persister les résultats (LE piège HF Jobs)
- [ ] Pousser les **prédictions par exemple** (query / attendu / prédit / verdicts 1-7)
      dans un repo dataset sur le Hub.
- [ ] Pousser les **métriques agrégées** (JSON) au même endroit.
- Rappel : le disque du job est éphémère. Un `print()` ne survit que dans les logs.
- **Fini quand** : tu peux relire tes prédictions sans relancer de génération.

### 1.6 Lancer
- [ ] `hf jobs uv run --flavor a10g-small --timeout 20m --secrets HF_TOKEN eval_xlam_job.py`
- [ ] Charger le modèle avec `revision=` pinné sur le SHA ci-dessus.
- **Fini quand** : tu as un chiffre d'exact match + le SHA noté à côté.

## Phase 2 — Rendre le chiffre crédible

- [ ] Dans le MÊME job, évaluer aussi :
      - `HuggingFaceTB/SmolLM2-135M` (base, non fine-tuné)
      - `HuggingFaceTB/SmolLM2-135M-Instruct`
      - baseline triviale : toujours le 1er outil, arguments vides
- [ ] n ≥ 500 exemples, et noter l'intervalle de confiance (~±4 pts à n=500).
- [ ] Analyse d'erreurs : lire 30-50 échecs, les ranger par catégorie
      (JSON cassé / outil halluciné / cardinalité / mauvaise valeur / troncature / boucle).
- **Fini quand** : tu peux dire "mon fine-tuning apporte +X points vs le modèle base,
      et voici les 2 causes d'échec dominantes".

>>> POINT D'ARRÊT. À la fin de la Phase 2, l'éval est faite et défendable. <<<

## Phase 3 — Bonus (plus tard, si utile)

- [ ] Courbe d'exact match par checkpoint (revisions step 500 → 3563, 8 points gratuits sur le Hub).
- [ ] Résultats par tranches : nb d'appels attendus (1 / 2 / 3+), nb d'outils dans le prompt.
- [ ] Comparaison appariée (McNemar) quand tu compareras deux variantes de fine-tuning.

---

## Décisions à trancher toi-même (les noter dans le script)

- [ ] L'ordre des appels de fonctions est-il sémantiquement significatif ? (→ liste ou multiset)
- [ ] Typage strict ou tolérant ? (`"5"` vs `5`, `"true"` vs `True`)
- [ ] dtype d'éval : bf16 sur GPU (cohérent avec l'entraînement) — s'y tenir sur tous les runs.
