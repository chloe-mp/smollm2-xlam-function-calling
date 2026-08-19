# Learning progress

## 2026-08-19 — Évaluation de fine-tuning (xLAM / SmolLM2-135M)

### Sujets vus
- Méthodologie d'éval d'un fine-tuning : set held-out, protocole de génération,
  échelle de métriques à 7 niveaux, baselines, analyse d'erreurs, slices.
- Contamination train/eval : reproduire un split à l'identique (seed + ordre des ops).
- Spécificités infra éphémère (HF Jobs) : pinner la révision Hub, persister les résultats.
- Lecture de courbes d'entraînement (trackio) : train vs eval, grad_norm, LR schedule.
- Écart de généralisation nul => limite de CAPACITÉ, pas de données.
- Pourquoi `mean_token_accuracy` ne dit rien sur l'exact match.

### Exercice fait
**Décomposition token accuracy → accuracy sur les tokens de contenu.** Réussi.
50 tokens × 0,81 = 40 corrects → 35 partent dans la structure JSON → 5 restants
sur 15 tokens de contenu → **~33 %**. Dérivé en 3 étapes avec scaffolding léger
(une relance par étape, aucune réponse donnée). Bonne intuition initiale :
a identifié seule que les tokens bien prédits sont ceux du JSON structurel.

### Faiblesses repérées
1. **Contamination train/eval** — le premier `eval_xlam.py` évaluait sur `ds[0]` du
   split train complet, donc sur une donnée d'entraînement. Réflexe à ancrer :
   avant tout chiffre, "d'où vient cet exemple ?".
2. **Lire les commentaires comme du code** — commentaire faux sur `hub_strategy="every_save"`
   ("pousse une fois à la fin" alors qu'il pousse à chaque save, 10 commits le prouvent).
3. **Échelle des axes** — piège des graphes auto-échelle (runtime qui "oscille" sur 0,1 %).
4. **Fluidité Python sur les built-ins** (auto-diagnostiquée "la syntaxe c'est la cata",
   mais c'est plus étroit que ça) :
   - `x = 500, 3000` crée un TUPLE — c'est la virgule qui fait le tuple, pas les parenthèses.
   - Signature de `range` : `range(stop)` / `range(start, stop)` / `range(start, stop, step)`,
     `stop` toujours EXCLU.
   - Confusion compte vs indice (2500 exemples restants vs indice de fin 3000).
   Racine commune : deviner au lieu de vérifier. Remède prescrit = garder un REPL Python
   ouvert en permanence et tester le doute en 5 secondes.
   ⚠️ NE PAS généraliser : le raisonnement est solide (voir "Réussites" ci-dessous),
   c'est la fluidité sur quelques built-ins qui manque. Se comble en quelques heures.

### Réussites notables (à relire les jours de doute)
- Insight central trouvé seule : les tokens bien prédits sont ceux de la structure JSON.
- Dérivation du 33 % en 3 étapes, sans qu'aucun chiffre ne soit donné.
- A proposé de supprimer `remove_columns` — solution MEILLEURE que celle attendue,
  et pour la bonne raison (garder `tools` et `query` sert aux niveaux 3 et à l'analyse d'erreurs).
- A repéré spontanément que `EVAL_SKIP` allait devenir du code mort.

5. **Tendance à empiler la théorie avant de pratiquer** — 3 pivots successifs vers
   "explique-moi encore" avant de toucher au code. Le passage à l'exercice a
   immédiatement produit plus de compréhension que les 3 tours précédents.

### Question résolue — teacher forcing (2026-08-19) ✅
Répondu juste, sans aide : la performance réelle est PIRE que ce que suggère
`mean_token_accuracy`, parce qu'en génération le modèle conditionne sur ses propres
erreurs. Nom du phénomène : **biais d'exposition** (exposure bias) -> accumulation
d'erreurs. À l'entraînement le modèle ne voit que des préfixes corrects ; dès sa
première erreur en génération il sort de sa distribution d'entraînement.

Bilan des deux distorsions, de sens OPPOSÉ :
- tokens non uniformes (structure JSON gratuite) -> `0,81^50` trop PESSIMISTE
- biais d'exposition -> `0,81` trop OPTIMISTE
=> amplitudes inconnues sans mesure. C'est LA justification de toute la Phase 1.

### Drill en attente
Dix minutes sur `range` / slicing / bornes / indice vs compte, à faire juste après la 1.2.

### Prochaine révision recommandée
- **Court terme (prochaine session)** : exercice 1.4 — écrire la fonction de scoring
  normalisée (niveaux 1→7). C'est le cœur intellectuel de l'éval.
- **J+3** : re-tester la contamination — "comment je reconstruis un split held-out
  identique et comment je le vérifie ?" sans regarder les notes.
- **J+7** : teacher forcing / compounding errors, et lecture de courbes
  (plateau réel vs plateau causé par le LR schedule).
