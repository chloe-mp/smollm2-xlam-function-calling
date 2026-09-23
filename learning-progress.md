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

### Faiblesse #5 — se représenter la FORME des données (auto-diagnostiquée)
Symptôme : incapable de traduire un placeholder ("mets ici une chaîne") en expression
concrète (`clean_eval[0]["answers"]`). Tendance à coller littéralement les noms
d'exemple donnés en illustration (`chaine`, `un_texte`, `overlap`).
Cause : ne garde pas en tête le type/la forme de chaque variable en cours.
CE N'EST PAS un déficit de raisonnement — c'est une habitude d'inspection manquante.

**Technique prescrite — "la descente"** : descendre dans la structure une couche
à la fois en REGARDANT à chaque étape, jamais en supposant.
    clean_eval                 -> Dataset(num_rows=2395)
    clean_eval[0]              -> dict
    clean_eval[0]["answers"]   -> str  ✅
**Réflexe devant un trou à remplir** : (1) de quel TYPE cette fonction a-t-elle besoin ?
(2) où est-ce que j'ai ce type ? -> descente.

C'est la même méthode qui lui a fait trouver le bug `remove_columns` (afficher
`column_names` au lieu de supposer). Elle sait le faire, elle ne le généralise pas encore.

### Diagnostic corrigé (2026-08-19) — IMPORTANT
Auto-diagnostic répété : "c'est trop abstrait pour moi", "la syntaxe c'est la cata".
**C'est faux et il faut le contredire.** Preuve dans la même session : dérivation du 33 %,
biais d'exposition trouvé seule, anticipation de l'utilité de `tools`/`query`.
Le raisonnement abstrait est là.

Ce qui manque réellement : **le suivi des TYPES**. Savoir, à chaque étape, ce qu'on
tient en main et ce que la fonction attend. Aggravé par l'API `datasets` qui est
objectivement piégeuse :
  clean_eval               -> Dataset
  clean_eval[0]            -> dict (une LIGNE)
  clean_eval[0]["answers"] -> str
  clean_eval["answers"]    -> list[str] (une COLONNE)
Indexer par int = ligne ; indexer par str = colonne. Même syntaxe, opérations différentes.

**Technique prescrite** : ne pas raisonner de tête. Enchaîner des `type()` dans le REPL
jusqu'à voir le chemin de ce qu'on a vers ce que la fonction veut.
Règle : "la fonction veut X, j'ai Y — quel est le chemin ?" résolu par inspection, pas par
déduction mentale.

Sous-pattern à surveiller : colle les noms de remplacement de l'assistant comme du code
littéral (`chaine`, `un texte`, `overlap`). Signe qu'elle lit le code comme du texte à
recopier plutôt que comme des slots à remplir. -> Toujours lui donner des exemples
CONCRETS avec ses vraies variables, jamais de pseudo-code.

### Drill en attente
Dix minutes sur `range` / slicing / bornes / indice vs compte.
À FUSIONNER avec un drill "descente de structure" (type/forme des objets) — même racine :
deviner au lieu de regarder.

### Session 2026-08-21 — Éval complète livrée (quiz 2/5)

L'exercice 1.4 est FAIT : scorer à 7 niveaux écrit, débuggé et tourné sur 500 held-out.
Résultat : ft 23,8 % exact-match / meilleur prompting 1,6 % / baseline triviale 0,2 %,
sur 7 conditions. Le scorer a survécu à 5 révisions successives en une journée.

**Trois bugs, trois natures différentes — c'est la leçon de la journée :**
  hashabilité    -> crash franc, trouvé SEULE, corrigé seule
  right-padding  -> AUCUNE erreur, scores faux et publiables
  skew 3.14/3.12 -> passe en local, casse sur le runner (PEP 649)
Le deuxième est le dangereux : un chiffre stable, reproductible et faux. Corrélé à la
longueur du prompt, donc touchant une sous-population identifiable -> ne s'annule pas
en moyenne, contrairement à du bruit.

### Signal fort à retenir (2026-08-21)
**A contredit l'assistant sur le fond, et avait raison.** Sur « un reviewer dira qu'il
manque le décodage contraint », elle répond « sur un 135M je ne sais pas si ça aurait
changé grand chose ». Simulation sur ses propres prédictions : L3 23 % -> 79 % (+56),
mais exact-match 1,6 % -> 3,2 % (+1,6 seulement). Son intuition d'ingénieure battait
l'argument d'autorité. À lui rappeler la prochaine fois qu'elle doute de son niveau.

Idem sur la relecture d'article : a relevé, à juste titre, qu'on lui reprochait
l'absence d'une section dans un brouillon annoncé comme inachevé.

### Angle mort confirmé : correctif appliqué ≠ mécanisme acquis
Quiz raté (2/5) sur des bugs qu'elle avait corrigés de sa main le matin même.
Nuance décisive : `EVAL_TODO.md` contenait DÉJÀ
`tok.padding_side = "left"  # décodeur : le padding à droite casse la génération`.
Elle avait la connaissance ; une réécriture du script l'a effacée.
=> Ce n'est pas une lacune conceptuelle, c'est une **perte par refactor**.
=> Remédiation : quand on remplace un bloc de code, énumérer les décisions qu'il
   portait avant de le jeter. Le TODO était le bon réflexe, il n'a pas été relu.

### Drill en attente
Dix minutes sur `range` / slicing / bornes / indice vs compte.
À FUSIONNER avec un drill "descente de structure" (type/forme des objets) — même racine :
deviner au lieu de regarder.

### Prochaine révision recommandée
- **J+2 (08-23)** : re-quiz à froid — (a) mécanisme exact du right-padding en génération
  par batch (le modèle prédit depuis la DERNIÈRE position ; en right-padding c'est un PAD),
  (b) PEP 649 et pourquoi aucun test local n'attrape un skew d'environnement.
- **J+7** : vacuous truth dans un contexte neuf (`all([])`, et le piège
  « toutes les validations sont passées » quand la liste de validations est vide).
- **Plus tard, quand le post-training démarrera** : elle a déjà construit sans le savoir
  le prérequis de GRPO — une fonction de récompense déterministe et graduée (`score_one`,
  7 niveaux). Pour DPO : échantillonner k complétions, trier par score, paires best/worst
  = labels de préférence gratuits. Le mode d'échec visé est identifié : le modèle s'arrête
  après le premier appel (130 des 158 sous-prédictions sont à exactement −1 ; 39,8 % d'exact
  match à 1 appel, 9,1 % à 2, 0 % à 4+).
  Ressource : AVB (@neural_avb) — CPT/SFT/DPO/GRPO sur un 135M, 2 h 22
  github.com/avbiswas/finetuning_recipes

---

## 2026-09-22/23 — LoRA & QLoRA sur 1.7B, puis juge LLM

### ⚠️ Nature de cette session — à lire avant le reste
Session **déléguée** : scripts écrits par l'assistant, sur demande explicite.
La règle d'`EVAL_TODO.md` (« je code, Claude relit ») n'a PAS été appliquée.
Ce qui suit est donc un journal de RÉSULTATS et de PIÈGES, pas une preuve
d'acquisition. Le contenu à s'approprier est listé en fin de section.

### Résultat principal — le plafond du 135M était bien un plafond de capacité
|                | exact-match | réf. non dérivables | jugé correct sur le reste |
|----------------|-------------|---------------------|---------------------------|
| lora-1.7b      | 75,11 %     | 127 (5,3 %)         | 84,55 %                   |
| qlora-1.7b     | 72,61 %     | 134 (5,6 %)         | 82,29 %                   |
| ft 135M        | 21,80 %     | —                   | —                         |

+53 points en passant de 135M full-FT à 1.7B LoRA, sur données identiques.
L'hypothèse de la Phase 1.1 (« le levier, c'est un modèle plus gros ») est vérifiée.
QLoRA : −2,5 pts d'exact-match pour −38 % de mémoire (5,87 Go vs 9,48 Go de pic),
et 9 min de MOINS que LoRA sur A100 — la déquantification 4 bits coûte moins que
la bande passante mémoire qu'elle économise. Contre-intuitif, mesuré.

**Correction d'un chiffre publié** : `ft` = 21,80 % sur les 2395, pas 23,8 %.
Le 23,8 % venait des 500 premiers. Même modèle, même révision, n plus grand.

### Les 4 crashs de la session ont TOUS la même forme
Un coût est payé, puis le processus meurt AVANT de persister quoi que ce soit.
1. **trackio + LoRA** : au 1er push (step 500), `rank_pattern={}` du LoraConfig
   devient un struct Parquet sans champ -> `ArrowNotImplementedError`. 2 × 15 min
   d'A100 perdues. Le full-FT du 135M n'avait pas de `peft_config`, d'où l'absence
   du bug avant.
2. **Solde HF épuisé** : les deux runs coupés net à 40 min (steps 1394 et 1528).
   `hub_strategy="every_save"` pousse l'adaptateur mais PAS l'optimizer
   -> rien de reprenable. Corrigé en `"checkpoint"` (+ `--resume`).
3. **`push_to_hub` de l'éval** : `repo_info.download_size -= deleted_size` avec
   un `download_size` absent du README écrit à la main -> `TypeError`, APRÈS
   l'upload des données et AVANT le commit. 1 h 45 d'A100 générée, zéro ligne
   sur le Hub, scores visibles uniquement dans les logs.
4. (Non-crash mais même famille) **`provider="auto"`** du juge : 0,06 appel/s
   en charge soutenue contre 4,0/s sur provider épinglé. 17 h d'ETA au lieu de 20 min.

=> **Règle à retenir** : tout ce qui coûte cher doit être persisté AVANT l'étape
   décorative (mise à jour de carte, sync de dashboard). Le repli `push_parquet_directly`
   applique ça : si le beau chemin échoue, écrire quand même le parquet brut.

### Pièges d'API rencontrés (transférables)
- `warmup_ratio` n'existe plus en transformers 5 : un float < 1 dans `warmup_steps`.
- TRL charge les modèles en **float32 par défaut** : sans `dtype="bfloat16"` explicite,
  un 1.7B pèse 6,8 Go au lieu de 3,4 et le « LoRA non quantifié » est faussement lourd.
- `apply_chat_template(...)` renvoie un `BatchEncoding` en transformers 5 :
  `len(...)` vaut 2 (le nombre de clés), pas la longueur en tokens. Piège de mesure.
- Provider épinglé ≠ détail de perf : deux providers servent des quantifications
  différentes du même modèle. Un verdict de juge ne doit pas dépendre du routage.

### Juge LLM — ce que la session a démontré
**Le barème est le juge.** Première version : 5 échecs sur 10 requalifiés en
« équivalents », dont des coordonnées GPS différentes et `{"$lt": 30}` vs `"<30"`.
Un juge permissif produit un score corrigé stable, reproductible et FAUX —
exactement le mode d'échec du right-padding (session du 08-21).
Correctifs qui ont marché :
- rappeler au juge que le scorer normalise DÉJÀ le typage, donc toute différence
  qu'il voit est réelle ;
- **deux questions indépendantes** (`reference_derivable`, `verdict`) : en un seul
  champ, la règle « dans le doute, incorrect » écrasait le signal de référence
  inutilisable, qui est précisément ce qu'on cherchait ;
- juger AUSSI les 3538 exact-match : ils forment un jeu de validation gratuit.
  Résultat : **99,89 % d'accord**, 4 désaccords, tous du côté « la référence est
  douteuse » — jamais en faveur du modèle. Le juge est prudent, pas complaisant.
- déterminisme vérifié : deux passages sur 20 cas, 100 % de verdicts identiques.

**Bon appel de Chloé** : avoir choisi de juger les 4790 prédictions et pas
seulement les 1252 échecs, contre la recommandation de l'assistant (qui n'y voyait
qu'un surcoût ×4). C'est ce choix qui a fourni le jeu de validation du juge, donc
la seule raison de croire les chiffres corrigés. À relire le jour où l'argument
d'autorité sera tentant — c'est le 3e cas noté dans ce fichier.

### Ce qu'il reste à s'approprier (sinon la session ne compte pas)
1. Re-dériver seule **pourquoi QLoRA peut être plus rapide que LoRA** (poids 4 bits
   = moins d'octets lus par token ; la déquantification est du calcul, pas de la mémoire).
2. Refaire à la main le calcul de **mémoire d'activation** qui impose batch 8 au lieu
   de 16 sur un 1.7B, et le KV cache de l'éval (32 × 2100 × 197 Ko ≈ 13 Go).
3. Réécrire `make_card_pushable` **sans regarder** : quelle est la condition exacte
   qui fait planter `push_to_hub`, et pourquoi la supprimer suffit.
4. Quiz à froid J+2 : les 4 crashs ci-dessus, énoncer pour chacun *à quel moment
   précis* le processus meurt par rapport au moment où le coût est payé.

### Suite logique
- Analyse d'erreurs Phase 2 sur les **960 erreurs réelles** : 710 sont des valeurs
  d'arguments (`wrong_value`). C'est un problème de contenu, pas de format — le
  format est acquis à 99,2 % (niveau 1-2).
- Les 257 références non dérivables sont identifiées ligne à ligne dans
  `Chloemp/xlam-smollm2-judge-results` : matière à un set d'éval nettoyé.
- DPO/GRPO : le juge fournit maintenant un signal de préférence en plus de
  `score_one`, sur les cas où l'exact-match est muet.
