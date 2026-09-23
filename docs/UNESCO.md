# UNESCO comme taxonomie principale

Le module importe le thésaurus local, sans requête réseau ni appel de modèle.
Le rapport `unesco-import-report.json` décrit les deux formats validés, pas un import effectué dans Neo4j.

## Charger Neo4j

Depuis `C:/code/PAKT/memory`, renseigner `NEO4J_URI`, `NEO4J_USERNAME`,
`NEO4J_PASSWORD` et éventuellement `NEO4J_DATABASE` dans `.env`, puis :

```powershell
.venv/Scripts/python.exe -m graph_memory.unesco C:/code/terminologies/UNESCO-thesaurus-2026 --verify-formats
```

Ajouter `--dry-run` pour valider sans base de données. `--report chemin.json`
enregistre le manifeste. RDF/XML et Turtle sont acceptés ; un dossier privilégie
son unique fichier Turtle. Aucune clé OpenRouter n'est nécessaire pour l'import.

Le service importe aussi au démarrage si `UNESCO_THESAURUS_PATH` est renseigné.
Docker Compose monte ce dossier en lecture seule dans `/terminologies/unesco` ;
pour Compose, fournir le chemin d'un dossier, pas d'un fichier.
En mode live, `UNESCO_REQUIRED=true` empêche de démarrer sans taxonomie importée.
La démonstration sans thésaurus conserve son fonctionnement historique.
Le classificateur factice de démonstration ne remplace pas le modèle sémantique de production.

## Données et relations

Le corpus fourni contient 4 500 concepts et 95 collections (7 domaines et 88
microthésaurus), avec 99 711 triplets RDF. Les concepts gardent leur URI UNESCO,
leur identifiant externe et un identifiant interne UUID déterministe dérivé de l'URI.
`origin` distingue `UNESCO` et `LOCAL`. Les cinq langues sont conservées dans
`pref_labels`, `alt_labels`, `hidden_labels` et `descriptions`. Les notes dont le
contenu est porté par `rdf:value` sont également extraites.

Les libellés affichés utilisent `UNESCO_LABEL_LANGUAGE` (français par défaut),
avec repli anglais. Les propriétés RDF originales sont conservées dans
`metadata.skos_properties`, et les notes dans `metadata.documentation`.

| Source | Graphe Neo4j |
| --- | --- |
| enfant `skos:broader` parent | parent `BROADER_THAN` enfant |
| parent `skos:narrower` enfant | même relation, dédupliquée |
| `skos:related` | une `RELATED_TO`, parcourue dans les deux sens |
| collections, membres et sous-groupes | `TaxonomyGroup` et `HAS_MEMBER` |

Les 4 343 liens hiérarchiques, 6 292 liens associatifs et 4 588 liens de collection
sont conservés. `metadata.skos_triples` garde les prédicats et directions d'origine.
Une boucle `related` présente dans le fichier est préservée et signalée dans le
manifeste. Les cycles hiérarchiques et références manquantes sont refusés.
Les collections servent à naviguer ; elles ne deviennent pas des sujets `ABOUT`.

Attention à l'exemple musical : dans ce corpus, **Music** (`concept353`) et
**Performing arts** (`concept355`) sont associés par `related`, pas par `broader`.
L'import respecte cette distinction. Il n'invente pas une hiérarchie officielle
Culture → Performing arts → Music.

## Indexation progressive

Pour chaque concept extrait d'un document, le résolveur cherche d'abord un
libellé ou synonyme existant, en privilégiant UNESCO. Deux recherches fournissent
ensuite au modèle sémantique une sélection bornée (20 concepts, alternés) :
l'index plein texte, interrogé avec le libellé et les parents proposés (sans la
description libre), sans les mots trop fréquents du thésaurus et avec les formes
singulier/pluriel ; et, une fois le thésaurus vectorisé, l'index vectoriel dédié
`ontology_vector` (libellés dans la langue d'affichage et en anglais, note
d'application). La vectorisation est une étape séparée et payante, reprenable et
idempotente, qui ne revectorise que les concepts modifiés ou un nouveau modèle :
`.venv/Scripts/python.exe -m graph_memory.unesco --embed`. Ces vecteurs ont leur
propre propriété et n'entrent jamais dans la recherche de contenu. Le modèle sémantique choisit un concept équivalent à
réutiliser ou les parents les plus proches pour un nouveau concept LOCAL.
Les concepts locaux existants et les parents explicitement proposés sont aussi
considérés. Plusieurs parents sont permis ; les dépendances locales sont
résolues dans l'ordre avant l'écriture transactionnelle.

Un concept LOCAL doit descendre d'UNESCO. Les identifiants inconnus, parents
absents et décisions de confiance inférieure à `TAXONOMY_PROVISIONAL_THRESHOLD`
(0,5 par défaut) écartent le concept sans créer d'orphelin ni rejeter le
document ; chaque écart est consigné avec sa raison dans l'enregistrement
`classification_review` du document. Entre ce seuil et `TAXONOMY_MATCH_THRESHOLD`
(0,75 par défaut), un nouvel enfant de parents valides est créé à titre
provisoire (`classification_status="provisional"`) et listé pour confirmation ;
la réutilisation d'un concept existant exige toujours la pleine confiance. Un
verdict d'écart est mémorisé par libellé jusqu'au changement d'instantané, de
seuil ou de version des prompts (`--clear-classification-cache` l'efface). Le
score est une estimation du modèle, pas une garantie de justesse.

Les documents, sections et passages obtiennent des relations `ABOUT` vers les
concepts retenus et leurs ancrages UNESCO. L'ingestion ne peut pas modifier les
relations officielles. Les extensions locales et leurs relations transversales
restent possibles. Les anciens documents ne sont pas automatiquement réindexés :
une migration explicite est nécessaire pour un corpus préexistant.

La réimportation du même instantané est idempotente, y compris entre RDF et TTL.
Une empreinte différente est refusée pour éviter de conserver silencieusement
des relations obsolètes : une mise à jour d'édition demande une migration explicite.
La langue d'affichage fait partie de cette empreinte.

## Navigation

`GET /memory/taxonomy` retourne le manifeste et les sept domaines racines.
Le bouton **UNESCO taxonomy** affiche ces domaines ; sélectionner un nœud et
**Expand neighbors** permet de parcourir groupes, concepts et documents indexés.
Les concepts portent le préfixe UNESCO ou LOCAL. Les recherches automatiques
réservent une partie de leurs points d'entrée à UNESCO, avec un budget total
inchangé et des accès directs aux documents toujours possibles.

Les API de nœuds exposent URI, origine, labels multilingues et métadonnées.
La lecture des voisins conserve les métadonnées des relations officielles.

## Attribution

Source : UNESCO Thesaurus, instantané local fourni, modification déclarée
2026-08-31. Titulaire : UNESCO. Licence déclarée dans le corpus :
[CC BY-SA 3.0 IGO](http://creativecommons.org/licenses/by-sa/3.0/igo/).
Le manifeste conserve les propriétés de source, droits et licence du schéma.
