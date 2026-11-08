# Football Player Similarity Engine

**Finding statistically similar football players across leagues using StatsBomb event data, role-specific features, UMAP, and nearest-neighbor search.**

[English](#english) · [Français](#français)

---

<a id="english"></a>
## English

### Overview

This project builds a **historical football player similarity engine** for the **2015/16 season**. Given a player, it retrieves players with comparable statistical profiles, including players from other leagues.

Instead of relying on goals and assists alone, the system transforms detailed StatsBomb match events into features describing **passing, ball progression, carrying, shooting, and defensive activity**. Players are compared **within the same broad playing role**.

The main retrieval method is **UMAP + NearestNeighbors**, with direct Euclidean distance on standardized features as a baseline. This is an **unsupervised similarity system**, not a match-outcome predictor, player-quality ranking, or supervised classifier.

### Dataset and coverage

Source: [StatsBomb Open Data](https://github.com/statsbomb/open-data), accessed using `statsbombpy`.

| League | Competition ID | Available 2015/16 matches | Eligible player profiles (≥ 900 minutes) |
| --- | ---: | ---: | ---: |
| Premier League | 2 | 380 | 327 |
| La Liga | 11 | 380 | 344 |
| Serie A | 12 | 380 | 343 |
| Ligue 1 | 7 | 377 | 342 |
| Bundesliga | 9 | 34 | 17 |
| **Total** | | **1,551** | **1,373** |

The processed dataset contains **43,044 player-match records** and **2,560 player–competition–season profiles** before applying the playing-time threshold. The final feature table contains **70 columns** (identifiers, raw statistics, engineered features, and metadata); **only a role-specific subset** is used to calculate similarity.

> **Coverage warning:** The Bundesliga subset contains just 34 matches, and Ligue 1 contains 377 rather than 380. The leagues are not equally represented, and the Bundesliga results should not be interpreted as league-wide coverage.

### Pipeline

```text
StatsBomb Open Data
        │
        ▼
Match catalog + event extraction + lineups
        │
        ▼
Cleaned event files and player minutes (Parquet)
        │
        ▼
Player-match → player-season aggregation (DuckDB)
        │
        ▼
Role-specific feature engineering
(per 90 minutes, passing, progression, shooting, defense)
        │
        ▼
Filter: ≥ 900 minutes + group by playing role
        │
        ▼
StandardScaler fitted separately for each role
        │
        ├───────────────┐
        ▼               ▼
Euclidean baseline   UMAP embedding (up to 5 dimensions)
                        │
                        ▼
                  NearestNeighbors
                        │
                        ▼
                Top-k similar players
                        │
                        ▼
          Notebook: stats + pass/shot maps
```

**Player identity:** Each profile uses the composite key `(competition_id, season_id, player_id)`. This prevents statistics for a player appearing in multiple competitions from being mixed together.

### How the UMAP method works

**UMAP (Uniform Manifold Approximation and Projection)** is a nonlinear dimensionality-reduction method. In this project, UMAP is used to create a compact representation of players that attempts to **preserve local neighborhood relationships** in their statistical features. A separate representation is learned for each playing role.

The complete process is:

**1. Represent each player using relevant statistics.**  The system selects a different set of features depending on the position: goalkeeper, centre-back, full-back, defensive midfielder, central midfielder, wide player, attacking midfielder, or forward. Examples include progressive passes per 90, passing completion rate, ball recoveries per 90, and non-penalty xG per 90. Features are defined from the available event data, not from manually assigned similarity labels.

**2. Normalize the feature scales.**  For each role, `StandardScaler` transforms each selected variable using its role-specific mean and standard deviation:

\[
z_{ij}=\frac{x_{ij}-\mu_j}{\sigma_j}
\]

This matters because a quantity such as forward passing distance could otherwise dominate a rate such as pass completion simply because of its numerical scale. Each role's scaler is fitted on its eligible players **from all included leagues together**.

**3. Learn a lower-dimensional representation using UMAP.**  UMAP starts from the standardized player vectors, models their local neighborhood relationships, and optimizes an embedding intended to retain those relationships as much as possible. The project uses:

```python
reducer = umap.UMAP(
    n_neighbors=min(10, len(X) - 1),
    n_components=min(5, X.shape[1], len(X) - 2),
    min_dist=0.1,
    metric="euclidean",
    random_state=42,
)
embedding = reducer.fit_transform(X.to_numpy())
```

- `n_neighbors=10` (where sufficient players are available) controls the local neighborhood scale UMAP considers when learning the representation.
- `n_components=5` (capped for smaller role datasets) produces an embedding with up to **five dimensions**, not a two-dimensional visualization.
- `min_dist=0.1` affects how closely embedded points can be packed.
- `metric="euclidean"` defines distances in the **standardized input space** while building UMAP's neighborhood structure.
- `random_state=42` helps make repeated runs reproducible and can cause UMAP to disable parallel execution, producing a harmless warning.

**4. Search the embedding for nearest neighbors.**  A separate scikit-learn `NearestNeighbors(metric="euclidean")` model is fitted to each role's UMAP coordinates. For a selected player, it retrieves the nearest profiles in that embedded space, removes the query player, and returns the top `k` neighbors. Candidates can come from **different leagues**, but must belong to the **same playing role**.

The retrieval distance is:

\[
d(i,j)=\left\|\mathbf{u}_i-\mathbf{u}_j\right\|_2
\]

where \(\mathbf{u}_i\) and \(\mathbf{u}_j\) are the players' coordinates **after UMAP**, not their original statistical vectors.

**5. Interpret the results using the original features.**  The notebook displays the nearest players, a heatmap of standardized role features, and a player-to-player comparison based on feature differences. It also shows completed-pass maps and shot maps from StatsBomb coordinates. These views help inspect the recommendations; **they do not provide a causal explanation of the nonlinear embedding**.

#### UMAP vs. the Euclidean baseline

| | Standardized Euclidean baseline | UMAP + NearestNeighbors |
| --- | --- | --- |
| Input for retrieval | Standardized role-feature vectors | UMAP coordinates learned from those vectors |
| Transformation | Linear rescaling only | Nonlinear dimensionality reduction |
| Main strength | Simple and directly interpretable | Can capture complex local relationships |
| Main limitation | Sensitive to feature choice and scale | Can distort global distances and rankings |

Both methods are implemented for comparison. **UMAP is not automatically better**: a useful assessment requires stability checks, feature-level inspection, and football-domain judgment. Neither method is trained with ground-truth labels of player similarity.

**Important interpretation rule:** A UMAP distance of `0.35` does **not** mean “35% similar.” Distances depend on the fitted role-specific embedding and should not be compared across roles or treated as calibrated similarity scores.

### Feature engineering

Examples of implemented feature families:

| Dimension | Examples |
| --- | --- |
| Passing | Attempts and completions per 90, pass completion rate, average pass length |
| Ball progression | Progressive passes per 90, completed forward distance per 90 |
| Carrying | Carries per 90, progressive carries per 90, average carry length |
| Shooting | Shots and non-penalty xG per 90, non-penalty xG per shot |
| Defensive activity | Pressures, counterpressures, interceptions, recoveries, clearances, blocks per 90 |
| Spatial activity | Average pressure height, high pressures per 90 |

In this implementation, a **progressive pass** is a *completed* pass that reduces the player's distance to the opposition goal by at least **25%**. A **progressive carry** follows the same distance-reduction rule for a carry. These are **project-specific operational definitions**, not universal football analytics standards.

The feature configuration is stored in `config/role_feature_sets.json`. Features are selected separately for eight roles, and players with fewer than 900 recorded minutes are excluded from similarity search to reduce noise from very small samples.

### Example: Wes Morgan (Premier League, 2015/16)

An observed query for `player_id=3813`, `competition_id=2`, returned examples such as:

| Method | Examples of nearby players |
| --- | --- |
| Standardized Euclidean | Robert Huth (Premier League), Kyriakos Papadopoulos (Bundesliga), Alessandro Gamberini (Serie A) |
| UMAP + NearestNeighbors | Guglielmo Stendardo (Serie A), Kyriakos Papadopoulos (Bundesliga), Brou Benjamin Angoua (Ligue 1) |

The rankings differ because UMAP changes the representation used for distance calculations. These results are **candidate statistical analogues**, not a verified ranking of footballing ability or playing-style equivalence.

### How to run

Run the commands from the project root in a Python environment with the project dependencies installed (`pandas`, `numpy`, `duckdb`, `pyarrow`, `statsbombpy`, `scikit-learn`, `umap-learn`, `matplotlib`, `mplsoccer`, and Jupyter support).

```bash
python -m src.extract_matches
python -m src.extract_events
python -m src.extract_lineups
python -m src.build_player_tables
python -m src.build_features
```

The extraction scripts reuse existing downloaded files when present. The Big Five aggregation and feature-engineering steps produce `data/processed/player_match_big5.parquet`, `player_season_big5.parquet`, and `player_features_big5.parquet`, preserving the original Premier League tables.

To query the similarity engine:

```python
from src.similarity import (
    build_role_matrices,
    build_umap_models,
    find_similar_players_umap,
)

matrices = build_role_matrices()
models = build_umap_models(matrices)

neighbors = find_similar_players_umap(
    player_id=3813,      # Wes Morgan
    competition_id=2,    # Premier League
    season_id=27,        # 2015/16
    matrices=matrices,
    models=models,
    top_k=10,
)
print(neighbors)
```

Open **`notebooks/05_similarity_demo_big5.ipynb`** to search by player name and explore the UMAP results, statistical comparisons, completed-pass maps, and shot maps. No deployed web application is required.

### Project structure

```text
config/
  role_feature_sets.json
src/
  extract_matches.py
  extract_events.py
  extract_lineups.py
  events.py
  lineups.py
  build_player_tables.py
  features.py
  build_features.py
  similarity.py
notebooks/
  02_data_engineering.ipynb
  05_similarity_demo_big5.ipynb
data/                         # Generated data; availability depends on local setup
  raw/
  interim/player_minutes/
  processed/events/
  processed/player_match_big5.parquet
  processed/player_season_big5.parquet
  processed/player_features_big5.parquet
```

### Validation and limitations

- **Implemented data checks:** match coverage, unique match-player records, season aggregation totals, playing-minute validity, role coverage, finite similarity inputs, and consistency of selected event-derived totals.
- **Evaluation status:** direct Euclidean and UMAP retrieval methods have been compared qualitatively. The cross-league outputs are **not** independently validated ground truth. More robust evaluation would test recommendation stability under feature removal, UMAP parameter changes, and football-specific case studies.
- **Uneven coverage:** only 34 Bundesliga matches are present; 17 Bundesliga profiles pass the 900-minute threshold. This limits representativeness and cross-league conclusions.
- **Context effects:** event statistics reflect team tactics, possession, opponent strength, and league playing styles. Per-90 normalization does not remove those biases.
- **Historic scope:** the data represent 2015/16, not contemporary player performance. This is a portfolio/research demonstration, not a current scouting recommendation system.

### Technologies

Python · pandas · NumPy · DuckDB · Parquet / PyArrow · StatsBombPy · scikit-learn · UMAP (`umap-learn`) · Jupyter · Matplotlib · mplsoccer

### Data attribution

Data: [StatsBomb Open Data](https://github.com/statsbomb/open-data). Credit StatsBomb when using or sharing analyses based on these data, and follow the repository's attribution and license requirements.

---

<a id="français"></a>
## Français

### Présentation

Ce projet est un **moteur de recherche de joueurs de football statistiquement similaires**, construit à partir des données événementielles StatsBomb de la saison **2015/2016**. Lorsqu'on sélectionne un joueur, le système propose des profils proches, y compris dans d'autres championnats.

L'objectif n'est pas simplement de comparer les buts et les passes décisives. Nous construisons des caractéristiques décrivant **la distribution du ballon, la progression par la passe, les courses avec ballon, la finition et l'activité défensive**. Les comparaisons sont effectuées **entre joueurs du même grand rôle**.

La méthode principale combine **UMAP et NearestNeighbors**. Une recherche par distance euclidienne sur les variables standardisées sert de référence. Il s'agit d'un système **non supervisé**, et non d'un modèle de prédiction de résultats, d'un classement de talent ou d'un classificateur.

### Données et périmètre

Source : [StatsBomb Open Data](https://github.com/statsbomb/open-data), via `statsbombpy`.

| Championnat | Matchs disponibles en 2015/16 | Profils éligibles (≥ 900 minutes) |
| --- | ---: | ---: |
| Premier League | 380 | 327 |
| La Liga | 380 | 344 |
| Serie A | 380 | 343 |
| Ligue 1 | 377 | 342 |
| Bundesliga | 34 | 17 |
| **Total** | **1 551** | **1 373** |

Le pipeline produit **43 044 lignes joueur-match** et **2 560 profils joueur–championnat–saison** avant le filtrage par temps de jeu. Le fichier final comporte **70 colonnes**, mais le calcul de similarité n'utilise que les variables pertinentes pour chaque rôle.

> **Limite des données :** seulement 34 matchs de Bundesliga sont disponibles dans l'extraction, et 377 pour la Ligue 1. Les cinq championnats ne sont donc pas couverts de façon équivalente.

### Fonctionnement général

```text
StatsBomb Open Data
        ↓
Extraction des matchs, événements et compositions
        ↓
Nettoyage et stockage Parquet
        ↓
Agrégation joueur-match puis joueur-saison (DuckDB)
        ↓
Création de variables statistiques par rôle
        ↓
Filtrage ≥ 900 minutes + regroupement par poste
        ↓
StandardScaler propre à chaque rôle
        ↓
    ┌──────────────────┴─────────────────┐
    ↓                                    ↓
Distance euclidienne                UMAP (jusqu'à 5D)
                                         ↓
                                  NearestNeighbors
                                         ↓
                               Joueurs les plus proches
                                         ↓
                    Notebook : statistiques et cartes de terrain
```

Chaque profil est identifié par `(competition_id, season_id, player_id)` pour éviter de fusionner par erreur les statistiques d'un joueur ayant évolué dans plusieurs championnats.

### Comprendre précisément notre méthode UMAP

**UMAP (Uniform Manifold Approximation and Projection)** est une méthode **non linéaire de réduction de dimensionnalité**. Nous l'utilisons pour représenter les joueurs dans un espace plus compact tout en essayant de **préserver les voisinages locaux** définis par leurs statistiques.

Notre méthode suit cinq étapes.

**1. Transformer chaque joueur en vecteur de caractéristiques.**  Un joueur est décrit par des variables adaptées à son rôle : passes progressives par 90 minutes, taux de réussite des passes, récupérations de balle, xG hors penalty, etc. Nous utilisons huit rôles : gardien, défenseur central, latéral, milieu défensif, milieu central, joueur de couloir, milieu offensif et attaquant. Il n'existe pas de « bonne réponse » de similarité fournie au modèle pendant l'apprentissage.

**2. Standardiser les caractéristiques avec `StandardScaler`.**  Pour chaque rôle, nous centrons et réduisons chaque variable :

\[
z_{ij}=\frac{x_{ij}-\mu_j}{\sigma_j}
\]

Ainsi, une mesure exprimée en distance parcourue par les passes ne domine pas automatiquement une proportion comme le taux de réussite, simplement à cause des unités. **Tous les championnats sont standardisés ensemble, à l'intérieur de chaque rôle.**

**3. Apprendre une représentation UMAP.**  À partir des vecteurs standardisés, UMAP étudie les voisinages des joueurs et cherche une représentation de dimension plus faible qui en conserve autant que possible les relations locales.

Notre configuration est :

```python
umap.UMAP(
    n_neighbors=min(10, len(X) - 1),
    n_components=min(5, X.shape[1], len(X) - 2),
    min_dist=0.1,
    metric="euclidean",
    random_state=42,
)
```

- **`n_neighbors` :** jusqu'à 10 voisins sont pris en compte pour construire la structure locale.
- **`n_components` :** chaque joueur est représenté par **jusqu'à cinq coordonnées** ; notre moteur n'utilise donc pas simplement un graphique en deux dimensions.
- **`min_dist` :** détermine à quel point les points de la représentation peuvent être rapprochés.
- **`metric` :** la distance euclidienne est utilisée pour définir les voisinages **avant la réduction de dimension**.
- **`random_state=42` :** permet de mieux reproduire les résultats. L'avertissement éventuel sur `n_jobs` est normal.

**4. Trouver les joueurs les plus proches.**  Une fois l'espace UMAP appris, nous entraînons un index `NearestNeighbors` sur les coordonnées obtenues. Pour un joueur donné, le programme récupère les `k` profils les plus proches **dans cet espace réduit**, en excluant le joueur interrogé.

La distance utilisée pour la recherche est :

\[
d(i,j)=\left\|\mathbf{u}_i-\mathbf{u}_j\right\|_2
\]

Les vecteurs \(\mathbf{u}_i\) et \(\mathbf{u}_j\) correspondent aux **coordonnées UMAP**, et non aux statistiques de départ. On peut obtenir des voisins d'autres championnats, mais **toujours parmi les joueurs appartenant au même rôle**.

**5. Contrôler les recommandations grâce aux statistiques originales.**  Le notebook permet d'afficher les voisins, une heatmap des variables standardisées, les différences statistiques entre deux joueurs, puis leurs cartes de passes réussies et de tirs. Ces graphiques aident à interpréter les profils, mais ne constituent **pas une explication causale de la transformation UMAP**.

#### Pourquoi utiliser UMAP plutôt que la simple distance euclidienne ?

La distance euclidienne standardisée compare directement les variables sélectionnées. C'est une méthode simple et utile comme point de référence.

UMAP introduit une transformation **non linéaire** qui peut faire ressortir des relations locales plus complexes. Deux joueurs peuvent donc être proches avec UMAP sans être exactement les plus proches dans l'espace statistique initial.

En contrepartie, la projection UMAP **peut déformer les distances et modifier le classement des voisins**. Il serait incorrect d'affirmer qu'elle est nécessairement meilleure que la méthode euclidienne sans évaluation supplémentaire.

**À retenir : une distance UMAP de `0,35` n'est pas un score de similarité de 35 %.** Les distances dépendent de la représentation apprise pour un rôle donné ; elles ne sont pas directement comparables entre rôles ni calibrées en pourcentages.

### Variables construites

| Dimension | Exemples |
| --- | --- |
| Passes | Passes tentées/réussies par 90 minutes, précision, longueur moyenne |
| Progression | Passes progressives/90, distance gagnée par les passes réussies/90 |
| Conduite de balle | Courses avec ballon/90, courses progressives/90, distance moyenne |
| Finition | Tirs/90, xG hors penalty/90, xG hors penalty par tir |
| Défense | Pressions, contre-pressions, interceptions, récupérations, dégagements et blocs/90 |
| Positionnement des actions | Hauteur moyenne des pressions, pressions hautes/90 |

Dans notre code, une **passe progressive** est une passe *réussie* qui réduit d'au moins **25 %** la distance à la cage adverse. Une **course progressive** applique un principe similaire. Ce sont les **définitions retenues dans ce projet**, et non des standards universels.

Les variables propres à chaque rôle sont définies dans `config/role_feature_sets.json`. Nous conservons uniquement les joueurs ayant au moins **900 minutes enregistrées** pour limiter l'instabilité des statistiques calculées sur de très petits échantillons.

### Exemple : Wes Morgan, Premier League 2015/16

Lors d'une recherche avec `player_id=3813` et `competition_id=2`, les méthodes ont notamment proposé :

| Méthode | Exemples de joueurs proches |
| --- | --- |
| Distance euclidienne | Robert Huth (Premier League), Kyriakos Papadopoulos (Bundesliga), Alessandro Gamberini (Serie A) |
| UMAP + NearestNeighbors | Guglielmo Stendardo (Serie A), Kyriakos Papadopoulos (Bundesliga), Brou Benjamin Angoua (Ligue 1) |

La différence entre les deux classements illustre l'effet de la représentation non linéaire. Ce sont des **profils statistiquement proches selon les variables choisies**, pas une démonstration que leur valeur sportive ou leur style complet sont identiques.

### Exécuter le projet

Depuis la racine du dépôt, avec un environnement Python contenant les dépendances nécessaires (`pandas`, `numpy`, `duckdb`, `pyarrow`, `statsbombpy`, `scikit-learn`, `umap-learn`, `matplotlib`, `mplsoccer` et Jupyter) :

```bash
python -m src.extract_matches
python -m src.extract_events
python -m src.extract_lineups
python -m src.build_player_tables
python -m src.build_features
```

Les scripts d'extraction réutilisent les fichiers déjà présents lorsqu'ils existent. Les nouvelles tables sont sauvegardées dans `data/processed/` sous les noms `player_match_big5.parquet`, `player_season_big5.parquet` et `player_features_big5.parquet` pour préserver les fichiers Premier League initiaux.

Pour rechercher les voisins UMAP :

```python
from src.similarity import (
    build_role_matrices,
    build_umap_models,
    find_similar_players_umap,
)

matrices = build_role_matrices()
models = build_umap_models(matrices)

neighbors = find_similar_players_umap(
    player_id=3813,
    competition_id=2,
    season_id=27,
    matrices=matrices,
    models=models,
    top_k=10,
)
print(neighbors)
```

Le notebook **`notebooks/05_similarity_demo_big5.ipynb`** permet de choisir un joueur par son nom, de consulter ses dix voisins et d'explorer les statistiques, heatmaps et cartes de terrain. Le projet ne nécessite pas d'application web déployée.

### Structure du projet

```text
config/
  role_feature_sets.json
src/
  extract_matches.py
  extract_events.py
  extract_lineups.py
  events.py
  lineups.py
  build_player_tables.py
  features.py
  build_features.py
  similarity.py
notebooks/
  02_data_engineering.ipynb
  05_similarity_demo_big5.ipynb
data/
  raw/
  interim/player_minutes/
  processed/events/
  processed/player_match_big5.parquet
  processed/player_season_big5.parquet
  processed/player_features_big5.parquet
```

### Validation et limites

- **Vérifications réalisées dans le pipeline :** unicité des lignes joueur-match, couverture des matchs, cohérence des totaux agrégés, minutes positives, postes reconnus et valeurs de similarité finies.
- **Évaluation à poursuivre :** la comparaison entre les méthodes euclidienne et UMAP est exploratoire. Il reste à évaluer systématiquement la stabilité des voisins lorsque les variables et les paramètres UMAP changent, ainsi que la pertinence footballistique des recommandations.
- **Couverture inégale :** seuls 17 profils de Bundesliga atteignent notre seuil de 900 minutes, ce qui limite fortement leur représentativité.
- **Influence du contexte :** possession, système tactique, adversaires et styles de championnat influencent les statistiques. Une normalisation par 90 minutes ne corrige pas tous ces biais.
- **Données historiques :** ce projet analyse 2015/2016 et ne doit pas être présenté comme un moteur de scouting actualisé.

### Technologies

Python · pandas · NumPy · DuckDB · Parquet / PyArrow · StatsBombPy · scikit-learn · UMAP (`umap-learn`) · Jupyter · Matplotlib · mplsoccer

### Attribution des données

Données : [StatsBomb Open Data](https://github.com/statsbomb/open-data). Les analyses doivent mentionner StatsBomb et respecter les conditions d'attribution et de licence du dépôt.
