# Football Player Similarity Engine

A role-aware player-profile project built from StatsBomb Open Data for the
2015/16 English Premier League season.

The project currently focuses on producing a clean and reproducible feature
dataset. The similarity model itself has not been implemented yet.

## Project status

| Phase | Description | Status |
| --- | --- | --- |
| 1 | Data exploration | Completed |
| 2 | Data engineering | Completed and validated |
| 3 | Feature engineering | Completed and tested |
| 4 | Player similarity | Next step |

The validated feature-engineering output contains:

- 550 player-season rows;
- 327 players eligible for similarity comparisons;
- a minimum playing-time threshold of 900 minutes;
- 8 broad tactical roles;
- 70 dataset columns;
- role-specific feature sets with no missing values or constant variables for
  eligible players.

## Objective

The long-term objective is to identify players with similar statistical
profiles while respecting their tactical roles.

Comparing every player with the same variables would create misleading
results. A centre-back, a winger and a striker do not perform the same tasks.
This project therefore assigns players to broad roles and selects a relevant
feature set for each role before any similarity calculation is attempted.

## Data source

The project uses [StatsBomb Open Data](https://github.com/statsbomb/open-data)
for Premier League competition ID `2`, season ID `27` (2015/16).

Raw events and lineups are downloaded locally with `statsbombpy`. Large raw
and match-level generated files are excluded from Git. Small validated tables
and the final player feature table are kept in `data/processed/`.

## Feature engineering

The current feature table combines season totals with event-level spatial
information.

### Base features

- activity rates per 90 minutes;
- pass and dribble completion rates;
- counterpressure share;
- playing-time eligibility;
- broad tactical role.

### Event-level features

- pass length and forward progression;
- progressive passes;
- key passes, crosses, switches and through balls;
- carry length and progressive carries;
- non-penalty shooting volume and average non-penalty xG per shot;
- pressure height and pressures in the attacking third.

The final variables used for each role are stored in
`config/role_feature_sets.json`. Highly redundant variables were reviewed
within each role before the configuration was finalized.

## Role groups

The detailed StatsBomb positions are grouped into eight broader comparison
roles:

- goalkeeper;
- centre-back;
- full-back;
- defensive midfielder;
- central midfielder;
- wide player;
- attacking midfielder;
- forward.

The original StatsBomb position is preserved in the dataset.

## Project structure

```text
football-player-similarity/
├── config/
│   └── role_feature_sets.json
├── data/
│   ├── interim/
│   ├── processed/
│   └── raw/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_engineering.ipynb
│   └── 03_feature_engineering.ipynb
├── src/
│   ├── build_features.py
│   ├── events.py
│   ├── extract_events.py
│   ├── extract_lineups.py
│   ├── features.py
│   └── lineups.py
├── tests/
│   └── test_features.py
├── README.md
└── requirements.txt
```

The notebooks document the reasoning and exploratory decisions. The modules
under `src/` contain the reusable implementation:

- `features.py` defines the feature calculations;
- `build_features.py` loads the inputs, runs the calculations, validates the
  result and writes the final Parquet file.

## Installation

Python 3.11 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

## Reproducing the feature dataset

Run commands from the project root.

The event extractor processes every match listed in
`data/processed/matches.parquet`:

```bash
python -m src.extract_events
```

For a short extraction check, use a limit:

```bash
python -m src.extract_events --limit 5
```

The lineup pipeline can then rebuild the match-level playing-time files:

```bash
python -m src.extract_lineups
```

After the processed event files are available, rebuild the final feature
table with:

```bash
python -m src.build_features
```

The output is written to:

```text
data/processed/player_features.parquet
```

## Tests

Run the automated feature-definition tests from the project root:

```bash
python -m pytest -q
```

The current test suite covers:

- rates with zero denominators;
- per-90 calculations and role assignment;
- rejection of unknown positions;
- passing features;
- carrying features;
- shooting features;
- spatial pressing features.

## Current limitations

- The analysis covers only one competition season.
- The roles are intentionally broad and are based on each player's primary
  position.
- Goalkeeper comparisons currently focus mainly on distribution because
  goalkeeper-specific shot-stopping features have not yet been engineered.
- Team possession and tactical context are not yet used to adjust activity
  rates.
- No scaling, distance metric or player-ranking model has been implemented.

## Next step

Phase 4 will prepare role-specific feature matrices, scale their variables and
evaluate similarity measures. Modeling will begin only after the feature
inputs and validation rules are fixed.
