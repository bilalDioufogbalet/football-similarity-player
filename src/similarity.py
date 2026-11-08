
"""Role-aware player similarity across competitions and seasons."""

import json

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import umap


PROFILE_KEYS = ["competition_id", "season_id", "player_id"]


def build_role_matrices(
    data_path="data/processed/player_features_big5.parquet",
    config_path="config/role_feature_sets.json",
):
    """Build a standardized feature matrix for each playing role."""
    df = pd.read_parquet(data_path)

    with open(config_path, encoding="utf-8") as f:
        feature_sets = json.load(f)

    missing_keys = set(PROFILE_KEYS) - set(df.columns)
    if missing_keys:
        raise ValueError(f"Missing profile columns: {sorted(missing_keys)}")

    eligible = df.loc[df["eligible_for_similarity"]].copy()

    if not set(eligible["role"]).issubset(feature_sets):
        raise ValueError("Some player roles are missing from the configuration.")

    if eligible.duplicated(PROFILE_KEYS).any():
        raise ValueError("Duplicate player profiles across competitions/seasons.")

    matrices = {}

    for role, features in feature_sets.items():
        role_df = eligible.loc[eligible["role"] == role].copy()
        if role_df.empty:
            continue

        # A player profile is identified by league, season, and player ID.
        player_index = role_df.set_index(PROFILE_KEYS).index
        X = role_df[features].to_numpy(dtype=float)

        if not np.isfinite(X).all():
            raise ValueError(f"Invalid feature values in role: {role}")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        matrix = pd.DataFrame(
            X_scaled,
            columns=features,
            index=player_index,
        )
        players = role_df.set_index(PROFILE_KEYS)[
            ["player_name", "minutes_played"]
        ]

        matrices[role] = {
            "players": players,
            "X": matrix,
            "scaler": scaler,
        }

    return matrices


def _locate_player(player_id, matrices, competition_id=None, season_id=None):
    """Resolve a profile, rejecting ambiguous player IDs explicitly."""
    candidates = []

    for role, data in matrices.items():
        index = data["X"].index
        mask = index.get_level_values("player_id") == player_id

        if competition_id is not None:
            mask &= index.get_level_values("competition_id") == competition_id
        if season_id is not None:
            mask &= index.get_level_values("season_id") == season_id

        candidates.extend((role, data, key) for key in index[mask].tolist())

    if not candidates:
        raise ValueError(
            f"Player profile not found: player_id={player_id}, "
            f"competition_id={competition_id}, season_id={season_id}."
        )

    if len(candidates) > 1:
        available = [(key[0], key[1]) for _, _, key in candidates]
        raise ValueError(
            f"Ambiguous player_id={player_id}. "
            f"Available (competition_id, season_id): {available}. "
            "Specify competition_id (and season_id if necessary)."
        )

    return candidates[0]


def find_similar_players(
    player_id, matrices, top_k=10, competition_id=None, season_id=None
):
    """Find same-role neighbors in the standardized feature space."""
    if top_k < 1:
        raise ValueError("top_k must be positive.")

    _, data, player_key = _locate_player(
        player_id, matrices, competition_id, season_id
    )
    X = data["X"]
    player_vector = X.loc[player_key].to_numpy()

    distances = pd.Series(
        np.linalg.norm(X.to_numpy() - player_vector, axis=1),
        index=X.index,
        name="distance",
    )

    nearest = distances.drop(index=[player_key]).nsmallest(top_k)
    results = data["players"].loc[nearest.index].copy()
    results["distance"] = nearest.to_numpy()
    return results.reset_index()


def build_umap_models(matrices):
    """Fit a UMAP embedding and nearest-neighbor index for each role."""
    models = {}

    for role, data in matrices.items():
        X = data["X"]
        if len(X) < 4:
            raise ValueError(f"UMAP requires at least 4 profiles for role {role}.")

        reducer = umap.UMAP(
            n_neighbors=min(10, len(X) - 1),
            n_components=min(5, X.shape[1], len(X) - 2),
            min_dist=0.1,
            metric="euclidean",
            random_state=42,
        )
        embedding = reducer.fit_transform(X.to_numpy())

        neighbor_model = NearestNeighbors(metric="euclidean")
        neighbor_model.fit(embedding)

        models[role] = {
            "reducer": reducer,
            "embedding": embedding,
            "neighbors": neighbor_model,
        }

    return models


def find_similar_players_umap(
    player_id, matrices, models, top_k=10, competition_id=None, season_id=None
):
    """Find same-role neighbors in the UMAP representation."""
    if top_k < 1:
        raise ValueError("top_k must be positive.")

    role, data, player_key = _locate_player(
        player_id, matrices, competition_id, season_id
    )
    X = data["X"]
    embedding = models[role]["embedding"]
    neighbor_model = models[role]["neighbors"]

    position = X.index.get_loc(player_key)
    distances, indices = neighbor_model.kneighbors(
        embedding[position : position + 1],
        n_neighbors=min(top_k + 1, len(X)),
    )

    pairs = [
        (X.index[int(i)], float(distance))
        for distance, i in zip(distances[0], indices[0])
        if X.index[int(i)] != player_key
    ][:top_k]

    keys = [key for key, _ in pairs]
    results = data["players"].loc[keys].copy()
    results["distance"] = [distance for _, distance in pairs]
    return results.reset_index()
