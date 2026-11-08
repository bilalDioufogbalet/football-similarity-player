"""Build the final player feature-engineering dataset."""

import json
from pathlib import Path

import pandas as pd

from src.features import build_player_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PLAYER_SEASON_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "player_season.parquet"
)

EVENTS_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "events"
)

ROLE_FEATURE_SETS_PATH = (
    PROJECT_ROOT
    / "config"
    / "role_feature_sets.json"
)

PLAYER_FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "player_features.parquet"
)


def load_role_feature_sets(
    path: Path,
) -> dict[str, list[str]]:
    """Load and validate the role feature configuration."""

    if not path.exists():
        raise FileNotFoundError(
            f"Role feature configuration not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        feature_sets = json.load(file)

    if not isinstance(feature_sets, dict):
        raise ValueError(
            "Role feature configuration must be "
            "a dictionary."
        )

    for role, features in feature_sets.items():
        if (
            not isinstance(role, str)
            or not isinstance(features, list)
            or not all(
                isinstance(feature, str)
                for feature in features
            )
        ):
            raise ValueError(
                "Invalid role feature configuration."
            )

        if not features:
            raise ValueError(
                f"Role {role!r} has no selected features."
            )

        if len(features) != len(set(features)):
            raise ValueError(
                f"Role {role!r} contains duplicate features."
            )

    return feature_sets


def validate_role_features(
    player_features: pd.DataFrame,
    role_feature_sets: dict[str, list[str]],
) -> None:
    """
    Validate the features used by each comparison role.

    Only players passing the minimum-minute threshold
    are required to have complete similarity features.
    """

    available_roles = set(
        player_features["role"]
        .dropna()
        .unique()
    )

    configured_roles = set(role_feature_sets)

    if available_roles != configured_roles:
        raise ValueError(
            "Role configuration does not match the "
            "roles present in player_features."
        )

    for role, features in role_feature_sets.items():
        missing_columns = (
            set(features)
            - set(player_features.columns)
        )

        if missing_columns:
            raise ValueError(
                f"Missing features for {role}: "
                f"{sorted(missing_columns)}"
            )

        eligible_role_players = player_features.loc[
            (
                player_features[
                    "eligible_for_similarity"
                ]
                & player_features["role"].eq(role)
            ),
            features,
        ]

        if eligible_role_players.empty:
            raise ValueError(
                f"No eligible players for role {role!r}."
            )

        missing_values = (
            eligible_role_players
            .isna()
            .sum()
        )

        missing_values = missing_values.loc[
            missing_values > 0
        ]

        if not missing_values.empty:
            raise ValueError(
                f"Missing similarity values for {role}: "
                f"{missing_values.to_dict()}"
            )

        contains_infinite_values = (
            eligible_role_players
            .isin(
                [
                    float("inf"),
                    float("-inf"),
                ]
            )
            .any()
            .any()
        )

        if contains_infinite_values:
            raise ValueError(
                f"Infinite similarity values for {role}."
            )


def main() -> None:
    """Build, validate and export player features."""

    player_season = pd.read_parquet(
        PLAYER_SEASON_PATH
    )

    role_feature_sets = load_role_feature_sets(
        ROLE_FEATURE_SETS_PATH
    )

    player_features = build_player_features(
        player_season=player_season,
        events_directory=EVENTS_DIRECTORY,
    )

    validate_role_features(
        player_features=player_features,
        role_feature_sets=role_feature_sets,
    )

    PLAYER_FEATURES_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    player_features.to_parquet(
        PLAYER_FEATURES_PATH,
        index=False,
    )

    eligible_players = int(
        player_features[
            "eligible_for_similarity"
        ].sum()
    )

    print("Feature-engineering summary")
    print(f"All players: {len(player_features)}")
    print(f"Eligible players: {eligible_players}")
    print(
        f"Roles: {player_features['role'].nunique()}"
    )
    print(
        f"Dataset columns: "
        f"{len(player_features.columns)}"
    )
    print(f"Output: {PLAYER_FEATURES_PATH}")


if __name__ == "__main__":
    main()