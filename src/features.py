"""Feature-engineering utilities for player similarity."""

from pathlib import Path

import duckdb
import pandas as pd

PROFILE_KEYS = ["competition_id", "season_id", "player_id"]
MINIMUM_MINUTES = 900
ATTACKING_THIRD_X = 80.0
PROGRESSIVE_DISTANCE_RATIO = 0.75
OPPONENT_GOAL_X = 120.0
OPPONENT_GOAL_Y = 40.0


PER90_SOURCE_COLUMNS = [
    "passes_attempted",
    "passes_completed",
    "shots",
    "goals",
    "total_xg",
    "non_penalty_xg",
    "carries",
    "dribbles_attempted",
    "dribbles_completed",
    "pressures",
    "counterpressures",
    "interceptions",
    "ball_recoveries",
    "clearances",
    "blocks",
]


POSITION_TO_ROLE = {
    1: "goalkeeper",

    2: "full_back",
    6: "full_back",
    7: "full_back",
    8: "full_back",

    3: "centre_back",
    4: "centre_back",
    5: "centre_back",

    9: "defensive_midfielder",
    10: "defensive_midfielder",
    11: "defensive_midfielder",

    13: "central_midfielder",
    14: "central_midfielder",
    15: "central_midfielder",

    12: "wide_player",
    16: "wide_player",
    17: "wide_player",
    21: "wide_player",

    18: "attacking_midfielder",
    20: "attacking_midfielder",
    19: "attacking_midfielder",

    22: "forward",
    23: "forward",
    24: "forward",
    25: "forward",
}


def _event_source_sql(
    events_directory: Path,
    matches_path: str | Path | None,
) -> str:
    """Read events and optionally attach competition and season IDs."""

    events_file = (events_directory / "*.parquet").as_posix()
    events_file = events_file.replace("'", "''")
    events_source = (
        f"read_parquet('{events_file}', union_by_name = TRUE)"
    )

    if matches_path is None:
        # Keep the original single-league interface working.
        return events_source

    matches_path = Path(matches_path)
    if not matches_path.exists():
        raise FileNotFoundError(f"Matches file not found: {matches_path}")

    matches_file = matches_path.as_posix().replace("'", "''")
    return f"""(
        SELECT
            events.*,
            matches.competition_id,
            matches.season_id
        FROM {events_source} AS events
        INNER JOIN read_parquet('{matches_file}') AS matches
            ON events.match_id = matches.match_id
    ) AS event_data"""


def _profile_sql_parts(matches_path: str | Path | None):
    """Return profile keys for event aggregation and validations."""
    if matches_path is None:
        return "", "player_id", ["player_id"]
    return (
        "competition_id, season_id,",
        "competition_id, season_id, player_id",
        PROFILE_KEYS,
    )


def calculate_rate(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """
    Calculate a rate while preserving undefined cases.

    A zero denominator produces a missing value instead
    of an artificial zero.
    """

    valid_denominator = denominator.where(
        denominator > 0
    )

    return numerator / valid_denominator


def build_base_player_features(
    player_season: pd.DataFrame,
    minimum_minutes: int = MINIMUM_MINUTES,
) -> pd.DataFrame:
    """
    Create the base season-level player features.

    This function adds:

    - per-90 volume metrics;
    - completion and counterpressure rates;
    - broad player roles;
    - the playing-time eligibility flag.
    """

    required_columns = {
        "player_id",
        "primary_position_id",
        "minutes_played",
        "passes_attempted",
        "passes_completed",
        "dribbles_attempted",
        "dribbles_completed",
        "pressures",
        "counterpressures",
        *PER90_SOURCE_COLUMNS,
    }

    missing_columns = (
        required_columns
        - set(player_season.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing player-season columns: "
            f"{sorted(missing_columns)}"
        )

    # A player can appear in several competitions in the same season.
    keys = (
        PROFILE_KEYS
        if set(PROFILE_KEYS).issubset(player_season.columns)
        else ["player_id"]
    )
    if player_season.duplicated(keys).any():
        raise ValueError(f"Duplicate player-season profiles for {keys}.")

    if (
        player_season["minutes_played"] <= 0
    ).any():
        raise ValueError(
            "All players must have positive minutes."
        )

    player_features = player_season.copy()

    for column in PER90_SOURCE_COLUMNS:
        player_features[f"{column}_per90"] = (
            player_features[column]
            / player_features["minutes_played"]
            * 90
        )

    player_features["pass_completion_rate"] = (
        calculate_rate(
            player_features["passes_completed"],
            player_features["passes_attempted"],
        )
    )

    player_features["dribble_completion_rate"] = (
        calculate_rate(
            player_features["dribbles_completed"],
            player_features["dribbles_attempted"],
        )
    )

    player_features["counterpressure_share"] = (
        calculate_rate(
            player_features["counterpressures"],
            player_features["pressures"],
        )
    )

    player_features["role"] = (
        player_features["primary_position_id"]
        .map(POSITION_TO_ROLE)
    )

    if player_features["role"].isna().any():
        unsupported_positions = sorted(
            player_features.loc[
                player_features["role"].isna(),
                "primary_position_id",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Unsupported primary positions: "
            f"{unsupported_positions}"
        )

    player_features["eligible_for_similarity"] = (
        player_features["minutes_played"]
        >= minimum_minutes
    )

    return player_features




def build_passing_features(
    events_directory: str | Path,
    progressive_distance_ratio: float = (
        PROGRESSIVE_DISTANCE_RATIO
    ),
    *,
    matches_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Aggregate event-level passing features by player.

    A progressive pass is a completed pass that reduces
    the distance to the opponent goal by at least 25%.
    """

    events_directory = Path(events_directory)

    event_files = sorted(
        events_directory.glob("*.parquet")
    )

    if not event_files:
        raise FileNotFoundError(
            "No event Parquet files found in "
            f"{events_directory}."
        )

    if not 0 < progressive_distance_ratio < 1:
        raise ValueError(
            "progressive_distance_ratio must be "
            "between 0 and 1."
        )

    event_source_sql = _event_source_sql(
        events_directory, matches_path
    )
    keys_select_sql, keys_group_sql, profile_keys = (
        _profile_sql_parts(matches_path)
    )

    query = f"""
        WITH passes AS (
            SELECT
                {keys_select_sql}
                CAST(player_id AS BIGINT)
                    AS player_id,

                player
                    AS player_name_events,

                COALESCE(
                    TRY_CAST(
                        pass_completed AS BOOLEAN
                    ),
                    FALSE
                )
                    AS pass_completed,

                TRY_CAST(pass_length AS DOUBLE)
                    AS pass_length,

                location[1]
                    AS start_x,

                location[2]
                    AS start_y,

                pass_end_location[1]
                    AS end_x,

                pass_end_location[2]
                    AS end_y,

                COALESCE(
                    TRY_CAST(
                        pass_shot_assist AS BOOLEAN
                    ),
                    FALSE
                )
                    AS is_shot_assist,

                COALESCE(
                    TRY_CAST(
                        pass_goal_assist AS BOOLEAN
                    ),
                    FALSE
                )
                    AS is_goal_assist,

                COALESCE(
                    TRY_CAST(
                        pass_cross AS BOOLEAN
                    ),
                    FALSE
                )
                    AS is_cross,

                COALESCE(
                    TRY_CAST(
                        pass_switch AS BOOLEAN
                    ),
                    FALSE
                )
                    AS is_switch,

                COALESCE(
                    TRY_CAST(
                        pass_through_ball AS BOOLEAN
                    ),
                    FALSE
                )
                    AS is_through_ball

            FROM {event_source_sql}

            WHERE
                COALESCE(
                    TRY_CAST(is_pass AS BOOLEAN),
                    FALSE
                )

                AND player_id IS NOT NULL
        ),

        pass_distances AS (
            SELECT
                *,

                CASE
                    WHEN
                        start_x IS NOT NULL
                        AND start_y IS NOT NULL
                    THEN SQRT(
                        POWER(
                            {OPPONENT_GOAL_X}
                            - start_x,
                            2
                        )
                        + POWER(
                            {OPPONENT_GOAL_Y}
                            - start_y,
                            2
                        )
                    )
                END
                    AS start_goal_distance,

                CASE
                    WHEN
                        end_x IS NOT NULL
                        AND end_y IS NOT NULL
                    THEN SQRT(
                        POWER(
                            {OPPONENT_GOAL_X}
                            - end_x,
                            2
                        )
                        + POWER(
                            {OPPONENT_GOAL_Y}
                            - end_y,
                            2
                        )
                    )
                END
                    AS end_goal_distance

            FROM passes
        )

        SELECT
            {keys_select_sql}
            player_id,

            ANY_VALUE(player_name_events)
                AS player_name_events,

            COUNT(*)
                AS event_passes_attempted,

            SUM(
                CASE
                    WHEN pass_completed THEN 1
                    ELSE 0
                END
            )
                AS event_passes_completed,

            AVG(pass_length)
                AS average_pass_length,

            AVG(pass_length) FILTER (
                WHERE pass_completed
            )
                AS average_completed_pass_length,

            AVG(end_x - start_x) FILTER (
                WHERE
                    start_x IS NOT NULL
                    AND end_x IS NOT NULL
            )
                AS average_pass_progression,

            AVG(end_x - start_x) FILTER (
                WHERE
                    pass_completed
                    AND start_x IS NOT NULL
                    AND end_x IS NOT NULL
            )
                AS average_completed_pass_progression,

            SUM(
                CASE
                    WHEN
                        pass_completed
                        AND end_x > start_x
                    THEN end_x - start_x
                    ELSE 0.0
                END
            )
                AS completed_forward_distance,

            SUM(
                CASE
                    WHEN
                        pass_completed
                        AND start_goal_distance
                            IS NOT NULL
                        AND end_goal_distance
                            IS NOT NULL
                        AND end_goal_distance
                            <= (
                                {progressive_distance_ratio}
                                * start_goal_distance
                            )
                    THEN 1
                    ELSE 0
                END
            )
                AS progressive_passes,

            SUM(
                CASE
                    WHEN
                        is_shot_assist
                        OR is_goal_assist
                    THEN 1
                    ELSE 0
                END
            )
                AS key_passes,

            SUM(
                CASE WHEN is_cross THEN 1
                ELSE 0 END
            )
                AS crosses,

            SUM(
                CASE WHEN is_switch THEN 1
                ELSE 0 END
            )
                AS switches,

            SUM(
                CASE WHEN is_through_ball THEN 1
                ELSE 0 END
            )
                AS through_balls

        FROM pass_distances

        GROUP BY {keys_group_sql}

        ORDER BY {keys_group_sql}
    """

    connection = duckdb.connect()

    try:
        passing_features = (
            connection
            .execute(query)
            .df()
        )
    finally:
        connection.close()

    if passing_features.duplicated(profile_keys).any():
        raise ValueError(
            "Passing features contain duplicate players."
        )

    if (
        passing_features["event_passes_completed"]
        > passing_features["event_passes_attempted"]
    ).any():
        raise ValueError(
            "Completed passes exceed attempted passes."
        )

    if (
        passing_features["progressive_passes"]
        > passing_features["event_passes_completed"]
    ).any():
        raise ValueError(
            "Progressive passes exceed completed passes."
        )

    return passing_features


def build_carry_features(
    events_directory: str | Path,
    progressive_distance_ratio: float = (
        PROGRESSIVE_DISTANCE_RATIO
    ),
    *,
    matches_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Aggregate spatial ball-carrying features by player.

    A progressive carry reduces the distance to the
    opponent goal by at least 25%.
    """

    events_directory = Path(events_directory)

    event_files = sorted(
        events_directory.glob("*.parquet")
    )

    if not event_files:
        raise FileNotFoundError(
            "No event Parquet files found in "
            f"{events_directory}."
        )

    if not 0 < progressive_distance_ratio < 1:
        raise ValueError(
            "progressive_distance_ratio must be "
            "between 0 and 1."
        )

    event_source_sql = _event_source_sql(
        events_directory, matches_path
    )
    keys_select_sql, keys_group_sql, profile_keys = (
        _profile_sql_parts(matches_path)
    )

    query = f"""
        WITH carries AS (
            SELECT
                {keys_select_sql}
                CAST(player_id AS BIGINT)
                    AS player_id,

                player
                    AS player_name_events,

                location[1]
                    AS start_x,

                location[2]
                    AS start_y,

                carry_end_location[1]
                    AS end_x,

                carry_end_location[2]
                    AS end_y

            FROM {event_source_sql}

            WHERE
                COALESCE(
                    TRY_CAST(is_carry AS BOOLEAN),
                    FALSE
                )

                AND player_id IS NOT NULL
                AND location IS NOT NULL
                AND carry_end_location IS NOT NULL
        ),

        carry_distances AS (
            SELECT
                *,

                SQRT(
                    POWER(end_x - start_x, 2)
                    + POWER(end_y - start_y, 2)
                )
                    AS carry_length,

                end_x - start_x
                    AS carry_progression,

                SQRT(
                    POWER(
                        {OPPONENT_GOAL_X}
                        - start_x,
                        2
                    )
                    + POWER(
                        {OPPONENT_GOAL_Y}
                        - start_y,
                        2
                    )
                )
                    AS start_goal_distance,

                SQRT(
                    POWER(
                        {OPPONENT_GOAL_X}
                        - end_x,
                        2
                    )
                    + POWER(
                        {OPPONENT_GOAL_Y}
                        - end_y,
                        2
                    )
                )
                    AS end_goal_distance

            FROM carries
        )

        SELECT
            {keys_select_sql}
            player_id,

            ANY_VALUE(player_name_events)
                AS player_name_events,

            COUNT(*)
                AS spatial_carries,

            AVG(carry_length)
                AS average_carry_length,

            AVG(carry_progression)
                AS average_carry_progression,

            SUM(
                CASE
                    WHEN
                        end_goal_distance
                        <= (
                            {progressive_distance_ratio}
                            * start_goal_distance
                        )
                    THEN 1
                    ELSE 0
                END
            )
                AS progressive_carries

        FROM carry_distances

        GROUP BY {keys_group_sql}

        ORDER BY {keys_group_sql}
    """

    connection = duckdb.connect()

    try:
        carry_features = (
            connection
            .execute(query)
            .df()
        )
    finally:
        connection.close()

    if carry_features.duplicated(profile_keys).any():
        raise ValueError(
            "Carry features contain duplicate players."
        )

    if (
        carry_features["progressive_carries"]
        > carry_features["spatial_carries"]
    ).any():
        raise ValueError(
            "Progressive carries exceed total carries."
        )

    if (
        carry_features["average_carry_length"] < 0
    ).any():
        raise ValueError(
            "Negative average carry length detected."
        )

    return carry_features



def build_shooting_features(
    events_directory: str | Path,
    *,
    matches_path: str | Path | None = None,
) -> pd.DataFrame:
    """Aggregate non-penalty shooting features by player."""

    events_directory = Path(events_directory)

    event_files = sorted(
        events_directory.glob("*.parquet")
    )

    if not event_files:
        raise FileNotFoundError(
            "No event Parquet files found in "
            f"{events_directory}."
        )

    event_source_sql = _event_source_sql(
        events_directory, matches_path
    )
    keys_select_sql, keys_group_sql, profile_keys = (
        _profile_sql_parts(matches_path)
    )

    query = f"""
        WITH shots AS (
            SELECT
                {keys_select_sql}
                CAST(player_id AS BIGINT)
                    AS player_id,

                player
                    AS player_name_events,

                COALESCE(
                    TRY_CAST(is_penalty AS BOOLEAN),
                    FALSE
                )
                    AS is_penalty,

                COALESCE(
                    TRY_CAST(
                        shot_statsbomb_xg AS DOUBLE
                    ),
                    0.0
                )
                    AS shot_xg

            FROM {event_source_sql}

            WHERE
                COALESCE(
                    TRY_CAST(is_shot AS BOOLEAN),
                    FALSE
                )

                AND player_id IS NOT NULL
        )

        SELECT
            {keys_select_sql}
            player_id,

            ANY_VALUE(player_name_events)
                AS player_name_events,

            COUNT(*)
                AS event_shots,

            SUM(
                CASE
                    WHEN NOT is_penalty THEN 1
                    ELSE 0
                END
            )
                AS non_penalty_shots,

            SUM(
                CASE
                    WHEN NOT is_penalty THEN shot_xg
                    ELSE 0.0
                END
            )
                AS event_non_penalty_xg

        FROM shots

        GROUP BY {keys_group_sql}

        ORDER BY {keys_group_sql}
    """

    connection = duckdb.connect()

    try:
        shooting_features = (
            connection
            .execute(query)
            .df()
        )
    finally:
        connection.close()

    if shooting_features.duplicated(profile_keys).any():
        raise ValueError(
            "Shooting features contain duplicate players."
        )

    if (
        shooting_features["non_penalty_shots"]
        > shooting_features["event_shots"]
    ).any():
        raise ValueError(
            "Non-penalty shots exceed total shots."
        )

    if (
        shooting_features["event_non_penalty_xg"]
        < 0
    ).any():
        raise ValueError(
            "Negative non-penalty xG detected."
        )

    return shooting_features


def build_pressure_features(
    events_directory: str | Path,
    attacking_third_x: float = ATTACKING_THIRD_X,
    *,
    matches_path: str | Path | None = None,
) -> pd.DataFrame:
    """Aggregate spatial pressure features by player."""

    events_directory = Path(events_directory)

    event_files = sorted(
        events_directory.glob("*.parquet")
    )

    if not event_files:
        raise FileNotFoundError(
            "No event Parquet files found in "
            f"{events_directory}."
        )

    if not 0 <= attacking_third_x <= 120:
        raise ValueError(
            "attacking_third_x must be between "
            "0 and 120."
        )

    event_source_sql = _event_source_sql(
        events_directory, matches_path
    )
    keys_select_sql, keys_group_sql, profile_keys = (
        _profile_sql_parts(matches_path)
    )

    query = f"""
        WITH pressures AS (
            SELECT
                {keys_select_sql}
                CAST(player_id AS BIGINT)
                    AS player_id,

                player
                    AS player_name_events,

                location[1]
                    AS pressure_x

            FROM {event_source_sql}

            WHERE
                COALESCE(
                    TRY_CAST(is_pressure AS BOOLEAN),
                    FALSE
                )

                AND player_id IS NOT NULL
                AND location IS NOT NULL
        )

        SELECT
            {keys_select_sql}
            player_id,

            ANY_VALUE(player_name_events)
                AS player_name_events,

            COUNT(*)
                AS spatial_pressures,

            AVG(pressure_x)
                AS average_pressure_height,

            SUM(
                CASE
                    WHEN pressure_x >= {attacking_third_x}
                    THEN 1
                    ELSE 0
                END
            )
                AS high_pressures

        FROM pressures

        GROUP BY {keys_group_sql}

        ORDER BY {keys_group_sql}
    """

    connection = duckdb.connect()

    try:
        pressure_features = (
            connection
            .execute(query)
            .df()
        )
    finally:
        connection.close()

    if pressure_features.duplicated(profile_keys).any():
        raise ValueError(
            "Pressure features contain duplicate players."
        )

    if (
        pressure_features["high_pressures"]
        > pressure_features["spatial_pressures"]
    ).any():
        raise ValueError(
            "High pressures exceed total pressures."
        )

    valid_heights = pressure_features[
        "average_pressure_height"
    ].between(
        0,
        120,
        inclusive="both",
    )

    if not valid_heights.all():
        raise ValueError(
            "Pressure height outside the pitch."
        )

    return pressure_features


def build_player_features(
    player_season: pd.DataFrame,
    events_directory: str | Path,
    minimum_minutes: int = MINIMUM_MINUTES,
    *,
    matches_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Build the complete player feature-engineering table.

    The season-level table is combined with passing,
    carrying, shooting and spatial pressure features.
    """

    if matches_path is not None:
        missing = set(PROFILE_KEYS) - set(player_season.columns)
        if missing:
            raise ValueError(f"Missing profile key columns: {sorted(missing)}")

    profile_keys = (
        PROFILE_KEYS if matches_path is not None else ["player_id"]
    )

    if player_season.duplicated(profile_keys).any():
        raise ValueError(f"Duplicate profile keys: {profile_keys}")

    player_features = build_base_player_features(
        player_season=player_season,
        minimum_minutes=minimum_minutes,
    )

    passing_features = build_passing_features(
        events_directory,
        matches_path=matches_path,
    )

    carry_features = build_carry_features(
        events_directory,
        matches_path=matches_path,
    )

    shooting_features = build_shooting_features(
        events_directory,
        matches_path=matches_path,
    )

    pressure_features = build_pressure_features(
        events_directory,
        matches_path=matches_path,
    )

    # Validate event totals against the phase-2 table.
    expected_passes_attempted = int(
        player_features["passes_attempted"].sum()
    )

    reproduced_passes_attempted = int(
        passing_features[
            "event_passes_attempted"
        ].sum()
    )

    if (
        reproduced_passes_attempted
        != expected_passes_attempted
    ):
        raise ValueError(
            "Event attempted-pass total does not "
            "match player_season."
        )

    expected_passes_completed = int(
        player_features["passes_completed"].sum()
    )

    reproduced_passes_completed = int(
        passing_features[
            "event_passes_completed"
        ].sum()
    )

    if (
        reproduced_passes_completed
        != expected_passes_completed
    ):
        raise ValueError(
            "Event completed-pass total does not "
            "match player_season."
        )

    if int(
        carry_features["spatial_carries"].sum()
    ) != int(
        player_features["carries"].sum()
    ):
        raise ValueError(
            "Event carry total does not match "
            "player_season."
        )

    if int(
        shooting_features["event_shots"].sum()
    ) != int(
        player_features["shots"].sum()
    ):
        raise ValueError(
            "Event shot total does not match "
            "player_season."
        )

    non_penalty_xg_difference = abs(
        shooting_features[
            "event_non_penalty_xg"
        ].sum()
        - player_features[
            "non_penalty_xg"
        ].sum()
    )

    if non_penalty_xg_difference > 1e-6:
        raise ValueError(
            "Event non-penalty xG total does not "
            "match player_season."
        )

    if int(
        pressure_features[
            "spatial_pressures"
        ].sum()
    ) != int(
        player_features["pressures"].sum()
    ):
        raise ValueError(
            "Event pressure total does not match "
            "player_season."
        )

    rows_before_merges = len(player_features)

    player_features = player_features.merge(
        passing_features[
            [
                *profile_keys,
                "average_pass_length",
                "average_completed_pass_length",
                "average_pass_progression",
                "average_completed_pass_progression",
                "completed_forward_distance",
                "progressive_passes",
                "key_passes",
                "crosses",
                "switches",
                "through_balls",
            ]
        ],
        on=profile_keys,
        how="left",
        validate="one_to_one",
    )

    player_features = player_features.merge(
        carry_features[
            [
                *profile_keys,
                "average_carry_length",
                "average_carry_progression",
                "progressive_carries",
            ]
        ],
        on=profile_keys,
        how="left",
        validate="one_to_one",
    )

    player_features = player_features.merge(
        shooting_features[
            [
                *profile_keys,
                "non_penalty_shots",
            ]
        ],
        on=profile_keys,
        how="left",
        validate="one_to_one",
    )

    player_features = player_features.merge(
        pressure_features[
            [
                *profile_keys,
                "average_pressure_height",
                "high_pressures",
            ]
        ],
        on=profile_keys,
        how="left",
        validate="one_to_one",
    )

    if len(player_features) != rows_before_merges:
        raise ValueError(
            "Feature merges changed the player count."
        )

    if player_features.duplicated(profile_keys).any():
        raise ValueError("Duplicate player profiles detected after merges.")

    integer_count_columns = [
        "progressive_passes",
        "key_passes",
        "crosses",
        "switches",
        "through_balls",
        "progressive_carries",
        "non_penalty_shots",
        "high_pressures",
    ]

    player_features[integer_count_columns] = (
        player_features[integer_count_columns]
        .fillna(0)
        .astype("int32")
    )

    player_features["completed_forward_distance"] = (
        player_features[
            "completed_forward_distance"
        ]
        .fillna(0.0)
    )

    advanced_per90_columns = [
        "completed_forward_distance",
        "progressive_passes",
        "key_passes",
        "crosses",
        "switches",
        "through_balls",
        "progressive_carries",
        "non_penalty_shots",
        "high_pressures",
    ]

    for column in advanced_per90_columns:
        player_features[f"{column}_per90"] = (
            player_features[column]
            / player_features["minutes_played"]
            * 90
        )

    player_features[
        "average_non_penalty_xg_per_shot"
    ] = calculate_rate(
        player_features["non_penalty_xg"],
        player_features["non_penalty_shots"],
    )

    # Final logical validations.
    validation_pairs = [
        (
            "progressive_passes",
            "passes_completed",
        ),
        (
            "key_passes",
            "passes_attempted",
        ),
        (
            "crosses",
            "passes_attempted",
        ),
        (
            "switches",
            "passes_attempted",
        ),
        (
            "through_balls",
            "passes_attempted",
        ),
        (
            "progressive_carries",
            "carries",
        ),
        (
            "non_penalty_shots",
            "shots",
        ),
        (
            "high_pressures",
            "pressures",
        ),
    ]

    for subset_column, total_column in validation_pairs:
        if (
            player_features[subset_column]
            > player_features[total_column]
        ).any():
            raise ValueError(
                f"{subset_column} exceeds "
                f"{total_column}."
            )

    return player_features