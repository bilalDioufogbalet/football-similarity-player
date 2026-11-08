"""Lineup and playing-time transformations."""

import pandas as pd


def clock_to_seconds(
    value,
    match_end_seconds: int,
) -> int:
    """Convert a minute:second value to seconds."""

    if pd.isna(value):
        return match_end_seconds

    minute_text, second_text = (
        str(value).split(":")
    )

    return (
        int(minute_text) * 60
        + int(second_text)
    )


def prepare_player_minutes(
    lineups_raw: dict[str, pd.DataFrame],
    match_id: int,
    match_end_seconds: int,
) -> pd.DataFrame:
    """
    Build one playing-time row per player
    for a single match.
    """

    lineup_frames = []

    for team_name, team_lineup in (
        lineups_raw.items()
    ):
        team_lineup = team_lineup.copy()

        team_lineup["match_id"] = match_id
        team_lineup["team"] = team_name

        lineup_frames.append(team_lineup)

    lineups = pd.concat(
        lineup_frames,
        ignore_index=True,
    )

    players_used = (
        lineups
        .loc[
            lineups["positions"].str.len() > 0
        ]
        .copy()
    )

    position_intervals = (
        players_used
        .explode(
            "positions",
            ignore_index=True,
        )
    )

    position_details = pd.json_normalize(
        position_intervals["positions"]
    )

    position_intervals = pd.concat(
        [
            position_intervals
            .drop(columns="positions")
            .reset_index(drop=True),

            position_details
            .reset_index(drop=True),
        ],
        axis=1,
    )

    position_intervals["from_seconds"] = (
        position_intervals["from"]
        .apply(
            lambda value: clock_to_seconds(
                value,
                match_end_seconds,
            )
        )
    )

    position_intervals["to_seconds"] = (
        position_intervals["to"]
        .apply(
            lambda value: clock_to_seconds(
                value,
                match_end_seconds,
            )
        )
    )

    position_intervals["interval_seconds"] = (
        position_intervals["to_seconds"]
        - position_intervals["from_seconds"]
    )

    if (
        position_intervals["interval_seconds"] < 0
    ).any():
        raise ValueError(
            "Negative player interval detected."
        )

    position_intervals["interval_minutes"] = (
        position_intervals["interval_seconds"]
        / 60
    )

    player_minutes = (
        position_intervals
        .groupby(
            [
                "match_id",
                "team",
                "player_id",
                "player_name",
                "jersey_number",
            ],
            as_index=False,
        )
        .agg(
            minutes_played=(
                "interval_minutes",
                "sum",
            ),
            started=(
                "from_seconds",
                lambda values: (
                    values == 0
                ).any(),
            ),
        )
    )

    primary_position_indices = (
        position_intervals
        .groupby(
            ["match_id", "player_id"]
        )["interval_seconds"]
        .idxmax()
    )

    primary_positions = (
        position_intervals
        .loc[
            primary_position_indices,
            [
                "match_id",
                "player_id",
                "position_id",
                "position",
            ],
        ]
        .rename(
            columns={
                "position_id": (
                    "primary_position_id"
                ),
                "position": (
                    "primary_position"
                ),
            }
        )
    )

    player_minutes = player_minutes.merge(
        primary_positions,
        on=["match_id", "player_id"],
        how="left",
        validate="one_to_one",
    )

    return (
        player_minutes
        .sort_values(
            ["team", "player_id"]
        )
        .reset_index(drop=True)
    )