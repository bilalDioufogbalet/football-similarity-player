
"""Build player-match and player-season tables for Big Five 2015/16."""

from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MATCHES = ROOT / "data/processed/matches.parquet"
EVENTS_DIR = ROOT / "data/processed/events"
MINUTES_DIR = ROOT / "data/interim/player_minutes"
OUTPUT_DIR = ROOT / "data/processed"

# Event indicators and their corresponding statistics
EVENT_FLAGS = {
    "passes_attempted": "is_pass",
    "passes_completed": "pass_completed",
    "shots": "is_shot",
    "goals": "is_goal",
    "carries": "is_carry",
    "dribbles_attempted": "is_dribble",
    "dribbles_completed": "dribble_completed",
    "pressures": "is_pressure",
    "counterpressures": "is_counterpress",
    "interceptions": "is_interception",
    "ball_recoveries": "is_ball_recovery",
    "clearances": "is_clearance",
    "blocks": "is_block",
}

COUNT_COLUMNS = list(EVENT_FLAGS)
STAT_COLUMNS = COUNT_COLUMNS + ["total_xg", "non_penalty_xg"]



def main():
    """Build Big Five player-match and player-season tables."""

    matches = pd.read_parquet(MATCHES)
    expected_ids = set(matches["match_id"].astype(int))

    # Check that all match files are available
    for directory, prefix in [
        (EVENTS_DIR, "events_"),
        (MINUTES_DIR, "player_minutes_"),
    ]:
        missing = [
            mid for mid in expected_ids
            if not (directory / f"{prefix}{mid}.parquet").exists()
        ]

        if missing:
            raise ValueError(
                f"{len(missing)} files missing in {directory}: "
                f"{missing[:10]}"
            )

    events_path = str(EVENTS_DIR / "*.parquet").replace("'", "''")
    minutes_path = str(MINUTES_DIR / "*.parquet").replace("'", "''")
    matches_path = str(MATCHES).replace("'", "''")

    # Build SQL expressions for event statistics
    event_counts = ",\n".join(
        f"SUM(CASE WHEN COALESCE({flag}, FALSE) "
        f"THEN 1 ELSE 0 END)::INTEGER AS {stat}"
        for stat, flag in EVENT_FLAGS.items()
    )

    match_stats = ",\n".join(
        f"COALESCE(e.{stat}, 0)::INTEGER AS {stat}"
        for stat in COUNT_COLUMNS
    )

    season_stats = ",\n".join(
        f"SUM({stat})::INTEGER AS {stat}"
        for stat in COUNT_COLUMNS
    )

    with duckdb.connect() as con:

        # STEP 1: Build player-match statistics
        player_match = con.execute(f"""
            WITH event_teams AS (
                SELECT DISTINCT
                    match_id,
                    team,
                    team_id

                FROM read_parquet(
                    '{events_path}',
                    union_by_name=true
                )

                WHERE team_id IS NOT NULL
            ),

            event_totals AS (
                SELECT
                    match_id,
                    player_id,

                    {event_counts},

                    SUM(COALESCE(xg, 0)) AS total_xg,

                    SUM(COALESCE(non_penalty_xg, 0))
                        AS non_penalty_xg

                FROM read_parquet(
                    '{events_path}',
                    union_by_name=true
                )

                WHERE player_id IS NOT NULL

                GROUP BY
                    match_id,
                    player_id
            )

            SELECT
                pm.match_id,
                m.competition_id,
                m.season_id,
                m.match_date,

                et.team_id AS team_id,

                pm.team,
                pm.player_id,
                pm.player_name,
                pm.jersey_number,
                pm.started,
                pm.minutes_played,
                pm.primary_position_id,
                pm.primary_position,

                {match_stats},

                COALESCE(e.total_xg, 0.0)
                    AS total_xg,

                COALESCE(e.non_penalty_xg, 0.0)
                    AS non_penalty_xg

            FROM read_parquet(
                '{minutes_path}',
                union_by_name=true
            ) AS pm

            INNER JOIN read_parquet(
                '{matches_path}'
            ) AS m
                ON pm.match_id = m.match_id

            LEFT JOIN event_teams AS et
                ON pm.match_id = et.match_id
                AND pm.team = et.team

            LEFT JOIN event_totals AS e
                ON pm.match_id = e.match_id
                AND pm.player_id = e.player_id

            ORDER BY
                pm.match_id,
                pm.player_id
        """).df()

        # STEP 2: Validate player-match table
        if player_match.duplicated(
            ["match_id", "player_id"]
        ).any():
            raise ValueError("Duplicate player-match rows.")

        if set(player_match["match_id"]) != expected_ids:
            raise ValueError("Player-match coverage is incomplete.")

        if player_match["team_id"].isna().any():
            unmatched = (
                player_match.loc[
                    player_match["team_id"].isna(),
                    ["match_id", "team"]
                ]
                .drop_duplicates()
            )

            print("\nUnmatched teams:")
            print(unmatched.head(20).to_string(index=False))

            raise ValueError("Unmatched team IDs detected.")

        if not player_match["minutes_played"].gt(0).all():
            raise ValueError("Invalid playing minutes.")

        # STEP 3: Aggregate statistics by player and league
        con.register("player_match_data", player_match)

        player_season = con.execute(f"""
            WITH season_totals AS (
                SELECT
                    competition_id,
                    season_id,
                    player_id,

                    ANY_VALUE(player_name) AS player_name,

                    COUNT(*)::INTEGER AS appearances,

                    SUM(
                        CASE WHEN started THEN 1 ELSE 0 END
                    )::INTEGER AS starts,

                    SUM(minutes_played) AS minutes_played,

                    {season_stats},

                    SUM(total_xg) AS total_xg,

                    SUM(non_penalty_xg)
                        AS non_penalty_xg

                FROM player_match_data

                GROUP BY
                    competition_id,
                    season_id,
                    player_id
            ),

            position_totals AS (
                SELECT
                    competition_id,
                    season_id,
                    player_id,
                    primary_position_id,
                    primary_position,

                    SUM(minutes_played)
                        AS position_minutes

                FROM player_match_data

                GROUP BY
                    competition_id,
                    season_id,
                    player_id,
                    primary_position_id,
                    primary_position
            ),

            ranked_positions AS (
                SELECT
                    *,

                    ROW_NUMBER() OVER (
                        PARTITION BY
                            competition_id,
                            season_id,
                            player_id

                        ORDER BY
                            position_minutes DESC,
                            primary_position
                    ) AS position_rank

                FROM position_totals
            )

            SELECT
                s.competition_id,
                s.season_id,
                s.player_id,
                s.player_name,

                p.primary_position_id,
                p.primary_position,

                s.appearances,
                s.starts,
                s.minutes_played,

                {", ".join("s." + col for col in STAT_COLUMNS)}

            FROM season_totals AS s

            JOIN ranked_positions AS p
                ON s.competition_id = p.competition_id
                AND s.season_id = p.season_id
                AND s.player_id = p.player_id
                AND p.position_rank = 1

            ORDER BY
                s.competition_id,
                s.player_id
        """).df()

    # STEP 4: Validate player-season table
    season_key = [
        "competition_id",
        "season_id",
        "player_id",
    ]

    if player_season.duplicated(season_key).any():
        raise ValueError("Duplicate player-season rows.")

    if player_season["primary_position_id"].isna().any():
        raise ValueError("Missing primary positions.")

    for stat in ["minutes_played"] + STAT_COLUMNS:
        difference = abs(
            player_match[stat].sum()
            - player_season[stat].sum()
        )

        if difference > 1e-4:
            raise ValueError(
                f"Aggregation mismatch: {stat}"
            )

    # STEP 5: Save new datasets
    match_output = OUTPUT_DIR / "player_match_big5.parquet"
    season_output = OUTPUT_DIR / "player_season_big5.parquet"

    player_match.to_parquet(
        match_output,
        index=False,
    )

    player_season.to_parquet(
        season_output,
        index=False,
    )

    # STEP 6: Display extraction summary
    print("\n=== Big Five Player Tables ===")

    print("Player-match:", player_match.shape)
    print("Player-season:", player_season.shape)
    print("Matches:", player_match["match_id"].nunique())

    print("\nPlayers per competition:")
    print(
        player_season.groupby("competition_id")
        .size()
        .to_string()
    )


if __name__ == "__main__":
    main()
