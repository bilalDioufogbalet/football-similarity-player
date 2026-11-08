"""Event transformation utilities for StatsBomb data."""

import pandas as pd


CORE_EVENT_COLUMNS = [
    "match_id",
    "id",
    "index",
    "period",
    "timestamp",
    "minute",
    "second",
    "possession",
    "possession_team_id",
    "possession_team",
    "play_pattern",
    "team_id",
    "team",
    "player_id",
    "player",
    "position",
    "type",
    "location",
    "duration",
    "under_pressure",
    "counterpress",
]


PASS_EVENT_COLUMNS = [
    "pass_recipient_id",
    "pass_recipient",
    "pass_end_location",
    "pass_length",
    "pass_angle",
    "pass_height",
    "pass_body_part",
    "pass_type",
    "pass_outcome",
    "pass_technique",
    "pass_cross",
    "pass_cut_back",
    "pass_switch",
    "pass_through_ball",
    "pass_shot_assist",
    "pass_goal_assist",
    "pass_assisted_shot_id",
    "pass_aerial_won",
]


SHOT_EVENT_COLUMNS = [
    "shot_statsbomb_xg",
    "shot_outcome",
    "shot_type",
    "shot_body_part",
    "shot_technique",
    "shot_end_location",
    "shot_first_time",
    "shot_one_on_one",
    "shot_open_goal",
    "shot_deflected",
    "shot_key_pass_id",
]


MOVEMENT_EVENT_COLUMNS = [
    "carry_end_location",
    "dribble_outcome",
    "dribble_overrun",
    "dribble_nutmeg",
]


DEFENSIVE_EVENT_COLUMNS = [
    "duel_type",
    "duel_outcome",
    "interception_outcome",
    "ball_recovery_recovery_failure",
    "ball_recovery_offensive",
    "clearance_aerial_won",
    "clearance_body_part",
    "block_deflection",
]


EVENT_COLUMNS = list(dict.fromkeys(
    CORE_EVENT_COLUMNS
    + PASS_EVENT_COLUMNS
    + SHOT_EVENT_COLUMNS
    + MOVEMENT_EVENT_COLUMNS
    + DEFENSIVE_EVENT_COLUMNS
))


def prepare_events(
    events_raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Transform raw StatsBomb events into the processed
    event schema used by the similarity engine.
    """

    events = (
        events_raw
        .reindex(columns=EVENT_COLUMNS)
        .copy()
    )

    # Passing indicators
    events["is_pass"] = events["type"].eq("Pass")

    events["pass_completed"] = (
        events["is_pass"]
        & events["pass_outcome"].isna()
    )

    # Shooting indicators
    events["is_shot"] = events["type"].eq("Shot")

    events["is_goal"] = (
        events["is_shot"]
        & events["shot_outcome"].eq("Goal")
    )

    events["is_penalty"] = (
        events["is_shot"]
        & events["shot_type"].eq("Penalty")
    )

    events["xg"] = (
        events["shot_statsbomb_xg"]
        .fillna(0.0)
    )

    events["non_penalty_xg"] = (
        events["xg"]
        .where(
            ~events["is_penalty"],
            0.0
        )
    )

    # Ball movement indicators
    events["is_carry"] = events["type"].eq("Carry")
    events["is_dribble"] = events["type"].eq("Dribble")

    events["dribble_completed"] = (
        events["is_dribble"]
        & events["dribble_outcome"].eq("Complete")
    )

    # Defensive indicators
    events["is_pressure"] = (
        events["type"].eq("Pressure")
    )

    events["is_counterpress"] = (
        events["is_pressure"]
        & events["counterpress"].eq(True)
    )

    events["is_interception"] = (
        events["type"].eq("Interception")
    )

    events["is_ball_recovery"] = (
        events["type"].eq("Ball Recovery")
    )

    events["is_clearance"] = (
        events["type"].eq("Clearance")
    )

    events["is_block"] = (
        events["type"].eq("Block")
    )

    return events