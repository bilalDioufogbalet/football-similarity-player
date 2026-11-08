"""Tests for player feature engineering."""

import pandas as pd
import pytest

from src.features import (
    PER90_SOURCE_COLUMNS,
    build_base_player_features,
    calculate_rate,
)


def make_player_season_sample() -> pd.DataFrame:
    """Create a small valid player-season dataset."""

    data = {
        "player_id": [1, 2],
        "primary_position_id": [19, 23],
        "minutes_played": [900.0, 450.0],
    }

    for column in PER90_SOURCE_COLUMNS:
        data[column] = [10, 10]

    data["passes_attempted"] = [100, 50]
    data["passes_completed"] = [80, 40]

    data["dribbles_attempted"] = [10, 0]
    data["dribbles_completed"] = [5, 0]

    data["pressures"] = [20, 0]
    data["counterpressures"] = [5, 0]

    return pd.DataFrame(data)


def test_calculate_rate_preserves_undefined_rate():
    """A zero denominator must produce a missing value."""

    result = calculate_rate(
        numerator=pd.Series([8, 0]),
        denominator=pd.Series([10, 0]),
    )

    assert result.iloc[0] == pytest.approx(0.8)
    assert pd.isna(result.iloc[1])


def test_build_base_player_features():
    """Base features must reproduce the validated definitions."""

    player_season = make_player_season_sample()

    result = build_base_player_features(
        player_season
    )

    assert len(result) == 2
    assert result["player_id"].is_unique

    assert result["role"].tolist() == [
        "attacking_midfielder",
        "forward",
    ]

    assert result[
        "eligible_for_similarity"
    ].tolist() == [
        True,
        False,
    ]

    assert result.loc[
        0,
        "passes_attempted_per90",
    ] == pytest.approx(10.0)

    assert result.loc[
        1,
        "passes_attempted_per90",
    ] == pytest.approx(10.0)

    assert result.loc[
        0,
        "pass_completion_rate",
    ] == pytest.approx(0.8)

    assert result.loc[
        0,
        "dribble_completion_rate",
    ] == pytest.approx(0.5)

    assert pd.isna(
        result.loc[
            1,
            "dribble_completion_rate",
        ]
    )

    assert result.loc[
        0,
        "counterpressure_share",
    ] == pytest.approx(0.25)

    assert pd.isna(
        result.loc[
            1,
            "counterpressure_share",
        ]
    )


def test_build_base_features_rejects_unknown_position():
    """An unmapped position must raise an explicit error."""

    player_season = make_player_season_sample()

    player_season.loc[
        0,
        "primary_position_id",
    ] = 999

    with pytest.raises(
        ValueError,
        match="Unsupported primary positions",
    ):
        build_base_player_features(
            player_season
        )




from src.features import (
    PER90_SOURCE_COLUMNS,
    build_base_player_features,
    build_passing_features,
    calculate_rate,
)

def test_build_passing_features(tmp_path):
    """
    Passing features must follow the validated
    event-level definitions.
    """

    events_directory = tmp_path / "events"
    events_directory.mkdir()

    events = pd.DataFrame(
        {
            "player_id": [1, 1, 1, 2],
            "player": [
                "Player A",
                "Player A",
                "Player A",
                "Player B",
            ],
            "is_pass": [
                True,
                True,
                False,
                True,
            ],
            "pass_completed": [
                True,
                False,
                False,
                True,
            ],
            "pass_length": [
                10.0,
                20.0,
                None,
                30.0,
            ],
            "location": [
                [60.0, 40.0],
                [60.0, 40.0],
                [70.0, 40.0],
                [0.0, 40.0],
            ],
            "pass_end_location": [
                [90.0, 40.0],
                [70.0, 40.0],
                None,
                [30.0, 40.0],
            ],
            # Float values reproduce the optional
            # boolean schema found in the real files.
            "pass_shot_assist": [
                1.0,
                None,
                None,
                None,
            ],
            "pass_goal_assist": [
                None,
                None,
                None,
                1.0,
            ],
            "pass_cross": [
                None,
                1.0,
                None,
                None,
            ],
            "pass_switch": [
                None,
                None,
                None,
                1.0,
            ],
            "pass_through_ball": [
                1.0,
                None,
                None,
                None,
            ],
        }
    )

    events.to_parquet(
        events_directory / "match_1.parquet",
        index=False,
    )

    result = build_passing_features(
        events_directory
    ).set_index("player_id")

    player_a = result.loc[1]
    player_b = result.loc[2]

    assert player_a[
        "event_passes_attempted"
    ] == 2

    assert player_a[
        "event_passes_completed"
    ] == 1

    assert player_a[
        "average_pass_length"
    ] == pytest.approx(15.0)

    assert player_a[
        "average_completed_pass_length"
    ] == pytest.approx(10.0)

    assert player_a[
        "average_pass_progression"
    ] == pytest.approx(20.0)

    assert player_a[
        "average_completed_pass_progression"
    ] == pytest.approx(30.0)

    assert player_a[
        "progressive_passes"
    ] == 1

    assert player_a["key_passes"] == 1
    assert player_a["crosses"] == 1
    assert player_a["switches"] == 0
    assert player_a["through_balls"] == 1

    assert player_b[
        "event_passes_attempted"
    ] == 1

    assert player_b[
        "event_passes_completed"
    ] == 1

    assert player_b[
        "progressive_passes"
    ] == 1

    assert player_b["key_passes"] == 1
    assert player_b["switches"] == 1



from src.features import (
    PER90_SOURCE_COLUMNS,
    build_base_player_features,
    build_carry_features,
    build_passing_features,
    calculate_rate,
)


def test_build_carry_features(tmp_path):
    """
    Carry features must use spatial distance and
    the validated progressive-carry definition.
    """

    events_directory = tmp_path / "events"
    events_directory.mkdir()

    events = pd.DataFrame(
        {
            "player_id": [1, 1, 2, 2],
            "player": [
                "Player A",
                "Player A",
                "Player B",
                "Player B",
            ],
            "is_carry": [
                True,
                True,
                True,
                False,
            ],
            "location": [
                [60.0, 40.0],
                [60.0, 40.0],
                [0.0, 40.0],
                [20.0, 20.0],
            ],
            "carry_end_location": [
                [90.0, 40.0],
                [65.0, 50.0],
                [30.0, 40.0],
                None,
            ],
        }
    )

    events.to_parquet(
        events_directory / "match_1.parquet",
        index=False,
    )

    result = build_carry_features(
        events_directory
    ).set_index("player_id")

    player_a = result.loc[1]
    player_b = result.loc[2]

    assert player_a["spatial_carries"] == 2

    assert player_a[
        "average_carry_length"
    ] == pytest.approx(
        (
            30.0
            + (5.0**2 + 10.0**2) ** 0.5
        )
        / 2
    )

    assert player_a[
        "average_carry_progression"
    ] == pytest.approx(17.5)

    assert player_a[
        "progressive_carries"
    ] == 1

    assert player_b["spatial_carries"] == 1

    assert player_b[
        "average_carry_length"
    ] == pytest.approx(30.0)

    assert player_b[
        "average_carry_progression"
    ] == pytest.approx(30.0)

    assert player_b[
        "progressive_carries"
    ] == 1


from src.features import (
    PER90_SOURCE_COLUMNS,
    build_base_player_features,
    build_carry_features,
    build_passing_features,
    build_pressure_features,
    build_shooting_features,
    calculate_rate,
)


def test_build_shooting_features(tmp_path):
    """Penalty shots must be excluded from the profile."""

    events_directory = tmp_path / "events"
    events_directory.mkdir()

    events = pd.DataFrame(
        {
            "player_id": [1, 1, 1, 2, 2],
            "player": [
                "Player A",
                "Player A",
                "Player A",
                "Player B",
                "Player B",
            ],
            "is_shot": [
                True,
                True,
                False,
                True,
                True,
            ],
            "is_penalty": [
                False,
                True,
                False,
                False,
                False,
            ],
            "shot_statsbomb_xg": [
                0.10,
                0.76,
                None,
                0.30,
                None,
            ],
        }
    )

    events.to_parquet(
        events_directory / "match_1.parquet",
        index=False,
    )

    result = build_shooting_features(
        events_directory
    ).set_index("player_id")

    assert result.loc[1, "event_shots"] == 2
    assert result.loc[1, "non_penalty_shots"] == 1

    assert result.loc[
        1,
        "event_non_penalty_xg",
    ] == pytest.approx(0.10)

    assert result.loc[2, "event_shots"] == 2
    assert result.loc[2, "non_penalty_shots"] == 2

    assert result.loc[
        2,
        "event_non_penalty_xg",
    ] == pytest.approx(0.30)


def test_build_pressure_features(tmp_path):
    """The attacking-third boundary must be inclusive."""

    events_directory = tmp_path / "events"
    events_directory.mkdir()

    events = pd.DataFrame(
        {
            "player_id": [1, 1, 1, 2],
            "player": [
                "Player A",
                "Player A",
                "Player A",
                "Player B",
            ],
            "is_pressure": [
                True,
                True,
                False,
                True,
            ],
            "location": [
                [90.0, 40.0],
                [70.0, 30.0],
                [100.0, 40.0],
                [80.0, 20.0],
            ],
        }
    )

    events.to_parquet(
        events_directory / "match_1.parquet",
        index=False,
    )

    result = build_pressure_features(
        events_directory
    ).set_index("player_id")

    assert result.loc[
        1,
        "spatial_pressures",
    ] == 2

    assert result.loc[
        1,
        "average_pressure_height",
    ] == pytest.approx(80.0)

    assert result.loc[
        1,
        "high_pressures",
    ] == 1

    assert result.loc[
        2,
        "average_pressure_height",
    ] == pytest.approx(80.0)

    assert result.loc[
        2,
        "high_pressures",
    ] == 1