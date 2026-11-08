"""Extract lineups and calculate player minutes."""

import argparse
import json
from pathlib import Path

import pandas as pd
from statsbombpy import sb

from .lineups import prepare_player_minutes


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MATCHES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "matches.parquet"
)

PROCESSED_EVENTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "events"
)

RAW_LINEUPS_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "lineups"
)

PLAYER_MINUTES_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "player_minutes"
)


def save_raw_lineups(
    lineups_raw: dict,
    raw_path: Path,
) -> None:
    """Store the lineup dictionary as JSON."""

    payload = {}

    for team_name, team_lineup in (
        lineups_raw.items()
    ):
        payload[team_name] = json.loads(
            team_lineup.to_json(
                orient="records",
                force_ascii=False,
            )
        )

    raw_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def load_raw_lineups(
    raw_path: Path,
) -> dict[str, pd.DataFrame]:
    """Load the raw lineup JSON."""

    payload = json.loads(
        raw_path.read_text(
            encoding="utf-8"
        )
    )

    return {
        team_name: pd.DataFrame(players)
        for team_name, players in payload.items()
    }


def get_match_end_seconds(
    match_id: int,
) -> int:
    """Get the final event time for one match."""

    events_path = (
        PROCESSED_EVENTS_DIR
        / f"events_{match_id}.parquet"
    )

    events_time = pd.read_parquet(
        events_path,
        columns=["minute", "second"],
    )

    event_seconds = (
        events_time["minute"] * 60
        + events_time["second"]
    )

    return int(event_seconds.max())


def extract_one_lineup(
    match_id: int,
) -> tuple[str, int]:
    """Extract and process one match lineup."""

    raw_path = (
        RAW_LINEUPS_DIR
        / f"{match_id}.json"
    )

    player_minutes_path = (
        PLAYER_MINUTES_DIR
        / f"player_minutes_{match_id}.parquet"
    )

    if (
        raw_path.exists()
        and player_minutes_path.exists()
    ):
        return "skipped", 0

    if raw_path.exists():
        lineups_raw = load_raw_lineups(
            raw_path
        )
    else:
        lineups_raw = sb.lineups(
            match_id=match_id
        )

        save_raw_lineups(
            lineups_raw,
            raw_path,
        )

    match_end_seconds = (
        get_match_end_seconds(match_id)
    )

    player_minutes = prepare_player_minutes(
        lineups_raw=lineups_raw,
        match_id=match_id,
        match_end_seconds=match_end_seconds,
    )

    player_minutes.to_parquet(
        player_minutes_path,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )

    return "processed", len(player_minutes)


def main() -> None:
    """Run the lineup extraction pipeline."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N matches.",
    )

    args = parser.parse_args()

    RAW_LINEUPS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PLAYER_MINUTES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    matches = pd.read_parquet(
        MATCHES_PATH
    ).sort_values(
        ["match_date", "kick_off"]
    )

    if args.limit is not None:
        matches = matches.head(args.limit)

    match_ids = (
        matches["match_id"]
        .astype(int)
        .tolist()
    )

    processed_matches = 0
    skipped_matches = 0
    player_rows = 0
    failures = []

    for position, match_id in enumerate(
        match_ids,
        start=1,
    ):
        print(
            f"[{position}/{len(match_ids)}] "
            f"Match {match_id}"
        )

        try:
            status, row_count = (
                extract_one_lineup(match_id)
            )

            if status == "skipped":
                skipped_matches += 1
                print(
                    "  Already available — skipped"
                )
            else:
                processed_matches += 1
                player_rows += row_count
                print(
                    f"  Saved {row_count} "
                    f"player-minute rows"
                )

        except Exception as error:
            failures.append(
                (match_id, str(error))
            )

            print(f"  ERROR: {error}")

    print("\nLineup extraction summary")
    print(
        f"Processed matches: "
        f"{processed_matches}"
    )
    print(
        f"Skipped matches: "
        f"{skipped_matches}"
    )
    print(
        f"New player rows: {player_rows}"
    )
    print(f"Failures: {len(failures)}")

    if failures:
        print("\nFailed matches:")

        for match_id, message in failures:
            print(f"- {match_id}: {message}")

        raise RuntimeError(
            f"{len(failures)} matches failed."
        )


if __name__ == "__main__":
    main()