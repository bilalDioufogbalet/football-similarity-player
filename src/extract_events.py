"""Extract and process StatsBomb events match by match."""

import argparse
from pathlib import Path

import pandas as pd
from statsbombpy import sb

from .events import prepare_events


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MATCHES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "matches.parquet"
)

RAW_EVENTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "events"
)

PROCESSED_EVENTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "events"
)


def extract_one_match(match_id: int) -> tuple[str, int]:
    """Extract, transform, and store one match."""

    raw_path = (
        RAW_EVENTS_DIR
        / f"{match_id}.json"
    )

    processed_path = (
        PROCESSED_EVENTS_DIR
        / f"events_{match_id}.parquet"
    )

    if raw_path.exists() and processed_path.exists():
        return "skipped", 0

    if raw_path.exists():
        events_raw = pd.read_json(
            raw_path,
            orient="records",
            convert_dates=False,
        )
    else:
        events_raw = sb.events(
            match_id=match_id
        )

        events_raw.to_json(
            raw_path,
            orient="records",
            force_ascii=False,
        )

    events_processed = prepare_events(
        events_raw
    )

    events_processed.to_parquet(
        processed_path,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )

    return "processed", len(events_processed)


def main() -> None:
    """Run the event extraction pipeline."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N matches.",
    )

    args = parser.parse_args()

    RAW_EVENTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROCESSED_EVENTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    matches = pd.read_parquet(
        MATCHES_PATH
    )

    matches = matches.sort_values(
        ["match_date", "kick_off"]
    )

    if args.limit is not None:
        matches = matches.head(args.limit)

    match_ids = (
        matches["match_id"]
        .astype(int)
        .tolist()
    )

    failures = []
    processed_matches = 0
    skipped_matches = 0
    total_events = 0

    for position, match_id in enumerate(
        match_ids,
        start=1,
    ):
        print(
            f"[{position}/{len(match_ids)}] "
            f"Match {match_id}"
        )

        try:
            status, event_count = extract_one_match(
                match_id
            )

            if status == "skipped":
                skipped_matches += 1
                print("  Already available — skipped")
            else:
                processed_matches += 1
                total_events += event_count
                print(
                    f"  Saved {event_count} events"
                )

        except Exception as error:
            failures.append(
                (match_id, str(error))
            )

            print(f"  ERROR: {error}")

    print("\nExtraction summary")
    print(f"Processed matches: {processed_matches}")
    print(f"Skipped matches: {skipped_matches}")
    print(f"New events saved: {total_events}")
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