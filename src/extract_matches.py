
"""Build the match catalog for the Big Five leagues, 2015/16."""

from pathlib import Path
from shutil import copy2

import pandas as pd
from statsbombpy import sb

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/processed/matches.parquet"

COMPETITIONS = {
    "Premier League": 2,
    "La Liga": 11,
    "Serie A": 12,
    "Bundesliga": 9,
    "Ligue 1": 7,
}

SEASON_ID = 27




def main():
    """Build the Big Five 2015/16 match catalog."""

    # Load existing Premier League matches
    existing = pd.read_parquet(OUTPUT)
    all_matches = [existing]

    # Retrieve matches from the other four leagues
    for league, competition_id in COMPETITIONS.items():

        if competition_id == 2:
            continue  # Premier League already available

        matches = sb.matches(
            competition_id=competition_id,
            season_id=SEASON_ID,
        )

        matches["competition_id"] = competition_id
        matches["season_id"] = SEASON_ID

        # Validate column compatibility
        missing = set(existing.columns) - set(matches.columns)

        if missing:
            raise ValueError(
                f"{league}: missing columns {missing}"
            )

        all_matches.append(matches[existing.columns].copy())

        print(f"{league}: {len(matches)} matches")

    # Combine matches and remove duplicates
    combined = (
        pd.concat(all_matches, ignore_index=True)
        .drop_duplicates(subset="match_id")
        .reset_index(drop=True)
    )

    # Standardize dates and kick-off times
    combined["match_date"] = (
        pd.to_datetime(
            combined["match_date"].astype("string"),
            format="mixed",
            errors="raise",
        )
        .dt.strftime("%Y-%m-%d")
        .astype("string")
    )

    combined["kick_off"] = (
        combined["kick_off"].astype("string")
    )

    # Validate match identifiers
    if combined["match_id"].isna().any():
        raise ValueError("Missing match IDs detected.")

    # Backup original Premier League data
    backup = OUTPUT.with_name("matches_pl_backup.parquet")

    if not backup.exists():
        copy2(OUTPUT, backup)

    # Export safely to Parquet
    temp = OUTPUT.with_name("matches_updated.tmp.parquet")

    combined.to_parquet(
        temp,
        index=False,
        engine="pyarrow",
    )

    temp.replace(OUTPUT)

    # Display summary
    print("\n=== Match Extraction Summary ===")
    print(f"Total matches: {len(combined)}")
    print(f"New matches: {len(combined) - len(existing)}")

    print("\nMatches per competition:")
    print(
        combined.groupby("competition_id")
        .size()
        .to_string()
    )


if __name__ == "__main__":
    main()
