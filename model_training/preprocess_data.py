import json
import os

import numpy as np
import pandas as pd

# Columns for the processed match DataFrame
COLUMNS = [
    "matchid",
    "innings",
    "over",
    "ball",
    "runs",
    "wickets",
    "chasing_team_won",
    "total_chasing",
]


def process_json(folder: str, filename: str) -> pd.DataFrame:
    """
    Read a JSON file from the specified folder, parse cricket match data, and return a
    DataFrame containing runs, wickets, and chasing information.

    Parameters
    ----------
    folder : str
        The directory where the JSON file is located.
    filename : str
        The name of the JSON file to process.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
            - matchid: (str) the match ID (extracted from the filename).
            - innings: (int) which innings (1 or 2).
            - over: (int) zero-based over index.
            - ball: (int) the cumulative number of legal deliveries bowled in that innings.
            - runs: (int) the cumulative runs for that innings at that ball.
            - wickets: (int) the cumulative wickets in that innings at that ball.
            - chasing_team_won: (0, 1, or None) indicates whether the chasing team eventually won the match:
                0 if first-innings team won, 1 if second-innings team won, None if tie/no result.
            - total_chasing: (float) the total runs set by the first innings, NaN for innings=1.
    """
    file_path = os.path.join(folder, filename)
    with open(file_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    # Initialize a DataFrame with the expected columns
    match_df = pd.DataFrame(columns=COLUMNS)

    # Extract match ID from the filename
    match_id = filename.split(".")[0]

    # Track runs and wickets for each of the two innings
    total_runs = [0, 0]
    total_wickets = [0, 0]

    # Only process if we have exactly two innings
    if len(data.get("innings", [])) == 2:
        # Loop over each innings index
        for innings_index in range(2):
            ball_count = 0
            extra_runs = 0  # holds any wide/noball runs that do not count towards legal deliveries

            for over_index, over_data in enumerate(data["innings"][innings_index]["overs"]):
                for delivery in over_data["deliveries"]:
                    runs = delivery["runs"]["total"]

                    # If the delivery includes extras (e.g., wide, no-ball), 
                    # accumulate them into extra_runs without incrementing ball count.
                    if "extras" in delivery and (
                        "wides" in delivery["extras"] or "noballs" in delivery["extras"]
                    ):
                        extra_runs += runs
                    else:
                        # Add both the current delivery runs and any leftover extras.
                        total_runs[innings_index] += runs + extra_runs

                        # Check if a wicket occurred on this delivery
                        wicket_count = 0
                        if "wickets" in delivery:
                            wicket_count = len(delivery["wickets"])
                            total_wickets[innings_index] += wicket_count

                        # Now that it's a legal delivery, increment the ball count
                        ball_count += 1

                        # Append the new row to the match_df
                        match_df = pd.concat(
                            [
                                match_df,
                                pd.DataFrame(
                                    {
                                        "matchid": [match_id],
                                        "innings": [innings_index + 1],
                                        "over": [over_index],
                                        "ball": [ball_count],
                                        "runs": [total_runs[innings_index]],
                                        "wickets": [total_wickets[innings_index]],
                                        "chasing_team_won": [None],
                                        "total_chasing": [None],
                                    }
                                ),
                            ],
                            ignore_index=True,
                        )

                        # Reset extra_runs after counting them
                        extra_runs = 0

            # If extra_runs remain at the end of an innings, add them to the final row's total
            if extra_runs > 0:
                total_runs[innings_index] += extra_runs
                last_row_idx = match_df[match_df["innings"] == (innings_index + 1)].index[-1]
                match_df.at[last_row_idx, "runs"] += extra_runs
                extra_runs = 0

    # Determine which team ended up winning
    # If first-innings runs > second-innings runs => chasing_team_won = 0
    # If second-innings runs > first-innings runs => chasing_team_won = 1
    # If tie => None
    match_df["chasing_team_won"] = np.where(
        total_runs[0] > total_runs[1],
        0,
        np.where(total_runs[1] > total_runs[0], 1, None),
    )

    # The 'total_chasing' column is the runs scored by the first innings, but only for the second innings rows
    match_df["total_chasing"] = np.where(
        match_df["innings"] == 2, total_runs[0], np.nan
    )

    return match_df


def read_and_process_data() -> pd.DataFrame:
    """
    Read all JSON files in the 'historical_data' folder, process each one via `process_json`,
    and concatenate them into a single DataFrame. Filters out matches where the ball count
    exceeds 120 (data errors).

    Returns
    -------
    pd.DataFrame
        DataFrame containing the concatenated data from all valid JSON matches.
        The index is set to 'matchid', but note that some match IDs may appear multiple times
        if the data is stored ball-by-ball.
    """
    folder = "historical_data"
    files = os.listdir(folder)
    match_dfs = []

    for filename in files:
        match_df = process_json(folder, filename)

        # Exclude data with more than 120 legal deliveries in an innings (anomalous data)
        if match_df["ball"].max() <= 120:
            match_dfs.append(match_df)

    # Concatenate all match-level DataFrames
    preprocessed_df = pd.concat(match_dfs, ignore_index=True)
    preprocessed_df.set_index("matchid", inplace=True)

    return preprocessed_df


def main() -> None:
    """
    Main entry point: reads and processes the match data, then writes a CSV to disk.
    """
    preprocessed_df = read_and_process_data()
    preprocessed_df.to_csv("preprocessed_data.csv")


if __name__ == "__main__":
    main()