import os

import numpy as np
import pandas as pd
import pytest

from model_training.preprocess_data import process_json


@pytest.fixture
def sample_match():
    """
    Returns the file path for a single JSON match file to be tested.
    """
    return "tests/test_data/211028.json"


@pytest.fixture
def sample_ten_matches():
    """
    Returns a list of JSON match file paths for testing multiple matches.
    """
    return [
        "tests/test_data/211028.json",
        "tests/test_data/211048.json",
        "tests/test_data/222678.json",
        "tests/test_data/225263.json",
        "tests/test_data/225271.json",
    ]


@pytest.fixture
def expected_first_24_rows():
    """
    Returns a DataFrame with the expected first 24 rows of the processed data for match ID 211028.
    
    Columns:
        - matchid: string ID representing the match.
        - innings: innings number (1 or 2).
        - over: the over number (0-based in the data).
        - ball: the ball number within each over (1-based).
        - runs: cumulative runs scored up to that ball.
        - wickets: cumulative wickets fallen up to that ball.
        - chasing_team_won: binary indicator (0 or 1) if the chasing team eventually won.
        - total_chasing: total runs the chasing team needed (NaN until innings is finalized).
    """
    data = {
        "matchid": ["211028"] * 24,
        "innings": [1] * 24,
        "over": [
            0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2, 2,
            3, 3, 3, 3, 3, 3
        ],
        "ball": list(range(1, 25)),
        "runs": [
            0, 1, 1, 1, 1, 4,
            4, 4, 5, 5, 5, 6,
            10, 11, 11, 15, 19, 20,
            20, 24, 28, 28, 28, 29
        ],
        "wickets": [
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 1
        ],
        "chasing_team_won": [0] * 24,
        "total_chasing": [np.nan] * 24,
    }
    return pd.DataFrame(data)


@pytest.fixture
def expected_chasing_outcomes():
    """
    Returns a dictionary mapping match IDs to expected 'chasing_team_won' and 'total_chasing' values.
    """
    return {
        "211028": {"chasing_team_won": 0, "total_chasing": 179},
        "211048": {"chasing_team_won": 0, "total_chasing": 214},
        "222678": {"chasing_team_won": 1, "total_chasing": 133},
        "225263": {"chasing_team_won": 1, "total_chasing": 144},
        "225271": {"chasing_team_won": 0, "total_chasing": 163},
    }


def test_first_24_rows(sample_match, expected_first_24_rows):
    """
    Test that the first 24 rows of the processed DataFrame match the expected DataFrame
    for match ID 211028.
    """
    folder = os.path.dirname(sample_match)
    filename = os.path.basename(sample_match)

    result_df = process_json(folder, filename)
    result_first_24 = result_df.head(24).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        result_first_24,
        expected_first_24_rows,
        check_dtype=False,
        check_like=True
    )


def test_chasing_team_outcomes(sample_ten_matches, expected_chasing_outcomes):
    """
    For each of the ten match files, verify that the 'chasing_team_won' and 'total_chasing'
    columns have exactly one unique value, and that value matches the expected outcomes.
    """
    for file_path in sample_ten_matches:
        folder = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        match_id = filename.split(".")[0]

        result_df = process_json(folder, filename)

        # We expect exactly one unique value for each of these columns in the final DataFrame
        chasing_team_won_values = result_df["chasing_team_won"].dropna().unique()
        total_chasing_values = result_df["total_chasing"].dropna().unique()

        expected_outcome = expected_chasing_outcomes[match_id]

        # Check the 'chasing_team_won' column
        assert len(chasing_team_won_values) == 1, (
            f"Multiple 'chasing_team_won' values found in match {match_id}"
        )
        assert chasing_team_won_values[0] == expected_outcome["chasing_team_won"], (
            f"Incorrect 'chasing_team_won' for match {match_id}"
        )

        # Check the 'total_chasing' column
        assert len(total_chasing_values) == 1, (
            f"Multiple 'total_chasing' values found in match {match_id}"
        )
        assert total_chasing_values[0] == expected_outcome["total_chasing"], (
            f"Incorrect 'total_chasing' for match {match_id}"
        )