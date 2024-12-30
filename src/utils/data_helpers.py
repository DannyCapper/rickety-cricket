import math
import logging
import numpy as np
import random

from decimal import Decimal
from datetime import datetime, date

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def to_decimal(value):
    """
    Convert a numeric value to Decimal for DynamoDB storage.
    Returns None if the value is None, NaN, or infinite.
    """
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return None
    return Decimal(str(value))


def filter_mens_t20(matches):
    """
    Filter a list of matches to only include men's T20 matches 
    that have started but not yet ended.

    Parameters
    ----------
    matches : list of dict
        A list of match dictionaries as returned by an external API.

    Returns
    -------
    list of dict
        Filtered list containing only men's T20 matches in progress.
    """
    filtered = []
    for match in matches:
        match_type = match.get("matchType", "").lower()
        name = match.get("name", "")
        match_started = match.get("matchStarted", False)
        match_ended = match.get("matchEnded", False)

        # Exclude women's matches
        if "women" in name.lower():
            continue

        # Include if T20, started, and not ended
        if match_type == "t20" and match_started and not match_ended:
            filtered.append(match)

    return filtered


def select_random_match(filtered_matches):
    """
    Randomly select a match from a list of filtered matches.

    Parameters
    ----------
    filtered_matches : list of dict
        Filtered list of matches.

    Returns
    -------
    dict or None
        A randomly chosen match dictionary or None if no matches are available.
    """
    if filtered_matches:
        return random.choice(filtered_matches)
    return None


def parse_inning_team_name(inning_str):
    """
    Parse out the team name from a string like "Perth Scorchers Inning 1".

    Examples:
        "Brisbane Heat Inning 1" -> "Brisbane Heat"
        "Perth Scorchers Inning 2" -> "Perth Scorchers"

    Raises
    ------
    ValueError
        If the `inning_str` is empty or None.
    """
    if not inning_str:
        raise ValueError("Cannot parse an empty or None 'inning_str'.")

    return (inning_str
            .replace("Inning 1", "")
            .replace("Inning 2", "")
            .strip())


def extract_batting_order(score_data, all_teams):
    """
    Determine which of the teams in `all_teams` is batting first vs second,
    based on the first innings info in `score_data`. Raises an error if we
    cannot cleanly match a parsed name to the known teams.

    Parameters
    ----------
    score_data : list of dict
        Example:
        [
          {
            "r": 165,
            "w": 6,
            "o": 20,
            "inning": "Perth Scorchers Inning 1"
          },
          {
            "r": 34,
            "w": 3,
            "o": 6.2,
            "inning": "Brisbane Heat Inning 1"
          }
        ]
    all_teams : list of str
        The two teams, e.g. ["Perth Scorchers", "Brisbane Heat"].

    Returns
    -------
    (str, str)
        A tuple: (team_batting_first, team_batting_second).

    Raises
    ------
    ValueError
        If there's insufficient data to determine batting order, 
        or the parsed name doesn't match the known teams.
    """
    if not all_teams or len(all_teams) < 2:
        raise ValueError("Could not parse batting order: 'teams' is missing or incomplete.")

    if not score_data:
        raise ValueError("Could not parse batting order: 'score' data is empty or missing.")

    # Parse the first innings's "inning" field
    first_innings_inning_str = score_data[0].get("inning", "")
    if not first_innings_inning_str:
        raise ValueError("Could not parse batting order: first innings 'inning' field is empty or missing.")

    # Extract the name from e.g. "Perth Scorchers Inning 1"
    parsed_team_name = parse_inning_team_name(first_innings_inning_str)

    # Match parsed_team_name against one of the known teams
    team_a, team_b = all_teams[0], all_teams[1]

    if parsed_team_name == team_a:
        # Then team_a is batting first, team_b is second
        return team_a, team_b
    elif parsed_team_name == team_b:
        # Then team_b is batting first, team_a is second
        return team_b, team_a
    else:
        # If there's a mismatch or partial name, raise an error
        raise ValueError(
            f"Could not match parsed name '{parsed_team_name}' to known teams {all_teams}."
        )


def prepare_features(match_info):
    """
    Extract key features (innings, ball, runs, wickets, total_chasing) 
    from 'match_info' for use in a predictive model.
    Also infers which teams are batting first vs second 
    using match_info["data"]["teams"] and the first innings in 'score'.

    Parameters
    ----------
    match_info : dict
        A dictionary containing match information, including:
          - match_info["data"]["score"]
          - match_info["data"]["teams"]

    Returns
    -------
    dict
        A dictionary with feature keys (innings, ball, runs, wickets, total_chasing)
        and team names (team_batting_first, team_batting_second).

    Raises
    ------
    ValueError
        If we cannot determine the batting order or parse the innings details.
    """
    data_section = match_info.get("data", {})
    score_data = data_section.get("score", [])
    all_teams = data_section.get("teams", [])

    # Determine batting order (may raise ValueError if mismatch)
    team_batting_first, team_batting_second = extract_batting_order(score_data, all_teams)

    # Innings info
    innings = len(score_data)
    innings_index = innings - 1 if innings > 0 else 0

    runs = 0
    wickets = 0
    overs_float = 0.0

    if score_data:
        runs = score_data[innings_index].get("r", 0)
        wickets = score_data[innings_index].get("w", 0)
        overs_float = score_data[innings_index].get("o", 0.0)

    # Convert overs to balls (e.g., 5.2 overs -> 5 overs, 2 balls)
    overs_int = int(overs_float)
    fraction = overs_float - overs_int
    fractional_balls = round(fraction * 10)
    ball_count = overs_int * 6 + fractional_balls + 1

    # If there's at least one innings, that total is the "chasing" target for the second
    total_chasing = np.nan
    if innings > 1:
        total_chasing = score_data[0].get("r", np.nan)

    return {
        "innings": innings,
        "ball": ball_count,
        "runs": runs,
        "wickets": wickets,
        "total_chasing": total_chasing,
        "team_batting_first": team_batting_first,
        "team_batting_second": team_batting_second
    }


def extract_winning_team(status, teams):
    """
    Determine which team won based on the status string.

    Parameters
    ----------
    status : str
        A status string describing the result (e.g., "Team A won by 10 runs").
    teams : list of str
        The list of team names participating in the match.

    Returns
    -------
    str or None
        The name of the winning team, "Tie"/"No Result", or None if not found.
    """
    for team in teams:
        if team in status:
            return team

    lower_status = status.lower()
    if "tie" in lower_status or "draw" in lower_status:
        return "Tie"
    elif "no result" in lower_status:
        return "No Result"
    return None


def determine_chasing_team(score):
    """
    Identify which team was batting second (the chasing team) 
    based on the second innings info in the 'score' list.

    Parameters
    ----------
    score : list of dict
        A list of innings details, e.g.:
        [
            {"inning": "Team A Inning 1"},
            {"inning": "Team B Inning 2"}
        ].

    Returns
    -------
    str or None
        The chasing team's name, or None if it cannot be determined.
    """
    if len(score) >= 2:
        second_innings = score[1]
        inning_info = second_innings.get("inning", "")
        # e.g., "Team B Inning 2" -> "Team B"
        chasing_team = inning_info.replace(" Inning 1", "").replace(" Inning 2", "").strip()
        return chasing_team
    return None


def process_predictions(items):
    """
    Process a list of DynamoDB items (predictions). Convert Decimals to floats, 
    filter out incomplete predictions, and mark them as correct or incorrect.

    Parameters
    ----------
    items : list of dict
        Raw DynamoDB items (possibly including nested structures).

    Returns
    -------
    list of dict
        A list of processed predictions, each dict containing:
          - numeric fields converted to float,
          - 'is_correct': bool indicating if the prediction was correct,
          - 'predicted_at': datetime object if it exists in the item,
          - additional original fields as well.
    """
    processed_data = []
    logger.debug(f"Total items fetched from DynamoDB: {len(items)}")

    for item in items:
        logger.debug(f"Processing item: {item}")
        prediction_data = {}

        # Convert Decimal to float for numerical fields
        for key, value in item.items():
            if isinstance(value, Decimal):
                prediction_data[key] = float(value)
            else:
                prediction_data[key] = value

        logger.debug(f"Converted prediction data: {prediction_data}")
        logger.debug(f"chasing_team_won value: {prediction_data.get('chasing_team_won')}")

        # Only include predictions with a known chasing_team_won result
        chasing_team_won = prediction_data.get("chasing_team_won")
        if chasing_team_won is not None and chasing_team_won != "NULL":
            # Determine if the prediction was correct:
            predicted_win = prediction_data.get("probability", 0) > 0.5
            actual_win = bool(prediction_data.get("chasing_team_won"))
            prediction_data["is_correct"] = (predicted_win == actual_win)

            # Convert string timestamp to datetime if present
            if "predicted_at" in prediction_data:
                # Some timestamps might have 'Z' suffix, replace with UTC offset
                iso_str = prediction_data["predicted_at"].replace("Z", "+00:00")
                prediction_data["predicted_at"] = datetime.fromisoformat(iso_str)

            processed_data.append(prediction_data)
            logger.debug("Added prediction to processed data.")
        else:
            logger.debug("Skipping item due to missing or NULL chasing_team_won.")

    logger.info(f"Processed {len(processed_data)} completed predictions.")
    return processed_data


def calculate_weekly_accuracy(predictions):
    """
    Calculate the weekly prediction accuracy based on 'is_correct' field.
    Instead of storing just the week number, also store the ISO year to avoid
    hardcoding or mixing different years.

    Parameters
    ----------
    predictions : list of dict
        A list of prediction dictionaries with 'predicted_at' and 'is_correct'.

    Returns
    -------
    dict
        A dictionary keyed by (iso_year, iso_week), mapping to percentage accuracy.
        Example:
            {
                (2023, 45): 66.67,
                (2023, 46): 100.0,
                (2024, 1): 50.0,
                ...
            }
    """
    weekly_accuracy = {}

    for pred in predictions:
        # Skip if missing date or correctness info
        if (
            "predicted_at" not in pred
            or "is_correct" not in pred
            or pred.get("chasing_team_won") is None
        ):
            continue

        dt = pred["predicted_at"]
        iso_year, iso_week, _ = dt.isocalendar()  # e.g. (2024, 39, 1)
        key = (iso_year, iso_week)

        if key not in weekly_accuracy:
            weekly_accuracy[key] = {"correct": 0, "total": 0}

        weekly_accuracy[key]["total"] += 1
        if pred["is_correct"]:
            weekly_accuracy[key]["correct"] += 1

    # Convert counts to percentage
    for key, stats in weekly_accuracy.items():
        total = stats["total"]
        correct = stats["correct"]
        weekly_accuracy[key] = (correct / total * 100) if total > 0 else 0

    logger.debug(f"Weekly accuracy calculated for keys (year, week): {list(weekly_accuracy.keys())}")
    return weekly_accuracy


def prepare_chart_data(weekly_accuracy):
    """
    Prepare data for charting weekly accuracy, returning a list of
    date labels (starting Monday of each ISO week) and corresponding accuracies.

    Parameters
    ----------
    weekly_accuracy : dict
        A dictionary keyed by (iso_year, iso_week) with accuracy percentages as values.

    Returns
    -------
    tuple of (list[str], list[float])
        - A list of week labels in chronological order, e.g. ["Nov 06", "Nov 13", ...].
        - A list of accuracies in the same order.
    """
    # Sort by (year, week_num) tuples
    sorted_keys = sorted(weekly_accuracy.keys())
    accuracies = [weekly_accuracy[k] for k in sorted_keys]

    week_labels = []
    for (year, week_num) in sorted_keys:
        # Monday of that iso-week
        date_obj = date.fromisocalendar(year, week_num, 1)
        # Format as e.g. "Nov 06"
        week_labels.append(date_obj.strftime("%b %d"))

    return week_labels, accuracies