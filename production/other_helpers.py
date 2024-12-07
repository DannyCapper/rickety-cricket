import logging
import math
import numpy as np
import random
from decimal import Decimal

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def to_decimal(value):
    """
    Convert a numeric value to Decimal for DynamoDB.
    If value is None or invalid (NaN/Infinity), return None to represent null.
    """
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        # Return None so DynamoDB stores it as NULL
        return None
    return Decimal(str(value))

def filter_mens_international_t20(matches):
    """
    Filter matches to men's international T20 matches that are ongoing.
    Adjust logic as needed based on actual API data.
    Currently:
    - matchType == 't20'
    - matchStarted == True
    - matchEnded == False
    - 'women' not in name (to exclude women's matches)
    """
    filtered = []
    for match in matches:
        match_type = match.get('matchType', '').lower()
        # name = match.get('name', '').lower()
        match_started = match.get('matchStarted', False)
        match_ended = match.get('matchEnded', False)
        status = match.get('status', '').lower()

        # if 'women' in name:
        #     continue

        # if match_type == 't20' and 
        if match_started and not match_ended and status != 'match not started':
            filtered.append(match)

    return filtered

def select_random_match(filtered_matches):
    """
    Randomly select a match from the filtered matches, or return None if empty.
    """
    if filtered_matches:
        return random.choice(filtered_matches)
    else:
        return None

def prepare_features(match_info):
    """
    Prepares the feature vector for the model based on match_info.
    """
    innings = match_info['current_innings']
    runs = match_info['current_runs']
    wickets = match_info['current_wickets']
    overs_float = match_info['current_overs']
    total_chasing = match_info['total_chasing']

    if overs_float is None or overs_float == 0:
        balls_bowled = 0
    else:
        overs = int(overs_float)
        partial_over = overs_float - overs
        # This logic might need adjustment depending on how overs are represented
        # If overs are represented like 10.3 overs means 10 overs and 3 balls, we need a different calculation.
        # For now, assume partial_over * 10 gives balls.
        balls_bowled = overs * 6 + round(partial_over * 10)
    ball_number = balls_bowled + 1

    features = {
        'innings': innings,
        'ball': ball_number,
        'runs': runs,
        'wickets': wickets,
        'total_chasing': total_chasing
    }

    return features

def extract_winning_team(status, teams):
    """
    Given a status string and a list of teams, attempt to identify the winning team.
    If the match ended in a tie, draw, or no result, return a placeholder.
    """
    for team in teams:
        if team in status:
            return team
    lower_status = status.lower()
    if 'tie' in lower_status or 'draw' in lower_status:
        return 'Tie'
    elif 'no result' in lower_status:
        return 'No Result'
    else:
        return None

def determine_chasing_team(score):
    """
    Determine the chasing team (the one batting second) from the score data.
    Assumes the second element in the score array corresponds to the chasing team's innings.
    """
    if len(score) >= 2:
        second_innings = score[1]
        inning_info = second_innings.get('inning', '')
        chasing_team = inning_info.replace(' Inning 1', '').replace(' Inning 2', '').strip()
        return chasing_team
    else:
        return None

def update_pending_results(api_key, predictions_table):
    """
    Update pending results for predictions where result is None.
    Calls get_match_result to determine match outcome and updates DynamoDB.
    """
    items = predictions_table.get_pending_predictions()
    for item in items:
        prediction_id = item['prediction_id']
        info = item.get('info', {})
        match_id = info.get('match_id')
        if not match_id:
            logger.warning(f"Match ID not found for prediction_id {prediction_id}")
            continue

        from api_helpers import get_match_result
        result, chasing_team_won = get_match_result(api_key, match_id)
        if result is not None and chasing_team_won is not None:
            predictions_table.update_match_result(prediction_id, result, chasing_team_won)
        else:
            logger.info(f"Match {match_id} is still ongoing.")