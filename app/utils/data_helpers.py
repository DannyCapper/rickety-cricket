import math
import numpy as np
from decimal import Decimal
from datetime import datetime
from boto3.dynamodb.types import TypeDeserializer
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def to_decimal(value):
    """
    Convert a numeric value to Decimal for DynamoDB.
    If value is None or invalid (NaN/Infinity), return None to represent null.
    """
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return None
    return Decimal(str(value))

def filter_mens_international_t20(matches):
    """
    Filter for men's international T20 matches. Adjust logic as needed based on actual API data.
    """
    filtered = []
    for match in matches:
        match_type = match.get('matchType', '').lower()
        name = match.get('name', '')
        match_started = match.get('matchStarted', False)
        match_ended = match.get('matchEnded', False)

        if 'women' in name.lower():
            continue
        if match_type == 't20' and match_started and not match_ended:
            filtered.append(match)
    return filtered

def select_random_match(filtered_matches):
    import random
    if filtered_matches:
        return random.choice(filtered_matches)
    return None

def prepare_features(match_info):
    innings = match_info['current_innings']
    runs = match_info['current_runs']
    wickets = match_info['current_wickets']
    overs_float = match_info['current_overs']
    total_chasing = match_info['total_chasing']

    if overs_float is None or overs_float == 0.0:
        balls_bowled = 0
    else:
        overs = int(overs_float)
        partial_over = overs_float - overs
        balls = round(partial_over * 6)
        balls_bowled = overs * 6 + balls
    ball_number = balls_bowled + 1

    return {
        'innings': innings if innings != 'Unknown' else np.nan,
        'ball': ball_number,
        'runs': runs,
        'wickets': wickets,
        'total_chasing': total_chasing if total_chasing is not None else np.nan
    }

def extract_winning_team(status, teams):
    for team in teams:
        if team in status:
            return team
    lower_status = status.lower()
    if 'tie' in lower_status or 'draw' in lower_status:
        return 'Tie'
    elif 'no result' in lower_status:
        return 'No Result'
    return None

def determine_chasing_team(score):
    if len(score) >= 2:
        second_innings = score[1]
        inning_info = second_innings.get('inning', '')
        chasing_team = inning_info.replace(' Inning 1', '').replace(' Inning 2', '').strip()
        return chasing_team
    return None

def process_predictions(items):
    deserializer = TypeDeserializer()
    processed_data = []

    for item in items:
        info = item.get('info', {})
        deserialized_info = {}
        for key, value in info.items():
            deserialized_info[key] = deserializer.deserialize(value)

        predicted_at_str = deserialized_info.get('predicted_at')
        probability_str = deserialized_info.get('probability')
        chasing_team_won_str = deserialized_info.get('chasing_team_won')

        if predicted_at_str and probability_str is not None and chasing_team_won_str is not None:
            predicted_at = datetime.fromisoformat(predicted_at_str)
            week_number = predicted_at.isocalendar()[1]
            year = predicted_at.year
            probability = float(probability_str)
            chasing_team_won_pred = 1 if probability >= 50.0 else 0
            actual_probability = float(chasing_team_won_str)
            chasing_team_won = 1 if actual_probability >= 0.5 else 0
            is_correct = 1 if chasing_team_won_pred == chasing_team_won else 0
            processed_data.append({
                'week_number': week_number,
                'year': year,
                'is_correct': is_correct
            })
        else:
            continue
    return processed_data

def calculate_weekly_accuracy(processed_data):
    weekly_results = defaultdict(lambda: {'correct': 0, 'total': 0})
    for data in processed_data:
        key = f"{data['year']}-W{data['week_number']}"
        weekly_results[key]['correct'] += data['is_correct']
        weekly_results[key]['total'] += 1

    weekly_accuracy = []
    for week in sorted(weekly_results.keys()):
        correct = weekly_results[week]['correct']
        total = weekly_results[week]['total']
        accuracy = (correct / total) * 100 if total > 0 else 0
        weekly_accuracy.append({'week': week, 'accuracy': accuracy})
    return weekly_accuracy

def prepare_chart_data(weekly_accuracy):
    weekly_accuracy.sort(key=lambda x: x['week'])
    weeks = [data['week'] for data in weekly_accuracy]
    accuracies = [data['accuracy'] for data in weekly_accuracy]
    return weeks, accuracies