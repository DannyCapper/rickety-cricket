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
    score = match_info['data']['score']

    innings = len(score)
    innings_index = innings - 1

    runs = score[innings_index]['r']
    wickets = score[innings_index]['w']
    overs = int(score[innings_index]['o'])
    ball = overs + 6 * round((overs - score[innings_index]['o']) * 6) + 1

    if innings == 1:
        total_chasing = np.nan
    else:
        total_chasing = score[0]['r']

    return {
        'innings': innings,
        'ball': ball,
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
    processed_data = []
    
    print(f"Total items fetched from DynamoDB: {len(items)}")  # Debug
    
    for item in items:
        print(f"Processing item: {item}")  # Debug
        prediction_data = {}
        
        # Convert Decimal to float for numerical values
        for key, value in item.items():
            if isinstance(value, Decimal):
                prediction_data[key] = float(value)
            else:
                prediction_data[key] = value
                
        print(f"Converted prediction data: {prediction_data}")  # Debug
        print(f"chasing_team_won value: {prediction_data.get('chasing_team_won')}")  # Debug
        
        # Only include predictions where we know the result
        if (prediction_data.get('chasing_team_won') is not None and 
            prediction_data.get('chasing_team_won') != 'NULL'):  # Check for DynamoDB NULL
            # Calculate if prediction was correct
            predicted_win = prediction_data.get('probability', 0) > 0.5
            actual_win = bool(prediction_data.get('chasing_team_won'))
            prediction_data['is_correct'] = predicted_win == actual_win
            
            # Convert prediction timestamp to datetime if it exists
            if 'predicted_at' in prediction_data:
                prediction_data['predicted_at'] = datetime.fromisoformat(
                    prediction_data['predicted_at'].replace('Z', '+00:00')
                )
            
            processed_data.append(prediction_data)
            print(f"Added to processed data")  # Debug
        else:
            print(f"Skipping item due to missing or NULL chasing_team_won")  # Debug
    
    print(f"Processed {len(processed_data)} completed predictions")
    return processed_data

def calculate_weekly_accuracy(predictions):
    weekly_accuracy = {}
    
    for pred in predictions:
        if ('predicted_at' not in pred or 
            'is_correct' not in pred or 
            pred.get('chasing_team_won') is None):  # Skip pending matches
            continue
            
        # Get the week number
        week = pred['predicted_at'].isocalendar()[1]
        
        if week not in weekly_accuracy:
            weekly_accuracy[week] = {'correct': 0, 'total': 0}
            
        weekly_accuracy[week]['total'] += 1
        if pred['is_correct']:
            weekly_accuracy[week]['correct'] += 1
    
    # Calculate percentages
    for week in weekly_accuracy:
        total = weekly_accuracy[week]['total']
        correct = weekly_accuracy[week]['correct']
        weekly_accuracy[week] = (correct / total * 100) if total > 0 else 0
    
    print(f"Weekly accuracy calculated for weeks: {list(weekly_accuracy.keys())}")  # Debug print
    return weekly_accuracy

def prepare_chart_data(weekly_accuracy):
    # Sort weeks to ensure chronological order
    sorted_weeks = sorted(weekly_accuracy.keys())
    accuracies = [weekly_accuracy[week] for week in sorted_weeks]
    
    # Convert week numbers to dates (first day of each week)
    week_labels = []
    for week in sorted_weeks:
        # Create date from ISO week
        date = datetime.strptime(f'2023-W{week}-1', '%Y-W%W-%w')
        week_labels.append(date.strftime('%b %d'))  # Format as "Nov 06"
    
    return week_labels, accuracies