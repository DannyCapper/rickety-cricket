from flask import Flask, render_template, request
import requests
import numpy as np
from catboost import CatBoostClassifier
import boto3
from boto3.dynamodb.conditions import Attr
from datetime import datetime

app = Flask(__name__)

# Load the CatBoost model once at startup
model = CatBoostClassifier()
model.load_model('../model_training/model.cbm')


@app.route('/')
def index():
    # Fetch live matches
    url = 'https://api.cricapi.com/v1/currentMatches'
    params = {'apikey': '57859138-e236-4a32-9507-560e9bf590dd'}
    response = requests.get(url, params=params)
    data = response.json()

    if data.get('status') != 'success':
        return render_template('error.html', message='Failed to retrieve live matches.')

    matches = data.get('data', [])

    # Filter out matches without IDs or names
    matches = [match for match in matches if match.get('id') and match.get('name')]

    return render_template('index.html', matches=matches)

@app.route('/predict', methods=['POST'])
def predict():
    match_id = request.form.get('match_id')

    # Fetch match details
    url = 'https://api.cricapi.com/v1/match_info'
    params = {'apikey': '57859138-e236-4a32-9507-560e9bf590dd', 'id': '3d012eaa-74fc-46fb-bad1-7e7f704c87a0'}
    response = requests.get(url, params=params)
    match_data = response.json()

    if match_data.get('status') != 'success':
        return render_template('error.html', message='Failed to retrieve match data.')

    # Prepare features
    match_info = get_match_info(match_data)
    feature_vector = prepare_features(match_info)

    # Ensure the features are in the correct order
    feature_names = ['innings', 'ball', 'runs', 'wickets', 'total_chasing']
    X = [feature_vector.get(f, np.nan) for f in feature_names]
    X = np.array(X).reshape(1, -1)

    # Make prediction
    try:
        probability = model.predict_proba(X)[:, 1][0] * 100
        probability = round(probability, 2)
    except Exception as e:
        return render_template('error.html', message=f'Error making prediction: {str(e)}')

    # Render the result
    return render_template('result.html', match_info=match_info, probability=probability)

def get_match_info(data):
    """
    Extracts match information from the API response and prepares it for feature extraction.
    """
    # Extract necessary information from the API response
    match_data = data.get('data', {})

    # Extract team names
    teams = match_data.get('teams', [])
    if len(teams) >= 2:
        team_batting_first = teams[0]
        team_batting_second = teams[1]
    else:
        team_batting_first = team_batting_second = 'Unknown'

    # Extract score information
    score_list = match_data.get('score', [])

    if score_list:
        # Assuming the last item is the latest score
        current_score = score_list[-1]
        current_inning = current_score.get('inning', '')
        runs = current_score.get('r', 0)
        wickets = current_score.get('w', 0)
        overs_str = current_score.get('o', '0')
    else:
        current_inning = ''
        runs = wickets = 0
        overs_str = '0'

    # Convert overs to float (e.g., '10.2' overs to 10.333...)
    try:
        if '.' in str(overs_str):
            overs_parts = str(overs_str).split('.')
            overs = int(overs_parts[0])
            balls = int(overs_parts[1])
            overs_float = overs + balls / 6
        else:
            overs_float = float(overs_str)
    except ValueError:
        overs_float = 0.0

    # Determine if it's first or second innings
    if 'Inning 1' in current_inning:
        innings_number = 1
    elif 'Inning 2' in current_inning:
        innings_number = 2
    else:
        innings_number = 'Unknown'

    # Get total runs of previous innings if it's the second innings
    total_chasing = None
    if innings_number == 2 and len(score_list) >= 1:
        # Assuming the first item in score_list is the previous innings
        previous_innings = score_list[0]
        total_chasing = previous_innings.get('r', 0)
    elif innings_number == 1:
        total_chasing = np.nan  # Not applicable in first innings
    else:
        total_chasing = np.nan  # Unknown innings

    # Prepare the match_info dictionary
    match_info = {
        'team_batting_first': team_batting_first,
        'team_batting_second': team_batting_second,
        'current_innings': innings_number,
        'current_runs': runs,
        'current_wickets': wickets,
        'current_overs': overs_float,
        'total_chasing': total_chasing,
        'current_inning_label': current_inning
    }

    return match_info

def prepare_features(match_info):
    """
    Prepares the feature vector for the model based on match_info.
    """
    # Extract values
    innings = match_info['current_innings']
    runs = match_info['current_runs']
    wickets = match_info['current_wickets']
    overs_float = match_info['current_overs']
    total_chasing = match_info['total_chasing']

    # Calculate balls bowled
    if overs_float is None or overs_float == 0.0:
        balls_bowled = 0
    else:
        overs = int(overs_float)
        partial_over = overs_float - overs
        balls = round(partial_over * 6)
        balls_bowled = overs * 6 + balls
    ball_number = balls_bowled + 1  # Assuming ball numbers start at 1

    # Prepare feature vector
    features = {
        'innings': innings if innings != 'Unknown' else np.nan,
        'ball': ball_number,
        'runs': runs,
        'wickets': wickets,
        'total_chasing': total_chasing if total_chasing is not None else np.nan
    }

    return features


from datetime import datetime
from boto3.dynamodb.types import TypeDeserializer

def process_predictions(items):
    deserializer = TypeDeserializer()
    processed_data = []

    for item in items:
        info = item.get('info', {})

        # Deserialize the 'info' map
        deserialized_info = {}
        for key, value in info.items():
            deserialized_info[key] = deserializer.deserialize(value)

        # Extract necessary fields
        predicted_at_str = deserialized_info.get('predicted_at')
        probability_str = deserialized_info.get('probability')
        chasing_team_won_str = deserialized_info.get('chasing_team_won')

        if predicted_at_str and probability_str is not None and chasing_team_won_str is not None:
            # Convert 'predicted_at' to datetime object
            predicted_at = datetime.fromisoformat(predicted_at_str)

            # Get week number and year
            week_number = predicted_at.isocalendar()[1]
            year = predicted_at.year

            # Convert probability to float
            probability = float(probability_str)

            # Convert to binary prediction using 0.5 threshold
            chasing_team_won_pred = 1 if probability >= 50.0 else 0

            # Get the actual result (assuming it's a float between 0 and 1)
            chasing_team_won = 1 if float(chasing_team_won_str) >= 0.5 else 0

            # Check if prediction matches actual result
            is_correct = 1 if chasing_team_won_pred == chasing_team_won else 0

            # Append processed data
            processed_data.append({
                'week_number': week_number,
                'year': year,
                'is_correct': is_correct
            })
        else:
            # Handle missing data
            continue

    return processed_data


def fetch_predictions():
        dynamodb = boto3.resource('dynamodb', region_name='eu-north-1')
        table = dynamodb.Table('YourTableName')  # Replace with your actual table name

        # Scan the table for items where 'info.result' exists
        response = table.scan(
            FilterExpression=Attr('info.result').exists() & Attr('info.chasing_team_won').exists()
        )
        items = response.get('Items', [])

        # Handle pagination if necessary
        while 'LastEvaluatedKey' in response:
            response = table.scan(
                FilterExpression=Attr('info.result').exists() & Attr('info.chasing_team_won').exists(),
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            items.extend(response.get('Items', []))

        return items


@app.route('/track_model_performance')
def track_model_performance():
    items = fetch_predictions()
    processed_data = process_predictions(items)
    weekly_accuracy = calculate_weekly_accuracy(processed_data)
    weeks, accuracies = prepare_chart_data(weekly_accuracy)

    # Calculate overall accuracy
    total_correct = sum(data['is_correct'] for data in processed_data)
    total_predictions = len(processed_data)
    overall_accuracy = (total_correct / total_predictions) * 100 if total_predictions > 0 else 0

    return render_template('track_model_performance.html',
                           weeks=weeks,
                           accuracies=accuracies,
                           overall_accuracy=overall_accuracy)


def prepare_chart_data(weekly_accuracy):
    # Sort the data by week
    weekly_accuracy.sort(key=lambda x: x['week'])

    # Separate weeks and accuracy for plotting
    weeks = [data['week'] for data in weekly_accuracy]
    accuracies = [data['accuracy'] for data in weekly_accuracy]

    return weeks, accuracies


from collections import defaultdict

def calculate_weekly_accuracy(processed_data):
    # Dictionary to store accuracy per week
    weekly_results = defaultdict(lambda: {'correct': 0, 'total': 0})

    for data in processed_data:
        key = f"{data['year']}-W{data['week_number']}"
        weekly_results[key]['correct'] += data['is_correct']
        weekly_results[key]['total'] += 1

    # Calculate accuracy per week
    weekly_accuracy = []
    for week in sorted(weekly_results.keys()):
        correct = weekly_results[week]['correct']
        total = weekly_results[week]['total']
        accuracy = (correct / total) * 100 if total > 0 else 0
        weekly_accuracy.append({
            'week': week,
            'accuracy': accuracy
        })

    return weekly_accuracy


from datetime import datetime
from boto3.dynamodb.types import TypeDeserializer

def process_predictions(items):
    deserializer = TypeDeserializer()
    processed_data = []

    for item in items:
        info = item.get('info', {})

        # Deserialize the 'info' map
        deserialized_info = {}
        for key, value in info.items():
            deserialized_info[key] = deserializer.deserialize(value)

        # Extract necessary fields
        predicted_at_str = deserialized_info.get('predicted_at')
        probability_str = deserialized_info.get('probability')
        chasing_team_won_str = deserialized_info.get('chasing_team_won')

        if predicted_at_str and probability_str is not None and chasing_team_won_str is not None:
            # Convert 'predicted_at' to datetime object
            predicted_at = datetime.fromisoformat(predicted_at_str)

            # Get week number and year
            week_number = predicted_at.isocalendar()[1]
            year = predicted_at.year

            # Convert probability to float
            probability = float(probability_str)

            # Convert to binary prediction using 0.5 threshold
            chasing_team_won_pred = 1 if probability >= 50.0 else 0

            # Get the actual result (assuming it's a float between 0 and 1)
            chasing_team_won = 1 if float(chasing_team_won_str) >= 0.5 else 0

            # Check if prediction matches actual result
            is_correct = 1 if chasing_team_won_pred == chasing_team_won else 0

            # Append processed data
            processed_data.append({
                'week_number': week_number,
                'year': year,
                'is_correct': is_correct
            })
        else:
            # Handle missing data
            continue

    return processed_data

if __name__ == '__main__':
    app.run(debug=True)