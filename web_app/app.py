# app.py

from flask import Flask, render_template, request
import requests
import os
import numpy as np
from catboost import CatBoostClassifier

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

if __name__ == '__main__':
    app.run(debug=True)