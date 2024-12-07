from flask import Flask, render_template, request
import requests
import numpy as np
import logging
import boto3
from api_helpers import get_api_key
from common_helpers import (
    prepare_features, process_predictions, fetch_predictions,
    calculate_weekly_accuracy, prepare_chart_data, get_match_info, filter_mens_international_t20,
    select_random_match
)
from catboost import CatBoostClassifier

app = Flask(__name__)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Load the CatBoost model once at startup
model = CatBoostClassifier()
model.load_model('../model_training/model.cbm')

@app.route('/')
def index():
    # Fetch live matches using get_api_key() and get_current_matches (from predict_winner or api_helpers)
    api_key = get_api_key()
    url = 'https://api.cricapi.com/v1/currentMatches'
    params = {'apikey': api_key}
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
    if not match_id:
        return render_template('error.html', message='No match selected.')

    api_key = get_api_key()
    url = 'https://api.cricapi.com/v1/match_info'
    params = {'apikey': api_key, 'id': match_id}
    response = requests.get(url, params=params)
    data = response.json()

    if data.get('status') != 'success':
        return render_template('error.html', message='Failed to retrieve match data.')

    match_info = get_match_info(data)
    if not match_info:
        return render_template('error.html', message='No match information available.')

    feature_vector = prepare_features(match_info)
    feature_names = ['innings', 'ball', 'runs', 'wickets', 'total_chasing']
    X = [feature_vector.get(f, np.nan) for f in feature_names]
    X = np.array(X).reshape(1, -1)

    try:
        probability = model.predict_proba(X)[:, 1][0] * 100
        probability = round(probability, 2)
    except Exception as e:
        return render_template('error.html', message=f'Error making prediction: {str(e)}')

    return render_template('result.html', match_info=match_info, probability=probability)

@app.route('/track_model_performance')
def track_model_performance():
    items = fetch_predictions(table_name='Predictions', region='eu-north-1')
    processed_data = process_predictions(items)
    weekly_accuracy = calculate_weekly_accuracy(processed_data)
    weeks, accuracies = prepare_chart_data(weekly_accuracy)

    total_correct = sum(data['is_correct'] for data in processed_data)
    total_predictions = len(processed_data)
    overall_accuracy = (total_correct / total_predictions) * 100 if total_predictions > 0 else 0

    return render_template('track_model_performance.html',
                           weeks=weeks,
                           accuracies=accuracies,
                           overall_accuracy=overall_accuracy)

if __name__ == '__main__':
    app.run(debug=True)