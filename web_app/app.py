import os
import logging
import numpy as np
from flask import Flask, render_template, request

from catboost import CatBoostClassifier

from rickety_cricket.utils.api_helpers import *
from rickety_cricket.utils.data_helpers import *
from rickety_cricket.utils.db_helpers import *

app = Flask(__name__)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Determine the model path relative to this file
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, '..', 'model_training', 'model.cbm')

# Load the CatBoost model once at startup
model = CatBoostClassifier()
model.load_model(model_path)

# Initialize DynamoDB resource and Predictions instance
dynamodb_resource = boto3.resource('dynamodb', region_name='eu-north-1')
predictions_table = Predictions(dynamodb_resource, 'Predictions')

@app.route('/')
def index():
    api_key = get_api_key()
    matches = get_current_matches(api_key)
    # Filter out matches without IDs or names
    matches = [m for m in matches if m.get('id') and m.get('name')]
    if not matches:
        return render_template('error.html', message='No matches available.')
    return render_template('index.html', matches=matches)

@app.route('/predict', methods=['POST'])
def predict():
    match_id = request.form.get('match_id')
    if not match_id:
        return render_template('error.html', message='No match selected.')

    api_key = get_api_key()
    data = get_match_info(api_key, match_id)
    if data is None:
        return render_template('error.html', message='Failed to retrieve match data.')

    match_info = data
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
        logger.error(f"Error making prediction: {e}")
        return render_template('error.html', message=f'Error making prediction: {str(e)}')

    return render_template('result.html', match_info=match_info, probability=probability)

@app.route('/track_model_performance')
def track_model_performance():
    # Use the predictions_table instance to fetch predictions
    items = predictions_table.fetch_predictions()
    processed_data = process_predictions(items)
    weekly_accuracy = calculate_weekly_accuracy(processed_data)
    weeks, accuracies = prepare_chart_data(weekly_accuracy)

    total_correct = sum(d['is_correct'] for d in processed_data)
    total_predictions = len(processed_data)
    overall_accuracy = (total_correct / total_predictions) * 100 if total_predictions > 0 else 0

    return render_template('track_model_performance.html',
                           weeks=weeks,
                           accuracies=accuracies,
                           overall_accuracy=overall_accuracy)

if __name__ == '__main__':
    app.run()