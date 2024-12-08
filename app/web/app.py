import os
import logging
import numpy as np
from flask import Flask, render_template, request

from catboost import CatBoostClassifier

from app.utils.api_helpers import *
from app.utils.data_helpers import *
from app.utils.db_helpers import *

app = Flask(__name__)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Determine the model path relative to this file
model = CatBoostClassifier()
current_dir = os.path.dirname(__file__)  # Gets app/web directory
parent_dir = os.path.dirname(current_dir)  # Goes up to app directory
model_path = os.path.join(parent_dir, 'model.cbm')
model.load_model(model_path)

# Initialize DynamoDB resource and Predictions instance
dynamodb_resource = boto3.resource('dynamodb', region_name='eu-north-1')
predictions_table = Predictions(dynamodb_resource, 'Predictions')

@app.route('/')
def index():
    try:
        api_key = get_api_key()
        matches = get_current_matches(api_key)
        filtered_matches = filter_mens_international_t20(matches)
        
        # Instead of returning error page, pass empty matches and a message
        message = None
        if not filtered_matches:
            message = "There are currently no men's T20 international matches in progress."
        
        return render_template('index.html', 
                             matches=filtered_matches, 
                             message=message)
    except Exception as e:
        logger.error(f"Error accessing matches: {e}")
        return render_template('index.html', 
                             matches=[], 
                             message="Unable to fetch match data. Please try again later.")

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