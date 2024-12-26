import os
import logging

import numpy as np
from flask import Flask, render_template, request
from catboost import CatBoostClassifier
import boto3

from src.utils.api_helpers import (
    get_api_key,
    get_current_matches,
    get_match_info
)
from src.utils.data_helpers import (
    filter_mens_international_t20,
    prepare_features,
    process_predictions,
    calculate_weekly_accuracy,
    prepare_chart_data
)
from src.utils.db_helpers import Predictions

app = Flask(__name__)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -------------------------------------------------------------------
#  Load the CatBoost model
# -------------------------------------------------------------------
current_dir = os.path.dirname(__file__) 
parent_dir = os.path.dirname(current_dir)
model_path = os.path.join(parent_dir, "model.cbm")

model = CatBoostClassifier()
model.load_model(model_path)

# -------------------------------------------------------------------
#  Initialize the DynamoDB resource and Predictions table helper
# -------------------------------------------------------------------
dynamodb_resource = boto3.resource("dynamodb", region_name="eu-north-1")
predictions_table = Predictions(dynamodb_resource, "Predictions")

@app.route("/")
def index():
    """
    Main endpoint for listing ongoing men's T20 international matches.
    
    1. Retrieves an API key from AWS Secrets Manager.
    2. Fetches all current matches via CricAPI.
    3. Filters only men's T20 matches that are in progress.
    4. Renders an index page with a list of filtered matches or a message if none.

    Returns
    -------
    flask.Response
        A rendered template 'index.html' containing match data or an error message.
    """
    try:
        api_key = get_api_key()
        matches = get_current_matches(api_key)
        filtered_matches = filter_mens_international_t20(matches)

        # If no matches are in progress, display a friendly message on the page.
        message = None
        if not filtered_matches:
            message = "There are currently no men's T20 international matches in progress."

        return render_template(
            "index.html",
            matches=filtered_matches,
            message=message
        )
    except Exception as exc:
        logger.error(f"Error accessing matches: {exc}", exc_info=True)
        return render_template(
            "index.html",
            matches=[],
            message="Unable to fetch match data. Please try again later."
        )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle the 'Predict' action triggered by a form submission.

    1. Retrieve the match_id from the POST form data.
    2. Use the API key to fetch detailed match information from CricAPI.
    3. Prepare feature vectors using local data_helpers.
    4. Make a prediction (probability of the chasing team winning).
    5. Render a results page with the prediction outcome.

    Returns
    -------
    flask.Response
        A rendered template 'result.html' with the match info and prediction probability,
        or 'error.html' if any step fails.
    """
    match_id = request.form.get("match_id")
    if not match_id:
        return render_template("error.html", message="No match selected.")

    api_key = get_api_key()
    data = get_match_info(api_key, match_id)
    if data is None:
        return render_template("error.html", message="Failed to retrieve match data.")

    match_info = data
    if not match_info:
        return render_template("error.html", message="No match information available.")

    # Prepare the feature vector for the model
    feature_vector = prepare_features(match_info)
    feature_names = ["innings", "ball", "runs", "wickets", "total_chasing"]
    X = [feature_vector.get(f, np.nan) for f in feature_names]
    X = np.array(X).reshape(1, -1)

    try:
        # Probability that the chasing team will win, in percentage form
        probability = model.predict_proba(X)[:, 1][0] * 100
        probability = round(probability, 2)
    except Exception as exc:
        logger.error(f"Error making prediction: {exc}", exc_info=True)
        return render_template("error.html", message=f"Error making prediction: {str(exc)}")

    return render_template(
        "result.html",
        match_info=match_info,
        probability=probability
    )


@app.route("/track_model_performance")
def track_model_performance():
    """
    Display a page showing weekly and overall model accuracy.

    Steps:
    1. Fetch all predictions from DynamoDB via Predictions class.
    2. Convert them into a uniform structure via process_predictions.
    3. Calculate weekly accuracy using calculate_weekly_accuracy.
    4. Prepare chart data (weeks & accuracies) for plotting.
    5. Compute overall accuracy across all completed predictions.
    6. Render a template that displays a chart of weekly accuracy and shows overall accuracy.

    Returns
    -------
    flask.Response
        Renders 'track_model_performance.html' with the chart data and overall accuracy.
    """
    # Fetch all predictions from DynamoDB
    items = predictions_table.fetch_predictions()

    # Convert raw items to a structured list of dicts (numeric fields as floats, etc.)
    processed_data = process_predictions(items)

    # Calculate weekly accuracy for each (year, iso_week)
    weekly_accuracy = calculate_weekly_accuracy(processed_data)

    # Convert that accuracy dictionary into chart-friendly lists
    weeks, accuracies = prepare_chart_data(weekly_accuracy)

    # Compute an overall accuracy metric
    total_correct = sum(d["is_correct"] for d in processed_data)
    total_predictions = len(processed_data)
    overall_accuracy = (total_correct / total_predictions) * 100 if total_predictions > 0 else 0

    return render_template(
        "track_model_performance.html",
        weeks=weeks,
        accuracies=accuracies,
        overall_accuracy=overall_accuracy
    )


if __name__ == "__main__":
    app.run(debug=False)