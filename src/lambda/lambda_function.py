import boto3
import logging
import numpy as np
import os
import time
from datetime import datetime
from catboost import CatBoostClassifier

from src.utils.api_helpers import (
    get_api_key,
    get_current_matches,
    get_match_info
)
from src.utils.db_helpers import Predictions
from src.utils.data_helpers import (
    filter_mens_international_t20,
    prepare_features, 
    select_random_match,
    to_decimal
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def main(event=None, context=None):
    """
    Main entry point for the Lambda function or script.
    Fetches current matches, selects a men's international T20 match,
    makes a prediction, inserts into DynamoDB, and updates pending results.

    If we cannot parse which team is batting first/second, a ValueError will be raised.
    """

    # 1. Retrieve the API key
    api_key = get_api_key()

    # 2. Initialize DynamoDB resource and Predictions table helper
    dynamodb_resource = boto3.resource("dynamodb", region_name="eu-north-1")
    predictions_table = Predictions(dynamodb_resource, "Predictions")

    # 3. Update any pending results first
    predictions_table.update_pending_results(api_key)

    # 4. Load the CatBoost model
    model = CatBoostClassifier()
    current_dir = os.path.dirname(__file__)     # e.g., src/lambda
    parent_dir = os.path.dirname(current_dir)   # e.g., src
    model_path = os.path.join(parent_dir, "model.cbm")
    model.load_model(model_path)

    # 5. Fetch current matches and filter for men's international T20
    matches = get_current_matches(api_key)
    filtered_matches = filter_mens_international_t20(matches)
    selected_match = select_random_match(filtered_matches)

    if not selected_match:
        logger.info("No men's international T20 matches are currently in play.")
        return

    match_id = selected_match.get("id")
    if not match_id:
        logger.warning("Selected match does not have an ID. Skipping prediction.")
        return

    # 6. Retrieve detailed match information
    match_info = get_match_info(api_key, match_id)
    if not match_info:
        logger.info("No match information available for the selected match.")
        return

    # 7. Prepare features (including batting teams) and make prediction
    try:
        feature_vector = prepare_features(match_info)
    except ValueError as e:
        logger.error(f"Failed to parse batting order or innings data: {e}")
        # Re-raise if you want the Lambda to fail, or you can return gracefully.
        raise

    feature_names = ["innings", "ball", "runs", "wickets", "total_chasing"]
    X = [feature_vector.get(f, 0) for f in feature_names]
    X = np.array(X).reshape(1, -1)

    probability = model.predict_proba(X)[:, 1][0]  # Probability of the chasing team winning
    probability_percent = probability * 100
    logger.info(f"Predicted Probability of chasing team winning: {probability_percent:.2f}%")

    # 8. Insert the prediction data into DynamoDB
    prediction_id = int(time.time())  # Simple integer-based unique identifier

    prediction_data = {
        "prediction_id": prediction_id,
        "predicted_at": datetime.utcnow().isoformat(),
        "match_id": match_id,

        # The newly extracted fields
        "team_batting_first": feature_vector.get("team_batting_first"),
        "team_batting_second": feature_vector.get("team_batting_second"),

        # Innings data
        "innings": to_decimal(feature_vector.get("innings")),
        "ball": to_decimal(feature_vector.get("ball")),
        "runs": to_decimal(feature_vector.get("runs")),
        "wickets": to_decimal(feature_vector.get("wickets")),
        "total_chasing": to_decimal(feature_vector.get("total_chasing")),

        # Probability & placeholders for final result
        "probability": to_decimal(probability),
        "chasing_team_won": None,
        "result": None
    }

    predictions_table.insert_prediction(prediction_data)
    logger.info(f"Inserted prediction data with prediction_id {prediction_id}")

if __name__ == "__main__":
    main()