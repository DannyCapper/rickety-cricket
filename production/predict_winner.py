import boto3
import logging
import numpy as np
import os
import time

from catboost import CatBoostClassifier
from datetime import datetime

from rickety_cricket.utils.api_helpers import *
from rickety_cricket.utils.db_helpers import *
from rickety_cricket.utils.data_helpers import *

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    """
    Main entry point for the Lambda function or script.
    Fetches current matches, selects a men's international T20 match,
    makes a prediction, inserts into DynamoDB, and updates pending results.
    """

    api_key = get_api_key()

    # Initialize DynamoDB resource
    dynamodb_resource = boto3.resource('dynamodb', region_name='eu-north-1')
    predictions_table = Predictions(dynamodb_resource, 'Predictions')

    # Update any pending results first
    predictions_table.update_pending_results(api_key)

    # Load the model
    model = CatBoostClassifier()
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, 'model.cbm')
    model.load_model(model_path)

    # Fetch current matches and filter for men's international T20
    matches = get_current_matches(api_key)
    filtered_matches = filter_mens_international_t20(matches)
    selected_match = select_random_match(filtered_matches)

    if selected_match:
        match_id = selected_match.get('id')
        if not match_id:
            logger.warning("Selected match does not have an ID. Skipping prediction.")
            return

        # Get match information using the selected match_id
        match_info = get_match_info(api_key, match_id)

        if match_info:
            print(match_info)
            if match_info['current_innings'] == 'Unknown':
                logger.info("Match hasn't started. Skipping prediction.")
            else:
                # Prepare features for prediction
                feature_vector = prepare_features(match_info)
                feature_names = ['innings', 'ball', 'runs', 'wickets', 'total_chasing']
                X = [feature_vector.get(f, 0) for f in feature_names]
                X = np.array(X).reshape(1, -1)

                # Make prediction
                probability = model.predict_proba(X)[:, 1][0]  # Probability of the chasing team winning
                probability_percent = probability * 100
                logger.info(f"Predicted Probability of chasing team winning: {probability_percent:.2f}%")

                # Prepare data to insert into DynamoDB
                prediction_id = int(time.time())  # Unique identifier

                # Construct the 'info' map
                info_data = {
                    'predicted_at': datetime.utcnow().isoformat(),
                    'match_id': match_id,
                    'team_batting_first': match_info.get('team_batting_first'),
                    'team_batting_second': match_info.get('team_batting_second'),
                    'innings': to_decimal(feature_vector.get('innings')),
                    'ball': to_decimal(feature_vector.get('ball')),
                    'runs': to_decimal(feature_vector.get('runs')),
                    'wickets': to_decimal(feature_vector.get('wickets')),
                    'total_chasing': to_decimal(feature_vector.get('total_chasing')),
                    'probability': to_decimal(probability),
                    'chasing_team_won': None,  # To be updated later when the match result is known
                    'result': None  # Initial value is None
                }

                # Create the prediction data with 'prediction_id' and 'info'
                prediction_data = {
                    'prediction_id': prediction_id,
                    'info': info_data
                }

                # Insert the prediction data into DynamoDB
                predictions_table.insert_prediction(prediction_data)
                logger.info(f"Inserted prediction data with prediction_id {prediction_id}")
        else:
            logger.info("No match information available for the selected match.")
    else:
        logger.info("No men's international T20 matches are currently in play.")


if __name__ == '__main__':
    main()
