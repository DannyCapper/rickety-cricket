import logging

from botocore.exceptions import ClientError
from boto3.dynamodb.conditions import Attr

from src.utils.api_helpers import get_match_result

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Predictions:
    """
    Encapsulates an Amazon DynamoDB table for storing and updating cricket match predictions.
    
    This class handles:
      - Checking if a table exists (and creating it if not).
      - Inserting new predictions as top-level attributes (rather than a nested 'info' map).
      - Fetching pending predictions (i.e., matches whose 'result' is not yet determined).
      - Updating results in the table once the match outcome is known.
      - Fetching all predictions from the table.
    """

    def __init__(self, dyn_resource, table_name):
        """
        Initialize the Predictions class with a DynamoDB resource and table name.

        Parameters
        ----------
        dyn_resource : boto3.resource
            A DynamoDB resource object.
        table_name : str
            The name of the DynamoDB table to use or create.
        """
        self.dyn_resource = dyn_resource
        self.table_name = table_name

        if self.table_exists(table_name):
            self.table = dyn_resource.Table(table_name)
            logger.info(f"Using existing table '{table_name}'.")
        else:
            self.table = self.create_table(table_name)
            logger.info(f"Created new table '{table_name}'.")

    def table_exists(self, table_name):
        """
        Check if the given DynamoDB table exists.

        Parameters
        ----------
        table_name : str
            The name of the table to check.

        Returns
        -------
        bool
            True if the table exists, otherwise False.
        """
        try:
            self.dyn_resource.meta.client.describe_table(TableName=table_name)
            return True
        except self.dyn_resource.meta.client.exceptions.ResourceNotFoundException:
            return False

    def create_table(self, table_name):
        """
        Create a new DynamoDB table with the specified name and a simple schema.

        Parameters
        ----------
        table_name : str
            The name of the table to create.

        Returns
        -------
        boto3.resources.factory.dynamodb.Table
            A reference to the newly created DynamoDB table.

        Raises
        ------
        ClientError
            If the table creation fails for any reason.
        """
        try:
            table = self.dyn_resource.create_table(
                TableName=table_name,
                KeySchema=[
                    {"AttributeName": "prediction_id", "KeyType": "HASH"},
                ],
                AttributeDefinitions=[
                    {"AttributeName": "prediction_id", "AttributeType": "N"},
                ],
                BillingMode="PAY_PER_REQUEST",
            )
            table.wait_until_exists()
            logger.info(f"Created table '{table_name}'.")
            return table
        except ClientError as err:
            logger.error(
                f"Couldn't create table '{table_name}'. "
                f"Error: {err.response['Error']['Code']}: {err.response['Error']['Message']}"
            )
            raise

    def insert_prediction(self, prediction_data):
        """
        Insert a new prediction item into the DynamoDB table.

        Parameters
        ----------
        prediction_data : dict
            A dictionary representing the prediction record to store,
            including fields like 'match_id', 'predicted_at', 'probability', etc.

        Raises
        ------
        ClientError
            If the insert (put_item) fails.
        """
        try:
            self.table.put_item(Item=prediction_data)
            logger.info("Inserted prediction data into DynamoDB.")
        except ClientError as err:
            logger.error(
                f"Couldn't insert data into table '{self.table.name}'. "
                f"Error: {err.response['Error']['Code']}: {err.response['Error']['Message']}"
            )
            raise

    def get_pending_predictions(self):
        """
        Retrieve all prediction records from the table where 'result' is missing or None.

        Returns
        -------
        list of dict
            A list of pending prediction items from the table.
        """
        response = self.table.scan(
            FilterExpression=Attr("result").not_exists() | Attr("result").eq(None)
        )
        items = response.get("Items", [])

        # Handle pagination
        while "LastEvaluatedKey" in response:
            response = self.table.scan(
                FilterExpression=Attr("result").not_exists() | Attr("result").eq(None),
                ExclusiveStartKey=response["LastEvaluatedKey"]
            )
            items.extend(response.get("Items", []))

        return items

    def update_match_result(self, prediction_id, result, chasing_team_won):
        """
        Update the 'result' and 'chasing_team_won' fields for a given prediction.

        Parameters
        ----------
        prediction_id : int
            The unique prediction ID (primary key in DynamoDB).
        result : str
            A status string describing the final match result (e.g., 'Team A won by 10 runs').
        chasing_team_won : int
            An integer (0 or 1) indicating if the chasing team won or lost.

        Raises
        ------
        ClientError
            If the update operation fails.
        """
        try:
            self.table.update_item(
                Key={"prediction_id": prediction_id},
                UpdateExpression="SET #r = :r, #c = :c",
                ExpressionAttributeNames={"#r": "result", "#c": "chasing_team_won"},
                ExpressionAttributeValues={":r": result, ":c": chasing_team_won},
            )
            logger.info(
                f"Updated 'result' and 'chasing_team_won' for prediction_id={prediction_id} "
                f"to ({result}, {chasing_team_won})."
            )
        except ClientError as err:
            logger.error(
                f"Couldn't update result for prediction_id={prediction_id}. "
                f"Error: {err.response['Error']['Code']}: {err.response['Error']['Message']}"
            )
            raise

    def update_pending_results(self, api_key):
        """
        Check all pending predictions and update their result if the match has concluded.

        Steps:
        1. Retrieve all pending predictions (where 'result' is missing/None).
        2. For each pending prediction, call `get_match_result` to see if the match is finished.
        3. If finished, update the table with the final 'result' and 'chasing_team_won'.
        4. If still ongoing, do nothing (leave it pending).

        Parameters
        ----------
        api_key : str
            The API key required for retrieving match info.
        """
        items = self.get_pending_predictions()
        for item in items:
            if not isinstance(item, dict):
                logger.error(f"Skipped an invalid item: {item}")
                continue

            prediction_id = item.get("prediction_id")
            if prediction_id is None:
                logger.warning("Skipping item with missing 'prediction_id'.")
                continue

            match_id = item.get("match_id")
            if not match_id:
                logger.warning(
                    f"Skipping item with prediction_id={prediction_id} due to missing 'match_id'."
                )
                continue

            # Retrieve the final match result if available
            result, chasing_team_won = get_match_result(api_key, match_id)
            if result is not None and chasing_team_won is not None:
                self.update_match_result(prediction_id, result, chasing_team_won)
            else:
                logger.info(
                    f"Match {match_id} is still ongoing or has no result yet. "
                    f"Leaving prediction_id={prediction_id} as pending."
                )

    def fetch_predictions(self):
        """
        Retrieve all predictions from the DynamoDB table associated with this Predictions instance.

        Returns
        -------
        list of dict
            A list of all items (predictions) in the table.

        Raises
        ------
        ClientError
            If the scan operation fails.
        """
        try:
            response = self.table.scan()
            items = response.get("Items", [])

            # Handle pagination if there are more items
            while "LastEvaluatedKey" in response:
                response = self.table.scan(
                    ExclusiveStartKey=response["LastEvaluatedKey"]
                )
                items.extend(response.get("Items", []))

            logger.info(f"Fetched {len(items)} predictions from DynamoDB.")
            return items

        except ClientError as e:
            logger.error(
                f"Error fetching predictions from table '{self.table_name}'. "
                f"Error: {e.response['Error']['Code']}: {e.response['Error']['Message']}"
            )
            raise