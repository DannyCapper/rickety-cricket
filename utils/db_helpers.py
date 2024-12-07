import logging
from botocore.exceptions import ClientError
from boto3.dynamodb.conditions import Attr

from rickety_cricket.utils.api_helpers import get_match_result

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Predictions:
    """Encapsulates an Amazon DynamoDB table of predictions data."""

    def __init__(self, dyn_resource, table_name):
        self.dyn_resource = dyn_resource
        self.table_name = table_name

        if self.table_exists(table_name):
            self.table = dyn_resource.Table(table_name)
            logger.info(f"Using existing table {table_name}.")
        else:
            self.table = self.create_table(table_name)
            logger.info(f"Created new table {table_name}.")

    def table_exists(self, table_name):
        try:
            self.dyn_resource.meta.client.describe_table(TableName=table_name)
            return True
        except self.dyn_resource.meta.client.exceptions.ResourceNotFoundException:
            return False

    def create_table(self, table_name):
        try:
            table = self.dyn_resource.create_table(
                TableName=table_name,
                KeySchema=[
                    {"AttributeName": "prediction_id", "KeyType": "HASH"},
                ],
                AttributeDefinitions=[
                    {"AttributeName": "prediction_id", "AttributeType": "N"},
                ],
                BillingMode='PAY_PER_REQUEST',
            )
            table.wait_until_exists()
            logger.info(f"Created table {table_name}.")
            return table
        except ClientError as err:
            logger.error(
                f"Couldn't create table {table_name}. Error: {err.response['Error']['Code']}: {err.response['Error']['Message']}"
            )
            raise

    def insert_prediction(self, prediction_data):
        """
        Insert a prediction item into the DynamoDB table.
        """
        try:
            self.table.put_item(Item=prediction_data)
            logger.info("Inserted prediction data into DynamoDB.")
        except ClientError as err:
            logger.error(
                f"Couldn't insert data into table {self.table.name}. Error: {err.response['Error']['Code']}: {err.response['Error']['Message']}"
            )
            raise

    def get_pending_predictions(self):
        """
        Retrieve predictions where 'info.result' is not set or is None.
        """
        response = self.table.scan(
            FilterExpression=Attr('info.result').not_exists() | Attr('info.result').eq(None)
        )
        items = response.get('Items', [])

        while 'LastEvaluatedKey' in response:
            response = self.table.scan(
                FilterExpression=Attr('info.result').not_exists() | Attr('info.result').eq(None),
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            items.extend(response.get('Items', []))
        return items

    def update_match_result(self, prediction_id, result, chasing_team_won):
        """
        Update the 'info.result' and 'info.chasing_team_won' attributes for a given prediction.
        """
        try:
            self.table.update_item(
                Key={'prediction_id': prediction_id},
                UpdateExpression="SET info.#r = :r, info.#c = :c",
                ExpressionAttributeNames={'#r': 'result', '#c': 'chasing_team_won'},
                ExpressionAttributeValues={':r': result, ':c': chasing_team_won}
            )
            logger.info(f"Updated result and chasing_team_won for prediction_id {prediction_id}")
        except ClientError as err:
            logger.error(
                f"Couldn't update result for prediction_id {prediction_id}. Error: {err.response['Error']['Code']}: {err.response['Error']['Message']}"
            )
            raise

def update_pending_results(api_key, predictions_table):
    """
    Update pending results where matches have concluded.
    """
    items = predictions_table.get_pending_predictions()
    for item in items:
        if not isinstance(item, dict):
            logger.error(f"Item is not a dict: {item}")
            continue

        prediction_id = item.get('prediction_id')
        if prediction_id is None:
            logger.warning("No prediction_id found in item.")
            continue

        info = item.get('info', {})
        match_id = info.get('match_id')
        if not match_id:
            logger.warning(f"Match ID not found for prediction_id {prediction_id}")
            continue

        result, chasing_team_won = get_match_result(api_key, match_id)
        if result is not None and chasing_team_won is not None:
            predictions_table.update_match_result(prediction_id, result, chasing_team_won)
        else:
            logger.info(f"Match {match_id} is still ongoing or no result found.")