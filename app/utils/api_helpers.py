import boto3
import json
import logging
import requests

from botocore.exceptions import ClientError
from app.utils.data_helpers import extract_winning_team, determine_chasing_team

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_secret(secret_name):
    region_name = 'eu-north-1'
    session = boto3.session.Session()
    client = session.client(service_name='secretsmanager', region_name=region_name)
    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
        secret = get_secret_value_response['SecretString']
        return secret
    except ClientError as e:
        logger.error(f"Error retrieving secret {secret_name}: {e}")
        raise e

def get_api_key():
    secret = get_secret('cricket_data')
    secret_dict = json.loads(secret)
    return secret_dict['cricket-api-key']

def get_current_matches(api_key):
    url = 'https://api.cricapi.com/v1/currentMatches'
    params = {'apikey': api_key}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        matches = data.get('data', [])
        return matches
    else:
        logger.error(f"Failed to fetch current matches. Status Code: {response.status_code}")
        return []

def get_match_info(api_key, match_id):
    url = 'https://api.cricapi.com/v1/match_info'
    params = {'apikey': api_key, 'id': match_id}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data.get('status') != 'success':
            logger.error(f"Failed to fetch match info for match_id {match_id}.")
            return None
        return data
    else:
        logger.error(f"Failed to fetch match info for match_id {match_id}. Status Code: {response.status_code}")
        return None

def get_match_result(api_key, match_id):
    """
    Determine the match result and whether the chasing team won.
    Returns (result_string, chasing_team_won) or (None, None) if ongoing.
    """
    url = f'https://api.cricapi.com/v1/match_info?apikey={api_key}&id={match_id}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        match_info = data.get('data', {})
        status = match_info.get('status', '')
        teams = match_info.get('teams', [])
        score = match_info.get('score', [])

        if status and teams and score:
            status_lower = status.lower()
            finished_keywords = ['won by', 'tie', 'draw', 'no result', 'abandoned']
            if any(keyword in status_lower for keyword in finished_keywords):
                winning_team = extract_winning_team(status, teams)
                if not winning_team:
                    logger.warning(f"Could not determine winning team from status '{status}'")
                    return None, None
                chasing_team = determine_chasing_team(score)
                if not chasing_team:
                    logger.warning(f"Could not determine chasing team for match_id {match_id}")
                    return None, None
                chasing_team_won = 1 if winning_team == chasing_team else 0
                return status, chasing_team_won
            else:
                # Match still ongoing
                logger.info(f"Match {match_id} status: {status}")
                return None, None
        else:
            logger.warning(f"Missing data for match_id {match_id}")
            return None, None
    else:
        logger.error(f"Failed to fetch match info for match_id {match_id}. Status Code: {response.status_code}")
        return None, None