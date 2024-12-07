import boto3
import json
import logging
import requests
from botocore.exceptions import ClientError
from other_helpers import extract_winning_team, determine_chasing_team

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
    secret = get_secret('cricket-api-key')
    secret_dict = json.loads(secret)
    return secret_dict['CRIC_API_KEY']

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
        match_data = data.get('data', {})

        teams = match_data.get('teams', [])
        if len(teams) >= 2:
            team_batting_first = teams[0]
            team_batting_second = teams[1]
        else:
            team_batting_first = team_batting_second = 'Unknown'

        score_list = match_data.get('score', [])
        if score_list:
            current_score = score_list[-1]
            current_inning = current_score.get('inning', '')
            runs = current_score.get('r', 0)
            wickets = current_score.get('w', 0)
            overs = current_score.get('o', 0)
        else:
            current_inning = ''
            runs = wickets = overs = 0

        if 'Inning 1' in current_inning:
            innings_number = 1
        elif 'Inning 2' in current_inning:
            innings_number = 2
        else:
            innings_number = 'Unknown'

        if innings_number == 2 and len(score_list) >= 1:
            previous_innings = score_list[0]
            total_chasing = previous_innings.get('r', 0)
        elif innings_number == 1:
            # Not applicable in first innings, use NaN or -1
            total_chasing = float('nan')
        else:
            total_chasing = float('nan')

        result = {
            'team_batting_first': team_batting_first,
            'team_batting_second': team_batting_second,
            'current_innings': innings_number,
            'current_runs': runs,
            'current_wickets': wickets,
            'current_overs': overs,
            'total_chasing': total_chasing
        }

        return result
    else:
        logger.error(f"Failed to fetch match info for match_id {match_id}. Status Code: {response.status_code}")
        return None

def get_match_result(api_key, match_id):
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
                # The match has finished
                winning_team = extract_winning_team(status, teams)
                if winning_team in ('Tie', 'No Result') or winning_team is None:
                    logger.warning(f"Could not determine winning team from status '{status}'")
                    return None, None
                chasing_team = determine_chasing_team(score)
                if not chasing_team:
                    logger.warning(f"Could not determine chasing team for match_id {match_id}")
                    return None, None

                chasing_team_won = 1 if winning_team == chasing_team else 0
                result = status
                return result, chasing_team_won
            else:
                # Match is still ongoing
                logger.info(f"Match {match_id} status: {status}")
                return None, None
        else:
            logger.warning(f"Missing data for match_id {match_id}")
            return None, None
    else:
        logger.error(f"Failed to fetch match info for match_id {match_id}. Status Code: {response.status_code}")
        return None, None
