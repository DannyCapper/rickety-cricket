import json
import logging

import boto3
import requests
from botocore.exceptions import ClientError

from src.utils.data_helpers import extract_winning_team, determine_chasing_team

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_secret(secret_name: str) -> str:
    """
    Retrieve a secret value (string) from AWS Secrets Manager.

    Parameters
    ----------
    secret_name : str
        The name of the secret to retrieve.

    Returns
    -------
    str
        The retrieved secret as a JSON-formatted string.

    Raises
    ------
    ClientError
        If there is an error retrieving the secret from Secrets Manager.
    """
    region_name = "eu-north-1"
    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region_name)

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
        secret = get_secret_value_response["SecretString"]
        return secret
    except ClientError as exc:
        logger.error(f"Error retrieving secret {secret_name}: {exc}")
        raise exc


def get_api_key() -> str:
    """
    Retrieve the cricket API key from the "cricket_data" secret.

    Returns
    -------
    str
        The cricket API key.
    """
    secret = get_secret("cricket_data")
    secret_dict = json.loads(secret)
    return secret_dict["cricket-api-key"]


def get_current_matches(api_key: str) -> list:
    """
    Fetch the list of current matches from the CricAPI.

    Parameters
    ----------
    api_key : str
        The API key used to authenticate with the CricAPI.

    Returns
    -------
    list
        A list of dictionaries describing the current matches.
        Returns an empty list if the request fails.
    """
    url = "https://api.cricapi.com/v1/currentMatches"
    params = {"apikey": api_key}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        matches = data.get("data", [])
        return matches
    else:
        logger.error(f"Failed to fetch current matches. Status Code: {response.status_code}")
        return []


def get_match_info(api_key: str, match_id: str) -> dict:
    """
    Retrieve detailed information about a single match by its ID.

    Parameters
    ----------
    api_key : str
        The API key used to authenticate with the CricAPI.
    match_id : str
        The unique match identifier in CricAPI.

    Returns
    -------
    dict or None
        A dictionary containing match information if successful,
        or None if an error occurs or if the API reports a failure.
    """
    url = "https://api.cricapi.com/v1/match_info"
    params = {"apikey": api_key, "id": match_id}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        if data.get("status") != "success":
            logger.error(f"Failed to fetch match info for match_id {match_id}.")
            return None
        return data
    else:
        logger.error(f"Failed to fetch match info for match_id {match_id}. Status Code: {response.status_code}")
        return None


def get_match_result(api_key: str, match_id: str) -> tuple:
    """
    Determine the match result and whether the chasing team won.

    This function calls the CricAPI's /match_info endpoint to retrieve
    the most recent match information. It checks the match status to see
    if the game is finished (e.g., 'won by', 'tie', 'draw', 'no result', 'abandoned')
    and identifies the winning team and the team that was chasing.

    Parameters
    ----------
    api_key : str
        The API key used to authenticate with the CricAPI.
    match_id : str
        The unique match identifier in CricAPI.

    Returns
    -------
    tuple
        A tuple of the form (result_string, chasing_team_won), where:
          - result_string is the API's status string (e.g., "Team A won by X runs"),
          - chasing_team_won is 1 if the chasing team won, 0 if they lost,
            or None if the result could not be determined.

        If the match is still ongoing or data is missing, returns (None, None).
    """
    url = f"https://api.cricapi.com/v1/match_info?apikey={api_key}&id={match_id}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        match_info = data.get("data", {})
        status = match_info.get("status", "")
        teams = match_info.get("teams", [])
        score = match_info.get("score", [])

        # Check if the match status indicates a finished game
        if status and teams and score:
            status_lower = status.lower()
            finished_keywords = ["won by", "tie", "draw", "no result", "abandoned"]

            if any(keyword in status_lower for keyword in finished_keywords):
                winning_team = extract_winning_team(status, teams)
                if not winning_team:
                    logger.warning(f"Could not determine winning team from status '{status}'.")
                    return None, None

                chasing_team = determine_chasing_team(score)
                if not chasing_team:
                    logger.warning(f"Could not determine chasing team for match_id {match_id}.")
                    return None, None

                chasing_team_won = 1 if winning_team == chasing_team else 0
                return status, chasing_team_won
            else:
                # Match is still ongoing
                logger.info(f"Match {match_id} status: {status}")
                return None, None
        else:
            logger.warning(f"Missing or incomplete data for match_id {match_id}.")
            return None, None
    else:
        logger.error(
            f"Failed to fetch match info for match_id {match_id}. "
            f"Status Code: {response.status_code}"
        )
        return None, None