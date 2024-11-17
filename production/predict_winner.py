from catboost import CatBoostClassifier
import numpy as np
import os
import requests

API_KEY = '57859138-e236-4a32-9507-560e9bf590dd'
MATCH_ID = '3d012eaa-74fc-46fb-bad1-7e7f704c87a0'

def get_match_info(api_key, match_id):

    url = 'https://api.cricapi.com/v1/match_info'
    params = {
        'apikey': api_key,
        'id': match_id
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data.get('status') != 'success':
        print('Failed to retrieve match data')
        return None

    match_data = data.get('data', {})

    # Extract team names
    teams = match_data.get('teams', [])
    if len(teams) >= 2:
        team_batting_first = teams[0]
        team_batting_second = teams[1]
    else:
        team_batting_first = team_batting_second = 'Unknown'

    # Extract score information
    score_list = match_data.get('score', [])

    if score_list:
        # Assuming the last item is the latest score
        current_score = score_list[-1]
        current_inning = current_score.get('inning', '')
        runs = current_score.get('r', 0)
        wickets = current_score.get('w', 0)
        overs = current_score.get('o', 0)

    else:
        current_inning = ''
        runs = wickets = overs = 0

    # Determine if it's first or second innings
    if 'Inning 1' in current_inning:
        innings_number = 1
    elif 'Inning 2' in current_inning:
        innings_number = 2
    else:
        innings_number = 'Unknown'

    # Get total runs of previous innings if it's the second innings
    total_chasing = None
    if innings_number == 2 and len(score_list) >= 1:
        # Assuming the first item in score_list is the previous innings
        previous_innings = score_list[0]
        total_chasing = previous_innings.get('r', 0)
    elif innings_number == 1:
        total_chasing = np.nan  # Not applicable in first innings
    else:
        total_chasing = np.nan  # Unknown innings

    # Prepare the result
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


def prepare_features(match_info):
    """
    Prepares the feature vector for the model based on match_info.
    """
    # Extract values
    innings = match_info['current_innings']
    runs = match_info['current_runs']
    wickets = match_info['current_wickets']
    overs_float = match_info['current_overs']
    total_chasing = match_info['total_chasing']

    # Calculate balls bowled
    if overs_float is None or overs_float == 0:
        balls_bowled = 0
    else:
        overs = int(overs_float)
        partial_over = overs_float - overs
        balls_bowled = overs * 6 + round(partial_over * 10)
    ball_number = balls_bowled + 1  # Assuming ball numbers start at 1

    # Prepare feature vector
    features = {
        'innings': innings,
        'ball': ball_number,
        'runs': runs,
        'wickets': wickets,
        'total_chasing': total_chasing
    }

    return features


def main():

    api_key = API_KEY
    match_id = MATCH_ID

    match_info = get_match_info(api_key, match_id)

    if match_info:

        print('Match Information:')
        print(f"Team Batting First: {match_info['team_batting_first']}")
        print(f"Team Batting Second: {match_info['team_batting_second']}")
        print(f"Current Innings: {match_info['current_innings']}")
        print(f"Current Runs: {match_info['current_runs']}")
        print(f"Current Wickets: {match_info['current_wickets']}")
        print(f"Current Overs: {match_info['current_overs']}")
        if match_info['current_innings'] == 2:
            print(f"Previous Innings Total Runs: {match_info['previous_innings_total']}")

        model = CatBoostClassifier()
        model.load_model('../modelling/model.cbm')

        feature_vector = prepare_features(match_info)

        feature_names = [
            'innings',
            'ball',
            'runs',
            'wickets',
            'total_chasing'
        ]

        # Convert feature_vector to list in the correct order
        X = [feature_vector.get(f, np.nan) for f in feature_names]

        # Convert to the appropriate format (e.g., numpy array)
        X = np.array(X).reshape(1, -1)

        # Make prediction
        probability = model.predict_proba(X)[:, 1][0]
        print(f"\nPredicted Probability of {match_info['team_batting_second']} Winning: {probability * 100:.2f}%")

    else:
        print('No match information available.')
    

if __name__ == '__main__':
    main()