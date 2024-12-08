import json
import numpy as np
import os
import pandas as pd

COLUMNS = [
    'matchid',
    'innings',
    'over',
    'ball',
    'runs',
    'wickets',
    'chasing_team_won',
    'total_chasing'
]


def process_json(folder, filename):
     
    file_path = os.path.join(folder, filename)
     
    with open(file_path) as json_file:

        data = json.load(json_file)

        match_df = pd.DataFrame(columns=COLUMNS)

        matchid = filename.split('.')[0]

        year = data['info']['dates'][0].split('-')[0]

        total_runs = [0, 0]
        total_wickets = [0, 0]

        if len(data["innings"]) == 2:

            for i in range(2):

                ball = 0
                extra_runs = 0

                for j, over in enumerate(data["innings"][i]["overs"]):
                    
                    for k, delivery in enumerate(over["deliveries"]):
                        
                        runs = delivery["runs"]["total"]

                        if "extras" in delivery and ('wides' in delivery["extras"] or 'noballs' in delivery["extras"]):
                                
                            extra_runs += runs

                        else:

                            total_runs[i] += runs + extra_runs # account for runs from wides / noballs 
                    
                            wicket = 0
                            if "wickets" in delivery:
                                wicket = len(delivery["wickets"])
                                total_wickets[i] += wicket

                            ball += 1

                            match_df = pd.concat(
                                [
                                    match_df, pd.DataFrame({
                                        'matchid': [matchid],
                                        'innings': [i + 1],
                                        'over': [j],
                                        'ball': [ball],
                                        'runs': [total_runs[i]],
                                        'wickets': [total_wickets[i]],
                                        'chasing_team_won': None,
                                        'total_chasing': None
                                    })
                                ]
                            , ignore_index=True
                            )

                            extra_runs = 0      

                if extra_runs > 0:

                    total_runs[i] += extra_runs

                    last_row_idx = match_df[match_df['innings'] == (i + 1)].index[-1]
                    match_df.at[last_row_idx, 'runs'] += extra_runs

                    extra_runs = 0            

        match_df['chasing_team_won'] = np.where(total_runs[0] > total_runs[1], 0, np.where(total_runs[1] > total_runs[0], 1, None))
        match_df['total_chasing'] = np.where(match_df['innings'] == 2, total_runs[0], np.nan)

        return match_df


def read_and_process_data():

    folder = 'historical_data'

    files = os.listdir(folder)

    match_dfs = []

    for filename in files:

        match_df = process_json(folder, filename)
        if match_df['ball'].max() <= 120: # small number of data errors where number of balls > 120
            match_dfs.append(match_df)

    preprocessed_df = pd.concat(match_dfs, ignore_index=True)
    preprocessed_df.set_index('matchid', inplace=True)
    
    return preprocessed_df


def main():

    preprocessed_df = read_and_process_data()

    preprocessed_df.to_csv('preprocessed_data.csv')

    
if __name__ == '__main__':
    main()