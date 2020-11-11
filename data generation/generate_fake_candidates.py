"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved."""

import sys

import numpy as np
import pandas as pd
import yaml

yaml_path = sys.argv[1]
output_path = sys.argv[2]
# Load all the necessary parameters
params = yaml.load(open(yaml_path, 'r'), Loader=yaml.FullLoader)

# Initialize the dataframe to generate the candidates
print('Generating candidates')
df = pd.DataFrame(columns=['cand_id', 'gender', 'language', 'age', 'job_id', 'team', 'transfer'])

# Generate candidates with job + prob transfer
for i in range(params['NB_CANDIDATES']):
    # For each employee, select according to the probabilities each protected attribute and whether they were
    # transferred to the line manager
    df.loc[len(df)] = [(i + 1), np.random.choice(params['genders'], p=params['gender_probs']),
                       np.random.choice(params['language'], p=params['language_probs']),
                       np.random.choice(params['age'], p=params['age_probs']),
                       np.random.choice(range(params['NB_JOBS'])),
                       np.random.choice(params['teams'], p=params['team_probs']),
                       np.random.choice([0, 1], p=[1 - params['PROB_TRANSFER'], params['PROB_TRANSFER']])]

# Initialize the hired column to retrieve which candidates are selected
df['hired'] = [0] * len(df)

# For each job, identified with their id, we uniformly random pick one candidate to be the hired one.
for j in range(params['NB_JOBS']):
    tmp_df = df[df['job_id'] == j]
    hired = np.random.choice(tmp_df['cand_id'].values)
    df.loc[df['cand_id'] == hired, 'hired'] = 1

# Saving the file
print('Writing file')
df.to_csv(output_path + 'generated_candidates.csv')
