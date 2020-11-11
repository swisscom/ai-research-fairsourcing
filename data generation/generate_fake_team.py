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

# Initialize the dataframe to generate the employees
print('Generating NB_EMPLOYEES')
df = pd.DataFrame(columns=['emp_id', 'gender', 'language', 'age', 'team'])

# Generate employees with their random team and demographics
for i in range(params['NB_EMPLOYEES']):
    df.loc[len(df)] = [(i + 1), np.random.choice(params['genders'], p=params['gender_probs']),
                       np.random.choice(params['language'], p=params['language_probs']),
                       np.random.choice(params['age'], p=params['age_probs']),
                       np.random.choice(params['teams'], p=params['team_probs'])]

# Select one employee from each team to be manager
df['manager'] = [0] * len(df)
for t in params['teams']:
    # Select all the employee ids of the current team and select one at random
    manager_id = np.random.choice(df[df['team'] == t]['emp_id'].unique())
    # Give the manager label to this employee
    df.loc[df['emp_id'] == manager_id, 'manager'] = 1

# Saving the file
print('Writing file')
df.to_csv(output_path + 'generated_team.csv')
