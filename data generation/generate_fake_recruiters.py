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

# Initialize the dataframe to generate the recruiters
print('Generating recruiters')
df = pd.DataFrame(columns=['recr_id', 'gender', 'language', 'age'])

# Generate recruiters with demographics
for i in range(params['NB_RECRUITERS']):
    df.loc[len(df)] = [(i + 1), np.random.choice(params['genders'], p=params['gender_probs']),
                       np.random.choice(params['language'], p=params['language_probs']),
                       np.random.choice(params['age'], p=params['age_probs'])]

# Distribute the jobs across the recruiters
cand_df = pd.read_csv('../data/generated_candidates.csv')
# Retrieve all the different job ids
job_ids = cand_df['job_id'].unique()
job_df = pd.DataFrame(columns=['recr_id', 'job_id'])
# For each job id, uniformly random select one recruiter
for job in job_ids:
    job_df.loc[len(job_df)] = [np.random.choice(df['recr_id'].unique()), job]

# Merge the data
df = pd.merge(df, job_df, on=['recr_id', 'recr_id'])

# Saving the file
print('Writing file')
df.to_csv(output_path + 'generated_recruiters.csv')
