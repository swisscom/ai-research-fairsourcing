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

# Initialize the dataframe to generate the objectives
df = pd.DataFrame(columns=['team', 'metric', 'criteria', 'target', 'importance'])

# For each team and type of metric, generate 1 weight for all dimensions
print('Generating Metric Target and Importance')
for t in params['teams']:
    for m in params['metrics']:
        # Select in the list of strategies
        importance_strat = params['importance_levels'][np.random.choice(list(range(len(params['importance_levels']))))]
        # Report the relevant target + importance for the team, metric, criteria
        for i, c in enumerate(params['criteria']):
            df.loc[len(df)] = [t, m, c, np.random.choice(params['target_levels']), importance_strat[i]]

# Saving the file
print('Writing file')
df.to_csv(output_path + 'generated_objectives.csv')
