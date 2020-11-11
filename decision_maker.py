"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved.

DecisionMaker, used for modeling the interaction of decision maker in the sourcing process.
The DecisionMaker class contains the implementation of the different interactions.
Its function is to evaluate results with a specific subpopulation, i.e. team.
"""
from itertools import product

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .person import Person


class DecisionMaker(Person):
    """DecisionMaker class.
    The decide is used to compute the fairness of a decision maker.
    """

    def __init__(self, pers_id: str, team_id: str, decision_name: str, attributes: dict,
                 min_candidates_bucket=10):
        """ Inits DecisionMaker with its pers_id, team_id, decision_name, attributes, min_candidates_bucket
        Args:
            pers_id: A string that uniquely identifies this employee
            team_id: A string that uniquely identifies the team the employee belongs to
            decision_name: A string that indicates the binary column containing the decision made by this actor
            attributes: A Python dict that contains the criteria and categories the DecisionMaker belongs to
            min_candidates_bucket: An integer that is the minimum number of candidates necessary to visualise
                                on the landscape"""
        if not isinstance(min_candidates_bucket, int):
            raise TypeError('Argument: the min_candidates_bucket must be an integer')
        if min_candidates_bucket <= 0:
            raise ValueError('Argument: the min_candidates_bucket must be a positive integer')
        if len(decision_name) == 0:
            raise ValueError('Argument: the decision_name has length 0.')

        super().__init__(pers_id, team_id, attributes)
        self._decision = decision_name
        self._min_candidates = min_candidates_bucket

    def _compute_landscape(self, data: pd.DataFrame) -> pd.DataFrame:
        """ A function that groups all the employees and returns the necessary counts for the landscape
        Args:
            data: a Pandas DataFrame with a row per candidate
        Returns:
            A Pandas DataFrame that has the groups of candidates per demographic subpopulation, with their probability
            of positive decision
        """
        if len(data) == 0:
            raise ValueError('Argument : data is empty')
        if not isinstance(data, pd.DataFrame):
            raise TypeError('Argument : data is not a Pandas DataFrame')

        if 'age' not in data.columns:
            raise ValueError('Argument : data is missing age column.')
        if 'gender' not in data.columns:
            raise ValueError('Argument : data is missing gender column.')
        if 'language' not in data.columns:
            raise ValueError('Argument : data is missing language column.')
        if self._decision not in data.columns:
            raise ValueError('Argument : data is missing ', self._decision, ' column.')
        # Get all the combinations of profile possible
        products = product(sorted(data['age'].unique()), data['gender'].unique(),
                           data['language'].unique())
        # Prepare to store the metrics for each profile
        landscape = pd.DataFrame(columns=['age', 'gender', 'language', 'total', 'pos_count', 'probability'])
        # Go over all the profiles of age, gender and language
        for a, g, l in products:
            # Extract the temporary sub-dataframe
            curr_df = data[(data['age'] == a) & (data['gender'] == g) & (data['language'] == l)]
            # Number of candidates is the length
            total = len(curr_df)
            if total != 0 and sum(curr_df[self._decision].values) > 0:
                # If more than 0, count the number of positive decisions
                pos_count = curr_df[self._decision].value_counts()[1]
                # Compute the probability
                prob = pos_count / total
            else:
                pos_count = 0
                prob = 0
            # Store the data
            landscape.loc[len(landscape)] = [a, g, l, float(total), pos_count, prob]

        return landscape

    def _distance_candidate(self, cand_attributes: dict) -> int:
        """ Function that returns how many criteria differ between the candidate and the DecisionMaker
        Args:
            cand_attributes: A Python dict with the categories of each criteria that the candidate belongs to
        Returns:
            An integer that represent the number of different attributes the DecisionMaker has from the given candidate
        """
        if len(cand_attributes) == 0:
            raise ValueError('Argument: cand_attributes are empty')
        if len(self._attributes.keys()) != len(cand_attributes.keys()):
            raise ValueError(
                'Argument: cand_attributes does not have the same number of the attributes of Decision Maker')
        if any([attr not in cand_attributes.keys() for attr in self._attributes.keys()]):
            raise ValueError('Argument: cand_attributes does not match the attributes of Decision Maker')
        # Initialize the distance to 0
        d = 0
        # For each attribute,
        for attr in self._attributes.keys():
            # Compare the DecisionMaker's value to the Candidate's
            if self._attributes[attr] != cand_attributes[attr]:
                # If different increment the distance
                d += 1
        return d

    @staticmethod
    def _prule(distance_probs: dict) -> float:
        """ A function that computes the fairness value of a DecisionMaker's past decisions
        Args:
            distance_probs: a dict that given a distance with the recruiter returns the probability of positive decision
        Returns:
            A float that is the fairness value of this DecisionMaker
        """
        if len(distance_probs) == 0:
            raise ValueError('Argument: distance_probs is empty')
        if any([p < 0 for p in distance_probs.values()]):
            raise ValueError('Argument: distance_probs has negative values')
        if any([p > 1 for p in distance_probs.values()]):
            raise ValueError('Argument: distance_probs has values larger than 1')
        # Extract the min and max probabilities
        max_p = max(distance_probs.values())
        if max_p == 0:
            raise ValueError('Argument: distance_probs has max probability 0')
        # Compute the ratio of min over max
        return np.round(min(distance_probs.values()) / max_p, 4)

    def get_fairness(self, data: pd.DataFrame) -> (float, dict):
        """ A function that returns the Fairness value of this DecisionMaker given their history and their probability
            of positive decisions given the distance to the candidate.
        Args:
            data: a Pandas DataFrame that has one row per candidate and their decision
        Returns:
            A tuple of a float for the fairness value and a dict for the probabilities given the distances
        """
        if len(data) == 0:
            raise ValueError('Argument : data is empty')
        if not isinstance(data, pd.DataFrame):
            raise TypeError('Argument : data is not a Pandas DataFrame')
        # Get the grouped data
        landscape = self._compute_landscape(data)
        # Initialize the counts and probabilities for each distance
        distance_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        distance_probs = {0: 0, 1: 0, 2: 0, 3: 0}

        for i, r in landscape.iterrows():
            # Group the candidates according to their distance to the DecisionMaker
            d = self._distance_candidate(r[list(self._attributes.keys())].to_dict())
            distance_counts[d] += r['total']
            distance_probs[d] += r['pos_count']
        # Ensure that we consider the only the groups that have enough candidates
        for d in [0, 1, 2, 3]:
            if distance_counts[d] >= self._min_candidates:
                distance_probs[d] /= distance_counts[d]
            else:
                distance_probs.pop(d, None)
        # Compute the p-rule
        return self._prule(distance_probs), distance_probs

    def visualise_fairness(self, data: pd.DataFrame, distance_probs: dict, fig_size=(5, 5), save_fig=None,
                           dark_theme=True):
        """ A function that displays the different probabilities to visualise the fairness level
        Args:
            data: a Pandas DataFrame with the decision history of this DecisionMaker
            distance_probs: a Python dict with the probabilities of positive decisions given a distance
            fig_size: a tuple indicating the size of the plot, default (5, 5)
            save_fig: a file_name where to save the plot, default None (not saving)
            dark_theme: a boolean indicating if we want a lighter background, default True
        """
        if len(data) == 0:
            raise ValueError('Argument : data is empty')
        if not isinstance(data, pd.DataFrame):
            raise TypeError('Argument : data is not a Pandas DataFrame')
        if len(distance_probs) == 0:
            raise ValueError('Argument: distance_probs is empty')
        # Compute the counts of decisions
        counts = data[self._decision].value_counts(dropna=False)
        # Compute the overall probability
        avg_prob = counts[1] / (counts[1] + counts[0])
        # Prepare the figure
        _ = plt.figure(figsize=fig_size)
        if dark_theme:
            sns.set(rc={'figure.facecolor': 'lightgrey'})
        else:
            sns.set(rc={'figure.facecolor': 'none'})
        # Plot the data
        plt.axhline(y=avg_prob, color='black', ls='--', linewidth=1, alpha=0.9)
        sns.barplot(list(distance_probs.keys()), list(distance_probs.values()))
        plt.title("Distance distribution for given recruiter")
        plt.ylim((0, 1))
        # Saving figure if needed
        if save_fig is not None:
            fig = plt.gcf()
            fig.tight_layout()
            fig.savefig(save_fig, bbox_inches='tight')
        plt.show()

    def visualise_landscape(self, data: pd.DataFrame, fig_size=(25, 10), save_fig=None, dark_theme=True):
        """ A function that visualises in 5D the decision patterns of the given DecisionMaker
        Criteria -- You have one dimension across the plot, the second for the lines, the third with the columns.
        Decisions -- The size of the bubble shows how many candidates were in that group and the color is the
                    probability of positive decision
        The green star locates the attributes of the recruiter.
        Args:
            data: a Pandas DataFrame with the decision history of this DecisionMaker
            fig_size: a tuple indicating the size of the plot, default (5, 5)
            save_fig: a file_name where to save the plot, default None (not saving)
            dark_theme: a boolean indicating if we want a lighter background, default True
        """
        if len(data) == 0:
            raise ValueError('Argument : data is empty')
        if not isinstance(data, pd.DataFrame):
            raise TypeError('Argument : data is not a Pandas DataFrame')
        # Compute the landscape information
        landscape = self._compute_landscape(data)
        # Prepare the figure
        fig, axes = plt.subplots(1, 3, figsize=fig_size)
        if dark_theme:
            fig.patch.set_color('lightgrey')
        else:
            fig.patch.set_color('none')
        # For each sub-dimensions, plot the grid
        for i, g in enumerate(['M', 'F', 'O']):
            curr_df = landscape[landscape['gender'] == g]

            # Plot  landscape
            plt.rc('axes', axisbelow=True)
            axes[i].grid(color='grey', linestyle='-', linewidth=1, alpha=0.4)
            axes[i].scatter(curr_df['language'], curr_df['age'], s=curr_df['total'] * 10, c=curr_df['probability'],
                            cmap="YlOrBr", alpha=1, edgecolors="grey", linewidth=0.5)
            axes[i].set_yticks(curr_df['age'].unique())
            axes[i].set_title('Candidate Gender : ' + g)
            axes[i].set_xlabel("Candidate Language")
            axes[i].set_ylabel("Candidate Age")

        # Plot balloon size legend
        legend2_line2d = list()
        legend2_line2d.append(mlines.Line2D([0], [0],
                                            linestyle='none',
                                            marker='o',
                                            alpha=1,
                                            markersize=np.sqrt(10),
                                            markerfacecolor='none',
                                            markeredgecolor='black'))
        legend2_line2d.append(mlines.Line2D([0], [0],
                                            linestyle='none',
                                            marker='o',
                                            alpha=1,
                                            markersize=np.sqrt(100),
                                            markerfacecolor='none',
                                            markeredgecolor='black'))
        legend2_line2d.append(mlines.Line2D([0], [0],
                                            linestyle='none',
                                            marker='o',
                                            alpha=1,
                                            markersize=np.sqrt(1000),
                                            markerfacecolor='none',
                                            markeredgecolor='black'))

        _ = plt.legend(legend2_line2d,
                       ['1', '10', '100'],
                       title='Total Candidates',
                       numpoints=1,
                       fontsize=10,
                       bbox_to_anchor=(1., 0.8),  # loc='best',
                       frameon=False,
                       labelspacing=3,
                       handlelength=5,
                       borderpad=4
                       )

        # Add the information about the DecisionMaker
        g_idx = ['M', 'F', 'O'].index(self._attributes['gender'])
        axes[g_idx].scatter(self._attributes['language'], self._attributes['age'], s=150, c='green', marker="*")
        # Saving the figure if needed
        if save_fig is not None:
            fig = plt.gcf()
            fig.tight_layout()
            fig.savefig(save_fig, bbox_inches='tight')

        plt.show()
