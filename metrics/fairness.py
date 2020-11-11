"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved.

Fairness, used for calculating the fairness of a team.
The Fairness class contains the implementation of the fairness metric.
Its function is to evaluate results with a specific subpopulation, i.e. team.
"""
import operator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .social_metric import SocialMetric


class Fairness(SocialMetric):
    """Fairness class. Inherits the SocialMetric class.
    The Fairness is used to calculate the fairness metric.
    """

    def __init__(self, criteria: str, objective: float, importance: float, max_iter=100, rounding=4):
        """ Inits Fairness with its criteria, objective, importance, max_iter and rounding values
        Args:
            max_iter: An integer giving the max number of iterations to optimize the distance."""
        super().__init__(criteria, objective, importance, rounding)
        if not isinstance(max_iter, int):
            raise TypeError('Argument: max_iter must be an integer')
        if max_iter <= 0:
            raise ValueError('Argument: max_iter must be positive')
        self._max_iter = max_iter

    def compute(self, data: pd.DataFrame, decision: str) -> (float, bool):
        """ Function that computes the value of the fairness for its criteria
        Args:
            data: a Pandas DataFrame with the data of the people concerned for this metric
            decision: a String indicating with binary column should be considered as the decision
        Returns:
            a tuple with a float that has the fairness value and a boolean indicating if the metric's objective is met
        """
        if len(data) == 0:
            raise ValueError('Argument: data is empty')
        if not isinstance(data, pd.DataFrame):
            raise TypeError('Argument: data is not a Pandas DataFrame')
        if len(decision) == 0:
            raise ValueError('Argument: decision is empty')
        if decision not in data.columns:
            raise ValueError('Argument: data does not contain the decision information: ' + decision)
        if not set(data[decision].unique()).issubset({0, 1}):
            raise ValueError('Argument: decisions in data are not binary, needs to be 0 and 1')
        # Retrieve the different groups present in the criteria columns
        groups = data[self._criteria].unique()
        # Compute the probability of positive decision for each group
        probs = np.zeros(len(groups))
        for i, l in enumerate(groups):
            probs[i] = data.loc[data[self._criteria] == l, decision].mean()
        # Compute the p-rule as the ratio of minimum over maximum probabilities
        p_rule = np.nanmin(probs) / np.nanmax(probs)
        # Round and compare to the objective
        return round(p_rule, self._rounding), p_rule >= self._objective

    def _dist_to_fairness(self, nb_min: int, len_min: int, nb_max: int, len_max: int) -> int:
        """Function that computes the distance to fairness given the min and max counts of total and positive decisions.
        The value returned is how many people should be moved from the max category to the min one to have
        a fairness metric above the objective.
        Args:
            nb_min: number of positive decisions in the minimum category
            len_min: number of people in the minimum category
            nb_max: number of positive decisions in the maximum category
            len_max: number of people in the maximum category
        Returns:
            an integer giving the number of people to move from category b to a to have a fairness value
            above the objective.
        """
        for arg in [nb_min, len_min, nb_max, len_max]:
            if not isinstance(arg, (int, np.int64)):
                raise TypeError('Argument: ' + str(arg) + ' not an integer')
            elif arg < 0:
                raise ValueError('Argument: ' + str(arg) + ' must not be negative')
        # Get the 2 probabilities
        p_min = nb_min / len_min
        p_max = nb_max / len_max
        if p_min > 1 or p_max > 1:
            raise ValueError('The arguments might have a wrong order, a probability is larger than 1.')
        if p_min > p_max:
            raise ValueError('The minimum probability is greater than maximum probability, ' +
                             'verify the order of arguments')
        # Compute a repetitive factor in the formula
        ratio = (len_min / len_max) * self._objective
        # Compute the distance to fairness
        d = int(np.ceil((1 / (1 + ratio)) * (ratio * nb_max - nb_min)))
        # Safety check that the new probabilities do indeed satisfy the fairness constraint
        min_prob = (nb_min + d) / len_min
        max_prob = (nb_max - d) / len_max
        # Making sure that we indeed have met the objective now
        if round(min_prob / max_prob, self._rounding) < self._objective:
            raise ValueError('The optimization failed')
        return abs(d)

    def distance(self, data: pd.DataFrame, decision: str) -> (dict, bool, float):
        """ Function that computes the distance between the current value and the objective through the actions
            needed to get there.
        Args:
            data: a Pandas DataFrame with the data of the people concerned for this metric
            decision: a String indicating with binary column should be considered as the decision
        Returns:
            a tuple with a dict that has the fairness needed actions, a boolean indicating if the computation converged,
             and a float with the achieved fairness value after those actions
        """
        if len(data) == 0:
            raise ValueError('Argument : data is empty')
        if not isinstance(data, pd.DataFrame):
            raise TypeError('Argument : data is not a Pandas DataFrame')
        if len(decision) == 0:
            raise ValueError('Argument: decision is empty')
        if decision not in data.columns:
            raise ValueError('Argument : data does not contain the decision information :' + decision)
        # Retrieve all the groups
        groups = data[self._criteria].dropna().unique()
        # Initialize dicts for the counts, the total number and the probability
        group_counts = {}
        group_size = {}
        group_prob = {}
        if len(groups) <= 1:
            raise ValueError('There are not enough categories in the data to compute a distance')
        # Populate the dicts with their relevant data
        for i, g in enumerate(groups):
            y = data[decision].copy()
            y = y[data[self._criteria] == g]

            group_counts[g] = y.sum()
            group_size[g] = len(y)
            group_prob[g] = group_counts[g] / group_size[g]
        # Initialize the dict to store the different changes
        changes = dict.fromkeys(group_prob.keys(), 0)
        # Get init prule
        max_group, max_prob = max(group_prob.items(), key=operator.itemgetter(1))
        min_group, min_prob = min(group_prob.items(), key=operator.itemgetter(1))
        # Start the iterations
        iterate = True
        i = 0
        # if the objective is not met, do one more round, until we met the max number of iterations
        while (round(min_prob / max_prob, self._rounding) < self._objective) & iterate:
            i += 1
            # Get the number of changes between those 2 groups
            d = self._dist_to_fairness(group_counts[min_group], group_size[min_group],
                                       group_counts[max_group], group_size[max_group])
            # Apply the returned change
            group_counts[min_group] += d
            group_counts[max_group] -= d
            iterate = (group_counts[max_group] >= 0)
            # Report in the changes
            if iterate:
                changes[min_group] += d
                changes[max_group] -= d
            # Compute the new probabilities
            for k in group_prob.keys():
                group_prob[k] = group_counts[k] / group_size[k]

            # Compute the new prule
            max_group, max_prob = max(group_prob.items(), key=operator.itemgetter(1))
            min_group, min_prob = min(group_prob.items(), key=operator.itemgetter(1))

            if i > self._max_iter:
                print('Reached maximum iterations')
                iterate = False
        # Return the changes, the convergence information and the final metric value
        return changes, iterate, round(min_prob / max_prob, self._rounding)

    def difficulty(self, init_value: float, distance: dict, final_value: float) -> float:
        """ Functions that quantifies the effort to meet the objective for this metric
        Args:
            init_value: a float giving the current metric value
            distance: a Python dict that gives the number of increments needed to reach the fairness objective
            final_value: the fairness value reached after making the increments
        Returns:
            A float with the difficulty score
        """
        if init_value < 0 or init_value > 1:
            raise ValueError('Argument : init_value has out of bound value')
        if final_value < 0 or final_value > 1:
            raise ValueError('Argument : final_value has out of bound value')
        if final_value < init_value:
            raise ValueError('Argument: init_value is higher than final_value')
        if init_value == final_value:
            raise ValueError('Arguments: init_value and final_value are equal')
        if len(distance) == 0:
            raise ValueError('Argument : distance is empty')
        if sum(list(distance.values())) != 0:
            raise ValueError('Argument: distance values do not sum to 0')
        # Measure the number of changes needed
        nb_swap = round(sum([x for x in distance.values() if x > 0]))
        # Compute the effort or difficulty
        diff = nb_swap / (final_value - init_value)
        return round(diff, self._rounding)

    def visualise(self, data: pd.DataFrame, decision: str, richness: int, fig_size=(15, 6), save_fig=None,
                  dark_theme=True):
        """ Plots a visualisation of the fairness value.
        Args:
            data: the list of datapoints to visualise
            decision: a String indicating with binary column should be considered as the decision
            richness: an integer number of categories present for the metric's criteria
            fig_size: a tuple indicating the size of the plot, default (15, 6)
            save_fig: a file_name where to save the plot, default None (not saving)
            dark_theme: a boolean indicating if we want a lighter background, default True
             """
        if len(data) == 0:
            raise ValueError('Argument: data is empty')
        if not isinstance(data, pd.DataFrame):
            raise TypeError('Argument: data is not a Pandas DataFrame')
        if len(decision) == 0:
            raise ValueError('Argument: decision is empty')
        if decision not in data.columns:
            raise ValueError('Argument: data does not contain the decision information :' + decision)
        if not isinstance(richness, (int, np.int64)):
            raise TypeError('Argument: richness should be an integer')
        if richness <= 0:
            raise ValueError('Argument: richness should be strictly positive')
        # Init figure size
        ax = plt.figure(figsize=fig_size)
        # Handle user background info
        if dark_theme:
            ax.patch.set_color('lightgrey')
        else:
            ax.patch.set_color('none')
        # Get the correct data
        filter_df = data[~data[self._criteria].isna()]
        filter_df[self._criteria] = filter_df[self._criteria].astype('category')
        # Create the appropriate number of colors
        aq_palette = sns.diverging_palette(0, 500, n=richness)
        sns.set_style("darkgrid")
        # Plot the data
        ax = sns.barplot(data=filter_df, x=decision, y=self._criteria, palette=aq_palette, ci=95)
        # Fix the axis as we have percentages
        ax.set_xlim((0, 1))
        plt.show()
        # Save the plot if needed
        if save_fig is not None:
            fig = plt.gcf()
            fig.tight_layout()
            fig.savefig(save_fig, bbox_inches='tight')

    def get_impact(self, data: pd.DataFrame, decision: str, candidate_category: str) -> float:
        """ Computes the impact of selecting someone with a given category for this metric
        Args:
            data: a Pandas DataFrame with the people in the current team
            decision: a String indicating with binary column should be considered as the decision
            candidate_category: the value of the metric's criteria for the new candidate
        Returns:
            a float with the impact of having a candidate with this category added to the team"""
        if len(data) == 0:
            raise ValueError('Argument: data is empty')
        if not isinstance(data, pd.DataFrame):
            raise TypeError('Argument: data is not a Pandas DataFrame')
        if decision not in data.columns:
            raise ValueError('Argument: data does not contain the decision information :' + decision)
        if self._criteria not in data.columns:
            raise ValueError('Argument: data does not contain the criteria information')
        if len(candidate_category) == 0:
            raise ValueError('Argument: candidate_category is empty')

        # Create the hypothetical team with the candidate inserted
        hyp_team = data[[self._criteria, decision]].copy().reset_index().drop('index', axis=1)
        hyp_team.loc[len(hyp_team)] = [candidate_category, 1]
        assert (len(hyp_team) == 1 + len(data))

        # Compute the fairness metrics for the newly created team
        new_fairness, _ = self.compute(hyp_team, decision)
        # Compute the previous team composition metrics
        old_fairness, _ = self.compute(data, decision)
        # Compute the difference of metrics, i.e. impact of the new employee
        metric_diff = new_fairness - old_fairness
        return metric_diff
