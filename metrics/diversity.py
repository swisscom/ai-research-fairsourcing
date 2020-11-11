"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved.

Diversity, used for calculating the diversity of a team.
The Diversity class contains the implementation of the diversity metric.
Its function is to evaluate results with a specific subpopulation, i.e. team.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .social_metric import SocialMetric


class Diversity(SocialMetric):
    """Diversity class. Inherits the SocialMetric class.
    The Diversity is used to calculate the diversity metric.
    """

    def __init__(self, criteria: str, objective: float, importance: float, richness: list, rounding=4):
        """ Inits Diversity with its its criteria, objective, importance, richness and rounding values
        Args:
            richness: A list of theoretical categories existing for the current criteria.
        """
        super().__init__(criteria, objective, importance, rounding)
        if len(richness) == 0:
            raise ValueError('Argument: richness is empty')
        self._richness = richness

    def _compute_hunter_gaston(self, counts: dict) -> float:
        """ Function that given the counts of people in each category returns the Hunter Gaston Diversity value.
        Args:
            counts: array of int representing the number of people in each category.
        Returns:
            a float
        """
        if len(counts) == 0:
            raise ValueError('Argument: counts is empty')
        if len(counts.keys()) != len(self._richness):
            raise ValueError('Argument: counts has a mismatch in the categories')
        if sum(counts.values()) == 0:
            raise ValueError('Argument: counts should have a non-null total')
        if any([c < 0 for c in counts.values()]):
            raise ValueError('Argument: counts has negative values')
        if any([not isinstance(c, (int, np.int, np.int64)) for c in counts.values()]):
            raise ValueError('Argument: counts has non-integer values')

        counts = list(counts.values())
        # Compute total number of people
        tot = sum(counts)
        acc = 0
        # Accumulate the nominator
        for i in range(len(self._richness)):
            acc += counts[i] * (counts[i] - 1)
        # Return the Hunter Gaston value
        return 1 - round(acc / (tot * (tot - 1)), self._rounding)

    def _get_counts(self, data: pd.DataFrame) -> dict:
        """ Function that computes the counts out of the team DataFrame.
        Args:
            data: a Pandas DataFrame with a row per person and the column with the criteria as name
        Returns:
            A Python dict with all the categories for this criteria and the count of people present
            in the team in each category.
        """
        if len(data) == 0:
            raise ValueError('Argument: data is empty')
        if not isinstance(data, pd.DataFrame):
            raise TypeError('Argument: data is not a Pandas DataFrame')
        if self._criteria not in data.columns:
            raise ValueError('Argument: data does not contain the criteria information')
        if any([v not in self._richness for v in data[self._criteria].unique()]):
            raise ValueError('Argument: data contains categories not in richness')
        # Get the counts of the correct column, after dropping the NaNs
        counts = dict(data[~data[self._criteria].isna()][self._criteria].value_counts(dropna=False))
        # In case not all categories are represented, add them with count equals 0
        if len(counts.keys()) < len(self._richness):
            for k in self._richness:
                if k not in list(counts.keys()):
                    counts[k] = 0
        return counts

    def _compute_uniform_counts(self, counts: dict) -> dict:
        """ Function that given a count distribution returns the ideal, most even distribution possible.
        Args:
            counts: a Python Dict with the categories as keys and the count of people as values.
        Returns:
            A Python dict with the categories as keys and the most even counts possible as values.
        """
        if not isinstance(counts, dict):
            raise TypeError('Argument: counts is not a dict')
        if len(counts) == 0:
            raise ValueError('Argument: counts is empty')
        # Retrieve the number of people to distribute
        tot = sum(list(counts.values()))
        if tot == 0:
            raise ValueError('Argument: counts has 0 members')
        # Retrieve the number of categories to fill
        richness = len(self._richness)
        # Compute how many people we can insert in each bucket
        base_count = np.floor(tot / richness)
        # Fill the buckets with this initial value
        buckets = [base_count] * richness
        # Compute how many remaining people we should place
        remaining_points = tot - sum(buckets)
        # Iteratively distribute the last people in the buckets (at most richness - 1)
        i = 0
        while remaining_points > 0:
            buckets[i] += 1
            remaining_points -= 1
            i += 1
        # Create the dict and return
        buckets = [int(b) for b in buckets]
        unif_groups = dict(zip(list(counts.keys()), buckets))
        return unif_groups

    def compute(self, data: pd.DataFrame) -> (float, dict):
        """ Function that computes the value of the diversity for its criteria
        Args:
            data: a Pandas DataFrame with the data of the people concerned for this metric
        Returns:
            a tuple with a float that has the diversity value and a boolean indication whether the metric's objective
            is met
        """
        if len(data) == 0:
            raise ValueError('Argument : data is empty')
        if not isinstance(data, pd.DataFrame):
            raise TypeError('Argument : data is not a Pandas DataFrame')
        if self._criteria not in data.columns:
            raise ValueError('Argument : data does not contain the criteria information')
        # From the data extract the counts
        counts = self._get_counts(data)
        # From the counts compute the true Hunter Gaston diversity
        HG = self._compute_hunter_gaston(counts)
        # From the counts compute the ideal Hunter Gaston value for the most diverse distribution
        HG_uniform = self._compute_hunter_gaston(self._compute_uniform_counts(counts))
        # Compute the standardized Hunter Gaston value
        diversity_value = round(HG / HG_uniform, self._rounding)
        # Return both the value and whether the objective is met
        return diversity_value, diversity_value >= self._objective

    def distance(self, data: pd.DataFrame) -> (dict, float):
        """ Function that computes the distance between the current value and the objective through the actions needed
            to get there.
        Args:
            data: a Pandas DataFrame with the data of the people concerned for this metric
        Returns:
            a tuple with a dict that has the diversity needed actions and a float with the achieved diversity value
            after those actions
        """
        if len(data) == 0:
            raise ValueError('Argument : data is empty')
        if not isinstance(data, pd.DataFrame):
            raise TypeError('Argument : data is not a Pandas DataFrame')
        # Get the initial counts
        counts = self._get_counts(data)

        # Initialize the counts that will receive the actions
        future_counts = counts.copy()
        # Initial diversity value
        unif_buckets = self._compute_uniform_counts(counts)
        HG_std = self._compute_hunter_gaston(counts) / self._compute_hunter_gaston(unif_buckets)
        # Loop to add a person to the team in the most appropriate bucket and stop when objective is met
        iteration = 0
        while HG_std < self._objective:
            iteration += 1
            # Add person in the minimum bucket
            future_counts[min(future_counts, key=future_counts.get)] += 1
            # Recompute the new diversity value
            HG_std = self._compute_hunter_gaston(future_counts) / self._compute_hunter_gaston(
                                                                    self._compute_uniform_counts(future_counts))
        # Isolate the decisions to be made
        increments = {}
        for k in counts.keys():
            increments[k] = future_counts[k] - counts[k]
        # Return the decisions and the final diversity value
        return increments, round(HG_std, self._rounding)

    def difficulty(self, init_value: float, distance: dict, final_value: float) -> float:
        """ Functions that quantifies the effort to meet the objective for this metric
        Args:
            init_value: a float giving the current metric value
            distance: a Python dict that gives the number of increments needed to reach the diversity objective
            final_value: the diversity value reached after making the increments
        Returns:
            A float with the difficulty score
        """
        if init_value < 0 or init_value > 1:
            raise ValueError('Argument: init_value has out of bound value')
        if final_value < 0 or final_value > 1:
            raise ValueError('Argument: final_value has out of bound value')
        if final_value < init_value:
            raise ValueError('Argument: init_value is higher than final_value')
        if init_value == final_value:
            raise ValueError('Arguments: init_value and final_value are equal')
        if len(distance) == 0:
            raise ValueError('Argument: distance is empty')
        if sum(list(distance.values())) < 0:
            raise ValueError('Argument: distance has negative changes')
        # Compute the ratio of the decision over the impact on the metric
        diff = sum(list(distance.values())) / (final_value - init_value)
        # Return the rounded value
        return round(diff, self._rounding)

    def visualise(self, data: pd.DataFrame, fig_size=(15, 6), save_fig=None, dark_theme=True):
        """ Plots a visualisation of the diversity value.
        Args:
            data: the list of datapoints to visualise
            fig_size: a tuple indicating the size of the plot, default (15, 6)
            save_fig: a file_name where to save the plot, default None (not saving)
            dark_theme: a boolean indicating if we want a lighter background, default True"""
        if len(data) == 0:
            raise ValueError('Argument : data is empty')
        if not isinstance(data, pd.DataFrame):
            raise TypeError('Argument : data is not a Pandas DataFrame')
        if self._criteria not in data.columns:
            raise ValueError('Argument : data does not contain the criteria information')
        # Init figure size
        ax = plt.figure(figsize=fig_size)
        # Handle either background or not
        if dark_theme:
            ax.patch.set_color('lightgrey')
        else:
            ax.patch.set_color('none')

        # Extract the counts
        counts = data[self._criteria].value_counts(normalize=True)
        # Normalize as percentages
        counts = counts.rename('Proportion').reset_index()
        counts = counts.sort_values(by='index').rename(columns={'index': 'criteria'})
        # Create the appropriate number of colors for the plot, depends on the richness
        aq_palette = sns.diverging_palette(0, 500, n=len(self._richness))
        sns.set_style("darkgrid")
        # Plot the values
        ax = sns.barplot(data=counts, y='criteria', x='Proportion', palette=aq_palette, ci=95)
        # Fix the x axis values, as we have percentages
        ax.set_xlim((0, 1))
        plt.show()
        # Save the figure if needed
        if save_fig is not None:
            fig = plt.gcf()
            fig.tight_layout()
            fig.savefig(save_fig, bbox_inches='tight')

    def get_impact(self, data: pd.DataFrame, candidate_category: str) -> float:
        """ Computes the impact of selecting someone with a given category for this metric
        Args:
            data: a Pandas DataFrame with the people in the current team
            candidate_category: the value of the metric's criteria for the new candidate
        Returns:
            a float with the impact of having a candidate with this category added to the team"""
        if len(data) == 0:
            raise ValueError('Argument : data is empty')
        if not isinstance(data, pd.DataFrame):
            raise TypeError('Argument : data is not a Pandas DataFrame')
        if self._criteria not in data.columns:
            raise ValueError('Argument : data does not contain the criteria information')
        if candidate_category not in self._richness:
            raise ValueError('Argument : candidate_category is not in the theoretical categories of this criteria')
        # Create the new hypothetical team, by adding the new member
        hyp_team = data[[self._criteria]].copy().reset_index().drop('index', axis=1)
        hyp_team.loc[len(hyp_team)] = candidate_category
        assert (len(hyp_team) == 1 + len(data))

        # Compute the diversity metrics for the newly created team
        new_diversity, _ = self.compute(hyp_team)
        # Compute the previous team composition metrics
        old_diversity, _ = self.compute(data)
        # Compute the difference of metrics, i.e. impact of the new employee
        metric_diff = new_diversity - old_diversity
        return metric_diff
