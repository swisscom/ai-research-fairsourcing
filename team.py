"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved.

Team, used for modelling the behavior of an organisational unit.
The Team class contains the implementation of the team object, using all the other classes.
Its function is to make the link between the different other elements of this project.
"""
from itertools import product
from math import pi

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .candidate import Candidate
from .decision_maker import DecisionMaker
from .metrics.diversity import Diversity
from .metrics.fairness import Fairness


class Team:
    """Team class.
    The Team is used to model the behavior of an organisational unit.
    """

    def __init__(self, team_id: str, manager: DecisionMaker, recruiters: list, team_df: pd.DataFrame,
                 diversity_metrics: list, cand_df: pd.DataFrame, fairness_metrics: list):
        """ Inits Team with its team_id, manager, recruiters, team_df, diversity metrics, cand_df and fairness metrics
        Args:
            team_id: A string giving a unique name to the considered team
            manager: A DecisionMaker object that has decision 'hired'
            recruiters: A list of DecisionMaker objects that have decision 'transfer'
            team_df: A Pandas DataFrame with a row per employee in this team
            diversity_metrics: A list of Diversity objects, one per criteria
            cand_df: A Pandas DataFrame with a row per candidate to this team
            fairness_metrics: A list of Fairness objects, one per criteria"""
        if len(team_id) == 0:
            raise ValueError('Argument: team_id is empty')
        if not isinstance(manager, DecisionMaker):
            print(type(manager))
            raise TypeError('Argument: manager should be a DecisionMaker')
        if not isinstance(recruiters, list):
            raise TypeError('Argument: recruiters should be a list')
        if any([not isinstance(r, DecisionMaker) for r in recruiters]):
            raise TypeError('Argument: recruiters should all be DecisionMaker instances')
        if not isinstance(team_df, pd.DataFrame):
            raise TypeError('Argument: team_df is not a DataFrame')
        if len(team_df) == 0:
            raise ValueError('Argument: team_df is empty')
        if not isinstance(cand_df, pd.DataFrame):
            raise TypeError('Argument: cand_df is not a DataFrame')
        if len(cand_df) == 0:
            raise ValueError('Argument: cand_df is empty')
        if not isinstance(diversity_metrics, list):
            raise TypeError('Argument: diversity_metrics should be a list')
        if any([not isinstance(d, Diversity) for d in diversity_metrics]):
            raise TypeError('Argument: diversity_metrics should all be Diversity instances')
        if round(sum([d._importance for d in diversity_metrics]), 3) != 1:
            raise ValueError('Argument: the importance sum of diversity_metrics should be equal to 1')
        if not isinstance(fairness_metrics, list):
            raise TypeError('Argument: fairness_metrics should be a list')
        if any([not isinstance(f, Fairness) for f in fairness_metrics]):
            raise TypeError('Argument: fairness_metrics should all be Fairness instances')
        if round(sum([f._importance for f in fairness_metrics]), 3) != 1:
            raise ValueError('Argument: the importance sum of fairness_metrics should be equal to 1')
        self._team_id = team_id
        self._manager = manager
        self._recruiters = recruiters
        self._diversity_state = team_df
        self._diversity_metrics = diversity_metrics
        self._fairness_state = cand_df
        self._fairness_metrics = fairness_metrics

    def compute_social_metric(self, metric: str, decision=None) -> float:
        """ Function that returns the aggregated value of the metric type specified, across all the criteria.
        Args:
            metric: a string that indicates if we are facing a Diversity or Fairness object
            decision: a string that indicates the binary column to use as decision if we have a Fairness object,
                    default None
        Returns:
            A float that is the weighted average of the diversity values with their importances.
        """
        if not isinstance(metric, str):
            raise TypeError('Argument: metric should be a string')
        if len(metric) == 0:
            raise ValueError('Argument: metric is empty')
        if metric.lower() not in ['diversity', 'fairness']:
            raise ValueError('Argument: this metric type is not supported')
        computed_values = []
        weights = []
        # Apply specificities of both types of SocialMetrics to collect the values and the weights
        if metric.lower() == 'diversity':
            for m in self._diversity_metrics:
                curr_val, _ = m.compute(self._diversity_state)
                computed_values.append(curr_val)
                weights.append(m._importance)

        elif metric.lower() == 'fairness':
            if decision is None:
                raise ValueError('Argument: decision is missing')
            for m in self._fairness_metrics:
                curr_val, _ = m.compute(self._fairness_state, decision)
                computed_values.append(curr_val)
                weights.append(m._importance)

        # Return the weighted average of the computed values and their importances
        return np.average(computed_values, weights=weights)

    def visualise_social_metric(self, metric: str, decision=None, color_id=0, fig_size=(15, 6), save_fig=None,
                                dark_theme=True):
        """ A function that displays on a spider chart the value of the metric for all criteria, as well as their
            respective objectives.
        Args:
            metric: a string that indicates if we are facing a Diversity or Fairness object
            decision: a string that indicates the binary column to use as decision if we have a Fairness object
            color_id: an index in case we'd like to change the color of the spider chart, default 0
            fig_size: a tuple indicating the size of the plot, default (15, 6)
            save_fig: a file_name where to save the plot, default None (not saving)
            dark_theme: a boolean indicating if we want a lighter background, default True
        """
        if not isinstance(metric, str):
            raise TypeError('Argument: metric should be a string')
        if len(metric) == 0:
            raise ValueError('Argument: metric is empty')
        if metric.lower() not in ['diversity', 'fairness']:
            raise ValueError('Argument: this metric type is not supported')
        if not isinstance(color_id, (int, np.int64)):
            raise TypeError('Argument: color_id must be an int')
        if color_id < 0:
            raise ValueError('Argument: color_id must be positive')
        # Prepare to receive the criteria name list, the objective values and the computed values
        criteria = []
        objectives = []
        computed_values = []
        # Deal with the specificities of each type of SocialMetric
        if metric.lower() == 'diversity':
            for m in self._diversity_metrics:
                criteria.append(m._criteria)
                objectives.append(m._objective)
                computed_values.append(m.compute(self._diversity_state)[0])

        elif metric.lower() == 'fairness':
            if decision is None:
                raise ValueError('Argument: decision is missing')
            for m in self._fairness_metrics:
                criteria.append(m._criteria)
                objectives.append(m._objective)
                computed_values.append(m.compute(self._fairness_state, decision)[0])

        assert (len(criteria) == len(computed_values))
        # Set up the figure to start plotting
        ax = plt.figure(figsize=fig_size)
        if dark_theme:
            ax.patch.set_color('lightgrey')
        else:
            ax.patch.set_color('none')
        # Create the palette with enough colors
        palette = plt.cm.get_cmap("tab20", 10)
        color = palette(color_id)
        # Construct the title
        title = metric + ' values for the team ' + self._team_id
        # Create the graph
        # Number of criteria
        n = len(criteria)
        # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
        angles = [i / float(n) * 2 * pi for i in range(n)]
        angles += angles[:1]

        # Initialise the spider plot
        ax = plt.subplot(1, 1, 1, polar=True)

        # If you want the first axis to be on top:
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)

        # Draw one axe per variable + add labels labels yet
        plt.xticks(angles[:-1], criteria, color='grey', size=10)

        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=8)
        plt.ylim(0, 1)

        # Create the list of values to plot for the diversity (append again the first value to close the area)
        values = computed_values.copy()
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
        ax.fill(angles, values, color=color, alpha=0.4)
        # Plot the values of the objectives
        ax.scatter(angles[:-1], objectives, color='r')

        # Add a title
        plt.title(title, size=11, color=color, y=1.1)
        # Saving the figure if needed
        if save_fig is not None:
            fig = plt.gcf()
            fig.tight_layout()
            fig.savefig(save_fig, bbox_inches='tight')

    def compute_cand_impacts(self, candidate: Candidate) -> (list, list, list, list):
        """ A function that returns the impacts of the candidate on both types of SocialMetric and all criteria,
            alongside their importance scores
        Args:
            candidate : a Candidate object with the demographic attributes
        Returns:
            4 lists, representing the diversity impacts, the diversity importance scores, the fairness impacts,
            the fairness importance scores
        """
        if not isinstance(candidate, Candidate):
            raise TypeError('Argument: candidate must be a Candidate instance')
        if candidate._team_id != self._team_id:
            raise ValueError('Argument: candidate has a different team_id')
        if any([c not in candidate._attributes.keys() for c in [m._criteria for m in self._diversity_metrics]]):
            raise ValueError('Argument: candidate has a mismatch of the attributes with the diversity metrics')
        if any([c not in candidate._attributes.keys() for c in [m._criteria for m in self._fairness_metrics]]):
            raise ValueError('Argument: candidate has a mismatch of the attributes with the fairness metrics')
        if candidate._decision not in self._fairness_state.columns:
            raise ValueError('Argument: candidate decision is not in this team fairness_state')
        # Collect the diversity impacts and the relative importance
        div_impacts = []
        div_weights = []
        for m in self._diversity_metrics:
            div_impacts.append(m.get_impact(self._diversity_state, candidate._attributes[m._criteria]))
            div_weights.append(m._importance)

        # Do the same for the fairness metrics
        fair_impacts = []
        fair_weights = []
        for m in self._fairness_metrics:
            fair_impacts.append(m.get_impact(self._fairness_state, candidate._decision,
                                             candidate._attributes[m._criteria]))
            fair_weights.append(m._importance)

        return div_impacts, div_weights, fair_impacts, fair_weights

    def compute_cand_agg_impacts(self, candidate: Candidate) -> (float, float):
        """ A function that returns the aggregated diversity and fairness impact of a candidate
        Args:
            candidate: A Candidate object that applies to this team
        Returns:
            A tuple of floats that represent the overall diversity and fairness impact of the candidate
        """
        # Retrieve all impacts and weights
        div_impacts, div_weights, fair_impacts, fair_weights = self.compute_cand_impacts(candidate)
        # Return the weighted averages
        return np.average(div_impacts, weights=div_weights), np.average(fair_impacts, weights=fair_weights)

    def _visualise_cand_impact(self, candidate: Candidate, metric: str, impacts: list, weights: list, show_avg=True,
                               yrange=None, fig_size=(5, 5), save_fig=None, dark_theme=True):
        """ A Function that visualise the impact of a candidate on one type of SocialMetric, for all criteria,
            potentially with the weighted average.
        Args:
            candidate: A Candidate instance that applies to this team
            metric: A String indicating if we deal with a Diversity or Fairness object
            impacts: The list of floats that represent the impact of the candidate
            weights: The list of weights for this SocialMetric
            show_avg: A boolean indicating if the overall impact should be shown, default True
            yrange: A tuple of integers indicating the range of the y axis, in case the values are not readable,
                    default None
            fig_size: a tuple indicating the size of the plot, default (15, 6)
            save_fig: a file_name where to save the plot, default None (not saving)
            dark_theme: a boolean indicating if we want a lighter background, default True
        """
        if not isinstance(candidate, Candidate):
            raise TypeError('Argument: candidate should be a Candidate instance')
        if len(candidate._attributes.keys()) != len(impacts):
            raise ValueError('Argument: impacts length does not match the number of attributes')
        if candidate._team_id != self._team_id:
            raise ValueError('Argument: candidate has a different team_id')
        if not isinstance(metric, str):
            raise TypeError('Argument: metric should be a string')
        if len(metric) == 0:
            raise ValueError('Argument: metric is empty')
        if metric.lower() not in ['diversity', 'fairness']:
            raise ValueError('Argument: this metric type is not supported')
        if not isinstance(impacts, list) or any([not isinstance(i, float) for i in impacts]):
            raise TypeError('Argument: impacts should a list of floats')

        # Prepare the figure
        ax = plt.figure(figsize=fig_size)
        if dark_theme:
            ax.patch.set_color('lightgrey')
        else:
            ax.patch.set_color('none')
        # Prepare the data as we need it for the visualisation
        df = pd.DataFrame({'criteria': list(candidate._attributes.keys()), 'impact': impacts})
        colors = ['g' if v >= 0 else 'r' for v in impacts]
        _ = sns.barplot(data=df, y='impact', x='criteria', palette=colors)
        plt.ylim(yrange)
        plt.title(metric + ' impact for cand ' + candidate._pers_id + ' and team ' + self._team_id)
        #  Show aggregated value if needed
        if show_avg:
            if not isinstance(weights, list) or any([not isinstance(w, float) for w in weights]):
                raise TypeError('Argument: weights should a list of floats')
            if len(impacts) != len(weights):
                raise ValueError('Arguments: impacts and weights should have the same size')
            if round(sum(weights), 3) != 1:
                raise ValueError('Argument: sum of weights should be equal to 1')
            plt.hlines(np.average(impacts, weights=weights), xmin=-0.5, xmax=2.5, linestyles='--', linewidth=1)
        # Saving figure if needed
        if save_fig is not None:
            fig = plt.gcf()
            fig.tight_layout()
            fig.savefig(save_fig, bbox_inches='tight')

        plt.show()

    def visualise_cand_impacts(self, candidate: Candidate, show_avg=True, yrange=None, fig_size=(5, 5), save_fig=None,
                               dark_theme=True):
        """ A function that visualise the impact of the candidate on both types of SocialMetrics.
        Args:
            candidate: A Candidate object that applies to this team
            show_avg: A boolean indicating if the overall impact should be shown, default True
            yrange: A tuple of integers indicating the range of the y axis, in case the values are not readable,
                    default None
            fig_size: a tuple indicating the size of the plot, default (15, 6)
            save_fig: a file_name where to save the plot, default None (not saving)
            dark_theme: a boolean indicating if we want a lighter background, default True
        """
        # Get all the impacts
        div_impacts, div_weights, fair_impacts, fair_weights = self.compute_cand_impacts(candidate)
        # Visualise the diversity impact
        _ = self._visualise_cand_impact(candidate, 'Diversity', div_impacts, div_weights, show_avg, yrange, fig_size,
                                        save_fig, dark_theme)
        # Visualise the fairness impact
        _ = self._visualise_cand_impact(candidate, 'Fairness', fair_impacts, fair_weights, show_avg, yrange, fig_size,
                                        save_fig, dark_theme)

    def compute_optimal_profile(self, attr_space: dict, decision: str, div_imp=0.5, n=1) -> pd.DataFrame:
        """ A function that retrieves N optimal profiles for the given team according to the attribute space
        Args:
            attr_space: A python dict with the criteria as keys and the theoretical categories as values
            decision: a string that indicates the binary column to use as decision for the Fairness metrics
            div_imp: a float with the relative importance of the Diversity overall impact with respect to the Fairness
                    one
            n: an integer representing the number of optimal profiles to return.
        Returns:
            A Pandas DataFrame with the demographic information of the optimal profiles as well as their impact scores.
        """
        if not isinstance(attr_space, dict):
            raise TypeError('Argument: attr_space must be a dict')
        if len(attr_space) == 0:
            raise ValueError('Argument: the attr_space is empty')
        if any([c not in self._diversity_state.columns for c in attr_space.keys()]):
            raise ValueError('Argument: at least one key of the attr_space is not in the diversity_state')
        if any([c not in self._fairness_state.columns for c in attr_space.keys()]):
            raise ValueError('Argument: at least one key of the attr_space is not in the fairness_state')
        if decision not in self._fairness_state.columns:
            raise ValueError('Argument: the decision is not included in the Fairness State')
        if div_imp < 0 or div_imp > 1:
            raise ValueError('Argument: the div_impact is outside of the bounds')
        if not isinstance(n, (int, np.int64)):
            raise TypeError('Argument: N must be an integer')
        if n <= 0:
            raise ValueError('Argument: N must be a positive number')
        # Prepare to store both impacts depending on the profile's attribute
        impacts = pd.DataFrame(columns=['attr', 'div_impact', 'fair_impact'])
        # Compute all possible profiles with all the combination of the attribute space
        all_profiles = [dict(zip(attr_space.keys(), i)) for i in list(product(*attr_space.values()))]
        # For each profile, retrieve both impact values and store
        for p in all_profiles:
            curr_cand = Candidate('0', self._team_id, p, decision)
            div_impact, fair_impact = self.compute_cand_agg_impacts(curr_cand)
            impacts.loc[len(impacts)] = [p, div_impact, fair_impact]
        # Compute the weighted sum of the impact with the weight for diversity
        impacts['tot_impact'] = round(div_imp * impacts['div_impact'] + (1 - div_imp) * impacts['fair_impact'], 4)
        # Sort the profiles according to decreasing impact
        impacts = impacts.sort_values(by='tot_impact', ascending=False)
        # Return the desired top N
        if n > len(impacts):
            raise ValueError('Argument n: max number of profiles is ', len(impacts))
        return impacts.head(n)

# def social_metric_difficulty_overview():
# def visualise_chronology(data, criteria):
