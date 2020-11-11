"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved.

Abstract SocialMetric class, used for evaluating a social aspect about a population.
The abstract SocialMetric class contains a basic skeleton and some implementation
details that will be shared among its children classes. Its function is to
evaluate results obtained using a certain model.
"""
from abc import ABC, abstractmethod

import numpy as np


class SocialMetric(ABC):
    """Abstract SocialMetric class.
    The Abstract SocialMetric class represents the parent class that is inherited
    if all concrete metric implementations.
    Attributes:
        _criteria: A string indicating the name of the protected attribute for this metric, also called criteria.
        _objective: A float denoting the target value the metric is aimed to reach.
        _importance: A float indicating the relative importance of this criteria when aggregated for the team.
        _rounding: An integer representing the number of decimal applied to round all values returned.
    """

    def __init__(self, criteria: str, objective: float, importance: float, rounding: int):
        """ Inits SocialMetric with its criteria, objective, importance and rounding values.
        """
        if not isinstance(criteria, str):
            raise TypeError('Argument: criteria must be a string')
        if len(criteria) == 0:
            raise ValueError('Argument: criteria is empty')
        if not isinstance(objective, (float, np.float64)):
            raise TypeError('Argument: objective must be a float')
        if objective < 0:
            raise ValueError('Argument: objective should not be negative')
        if objective > 1:
            raise ValueError('Argument: objective should not be larger than one')
        if not isinstance(importance, (float, np.float64)):
            raise TypeError('Argument: importance must be a float')
        if importance < 0:
            raise ValueError('Argument: importance should not be negative')
        if importance > 1:
            raise ValueError('Argument: importance should not be larger than one')
        if not isinstance(rounding, (int, np.int64)):
            raise TypeError('Argument: rounding should be an integer')
        if rounding < 0:
            raise ValueError('Argument: rounding should not be negative')

        self._criteria = criteria
        self._objective = objective
        self._importance = importance
        self._rounding = rounding
        super().__init__()

    @abstractmethod
    def compute(self):
        """ Evaluates the value of the implemented SocialMetric.
        Returns:
            a tuple with the value of the metric, a float, and whether the objective is reached, a boolean.
        """

    @abstractmethod
    def distance(self):
        """ Evaluates the distance value between the current value of the SocialMetric and the objective.
        Returns:
            the distance as a float
        """

    @abstractmethod
    def difficulty(self, init_value, distance, final_value):
        """ Returns an evaluation of the difficulty to reach the SocialMetric's objective, as a float.
        Args:
            init_value: the current value of the metric
            distance: the compute distance of the metric from the objective
            final_value: the metric value if all the distance changes were applied
        """

    @abstractmethod
    def visualise(self, data, richness):
        """ Plots a visualisation of the metric value.
        Args:
            data: the list of datapoints to visualise
            richness: the categories present for the metric's criteria"""

    @abstractmethod
    def get_impact(self, data, candidate_category):
        """ Computes the impact of selecting someone with a given category for this metric
        Args:
            data: a Pandas DataFrame with the people in the current team
            candidate_category: the value of the metric's criteria for the new candidate"""
