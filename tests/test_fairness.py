"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved.
"""

import pandas as pd
import pytest

from ..metrics.fairness import Fairness


# Evaluate the constructor
@pytest.mark.parametrize('attr, error',
                         [(['', 0.9, 0.9,  10, 2], ValueError),
                          (['age', -0.9, 0.9,  10, 2], ValueError),
                          (['age', 1.1, 0.9,  10, 2], ValueError),
                          (['age', 0.9, -0.9,  10, 2], ValueError),
                          (['age', 0.9, 1.1,  10, 2], ValueError),
                          (['age', 0.9, 0.9,  10, 2.2], TypeError),
                          (['age', 0.9, 0.9,  10, -2], ValueError),
                          (['age', 0.9, 0.9,  1.1, 2], TypeError),
                          (['age', 0.9, 0.9,  -1, 2], ValueError)])
def test_constructor_errors(attr, error):
    with pytest.raises(error):
        Fairness(*attr)


# Evaluate compute
fair_age = Fairness('age', 1., 1., 5, 2)
cands = pd.DataFrame(columns=['age'], data=['20s', '30s', '20s', '20s', '20s', '30s'])


@pytest.mark.parametrize('data, decision, error',
                         [([], 'A', ValueError),
                          (['bla'], 'A', TypeError),
                          (pd.DataFrame(columns=['A'], data=[1]), '', ValueError),
                          (pd.DataFrame(columns=['A'], data=[1]), 'B', ValueError)])
def test_compute_errors(data, decision, error):
    with pytest.raises(error):
        fair_age.compute(data, decision)


@pytest.mark.parametrize('decisions, value, obj',
                         [([1, 0, 1, 1, 1, 0], 0, False),
                          ([1, 1, 0, 1, 0, 0], 1, True),
                          ([1, 1, 1, 0, 0, 1], 0.5, False)])
def test_compute_values(decisions, value, obj):
    cands['d'] = decisions
    fair, check = fair_age.compute(cands, 'd')
    assert pytest.approx(fair, 0.001) == value
    assert check == obj


# Evaluate _dist_to_fairness
@pytest.mark.parametrize('attr, error',
                         [(['', 1, 1, 1], TypeError),
                          ([1, '', 1, 1], TypeError),
                          ([1, 1, '', 1], TypeError),
                          ([1, 1, 1, ''], TypeError),
                          ([-1, 1, 1, 1], ValueError),
                          ([1, -1, 1, 1], ValueError),
                          ([1, 1, -1, 1], ValueError),
                          ([1, 1, 1, -1], ValueError),
                          ([10, 1, 1, 1], ValueError),
                          ([1, 1, 10, 1], ValueError),
                          ([5, 10, 2, 10], ValueError)])
def test_dist_to_fairness_errors(attr, error):
    with pytest.raises(error):
        fair_age._dist_to_fairness(*attr)


@pytest.mark.parametrize('attr, value',
                         [([4, 5, 5, 5], 1),
                          ([1, 1, 1, 1], 0)])
def test_dist_to_fairness_values(attr, value):
    assert pytest.approx(fair_age._dist_to_fairness(*attr), 0.01) == value


# Evaluate distance
@pytest.mark.parametrize('data, decision, error',
                         [([], 'A', ValueError),
                          (['bla'], 'A', TypeError),
                          (pd.DataFrame(columns=['A'], data=[1]), '', ValueError),
                          (pd.DataFrame(columns=['A'], data=[1]), 'B', ValueError)])
def test_distance_errors(data, decision, error):
    with pytest.raises(error):
        fair_age.distance(data, decision)


@pytest.mark.parametrize('decisions, expected, value, ended',
                         [([1, 0, 1, 1, 0, 0], {'20s': -1, '30s': 1}, 1, True),
                          ([1, 1, 1, 0, 0, 1], {'20s': 0, '30s': 0}, 0.5, False),
                          ([1, 1, 0, 1, 0, 0], {'20s': 0, '30s': 0}, 1, True)])
def test_distance_values(decisions, expected, value, ended):
    cands['d'] = decisions
    swaps, converge, new_fair = fair_age.distance(cands, 'd')
    assert pytest.approx(new_fair, 0.001) == value
    assert swaps == expected
    assert converge == ended


# Evaluate difficulty
@pytest.mark.parametrize('init, dist, final, error',
                         [(-1, {}, 0.5, ValueError),
                          (1.5, {}, 0.5, ValueError),
                          (0.5, {}, -0.5, ValueError),
                          (0.5, {}, 1.5, ValueError),
                          (0.5, {'A': 1, 'B': -1}, 0.5, ValueError),
                          (0.8, {'A': 1, 'B': -1}, 0.5, ValueError),
                          (0.5, {'A': -1, 'B': -1}, 0.5, ValueError)])
def test_difficulty_errors(init, dist, final, error):
    with pytest.raises(error):
        fair_age.difficulty(init, dist, final)


@pytest.mark.parametrize('init, dist, final, value',
                         [(0, {'A': 1, 'B': -1}, 1, 1),
                          (0.9, {'A': 10, 'B': -10}, 0.91, 1000)])
def test_difficulty_values(init, dist, final, value):
    assert pytest.approx(fair_age.difficulty(init, dist, final), 0.01) == value


# Evaluate visualise
@pytest.mark.parametrize('data, decision, richness, error',
                         [([], 'A', 1, ValueError),
                          (['bla'], 'A', 1,  TypeError),
                          (pd.DataFrame(columns=['A'], data=[1]), '', 1, ValueError),
                          (pd.DataFrame(columns=['A'], data=[1]), 'B', 1, ValueError),
                          (pd.DataFrame(columns=['A'], data=[1]), 'A', 1.1, TypeError),
                          (pd.DataFrame(columns=['A'], data=[1]), 'A', -1, ValueError)])
def test_visualise_errors(data, decision, richness, error):
    with pytest.raises(error):
        fair_age.visualise(data, decision, richness)


# Evaluate get_impact
@pytest.mark.parametrize('data, decision, cat, error',
                         [(pd.DataFrame(), 'A', '20s', ValueError),
                          (['a', 'b'], 'A', '20s', TypeError),
                          (pd.DataFrame(columns=['A'], data=[1]), 'B', '20s', ValueError),
                          (pd.DataFrame(columns=['A'], data=[1]), 'A', '20s', ValueError),
                          (pd.DataFrame(columns=['A'], data=[1]), 'A', '', ValueError)])
def test_get_impact_errors(data, decision, cat, error):
    with pytest.raises(error):
        fair_age.get_impact(data, decision, cat)


@pytest.mark.parametrize('decisions, cat, value',
                         [([1, 0, 1, 1, 1, 0], '30s', 0.33),
                          ([1, 1, 0, 1, 0, 0], '30s', -0.25),
                          ([1, 1, 1, 1, 1, 1], '20s', 0)])
def test_get_impact_values(decisions, cat, value):
    cands['d'] = decisions
    assert pytest.approx(fair_age.get_impact(cands, 'd', cat), 0.01) == value
