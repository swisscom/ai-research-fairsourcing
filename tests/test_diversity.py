"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved.
"""

import pandas as pd
import pytest

from ..metrics.diversity import Diversity


# Evaluate the constructor
@pytest.mark.parametrize('attr, error',
                         [(['', 0.9, 0.9, ['a', 'b'], 2], ValueError),
                          (['age', -0.9, 0.9, ['a', 'b'], 2], ValueError),
                          (['age', 1.1, 0.9, ['a', 'b'], 2], ValueError),
                          (['age', 0.9, -0.9, ['a', 'b'], 2], ValueError),
                          (['age', 0.9, 1.1, ['a', 'b'], 2], ValueError),
                          (['age', 0.9, 0.9, ['a', 'b'], 2.2], TypeError),
                          (['age', 0.9, 0.9, ['a', 'b'], -2], ValueError),
                          (['age', 0.9, 0.9, [], 2], ValueError)])
def test_constructor_errors(attr, error):
    with pytest.raises(error):
        Diversity(*attr)


# Evaluate _compute_hunter_gaston
div_age = Diversity('age', 1., 1., ['20s', '30s'], 2)


@pytest.mark.parametrize('counts, error',
                         [({}, ValueError),
                          ({'20s': 1}, ValueError),
                          ({'20s': 0, '30s': 0}, ValueError),
                          ({'20s': 0, '30s': -1}, ValueError),
                          ({'20s': 0, '30s': 3.1}, ValueError)])
def test_compute_hunter_gaston_errors(counts, error):
    with pytest.raises(error):
        div_age._compute_hunter_gaston(counts)


@pytest.mark.parametrize('counts, value',
                         [({'20s': 5, '30s': 5}, 0.56),
                          ({'20s': 5, '30s': 0}, 0.),
                          ({'20s': 1, '30s': 1}, 1.)])
def test_compute_hunter_gaston_values(counts, value):
    assert pytest.approx(div_age._compute_hunter_gaston(counts), 0.01) == value


# Evaluate _get_counts
emps = pd.DataFrame(columns=['age'], data=['20s', '30s', '20s', '20s', '20s', '30s'])


@pytest.mark.parametrize('data, error',
                         [(pd.DataFrame(), ValueError),
                          (['20s', '30s'], TypeError),
                          (pd.DataFrame(columns=['bla'], data=['a', 'b']), ValueError),
                          (pd.DataFrame(columns=['age'], data=['20s', 'b']), ValueError)])
def test_get_counts_errors(data, error):
    with pytest.raises(error):
        div_age._get_counts(data)


@pytest.mark.parametrize('data, expected',
                         [(emps, {'20s': 4, '30s': 2}),
                          (pd.DataFrame(columns=['age'], data=['20s', '20s', '20s', '20s']), {'20s': 4, '30s': 0})])
def test_get_counts_value(data, expected):
    assert div_age._get_counts(data) == expected


# Evaluate _compute_uniform_counts
@pytest.mark.parametrize('counts, error',
                         [(['bla'], TypeError),
                          ({}, ValueError),
                          ({'a': 0, 'b': 0}, ValueError)])
def test_compute_uniform_counts_errors(counts, error):
    with pytest.raises(error):
        div_age._compute_uniform_counts(counts)


@pytest.mark.parametrize('counts, value',
                         [({'A': 4, 'B': 2}, {'A': 3, 'B': 3}),
                          ({'A': 1, 'B': 0}, {'A': 1, 'B': 0}),
                          ({'A': 10, 'B': 0}, {'A': 5, 'B': 5}),
                          ({'A': 10, 'B': 1}, {'A': 6, 'B': 5})])
def test_compute_uniform_counts_values(counts, value):
    assert div_age._compute_uniform_counts(counts) == value


# Evaluate compute
@pytest.mark.parametrize('data, error',
                         [([], ValueError),
                          (['bla'], TypeError),
                          (pd.DataFrame(columns=['A'], data=[1]), ValueError)])
def test_compute_errors(data, error):
    with pytest.raises(error):
        div_age.compute(data)


@pytest.mark.parametrize('data, value, obj',
                         [(['20s', '30s', '20s', '20s', '20s', '30s'], 0.88, False),
                          (['20s', '20s', '20s'], 0, False),
                          (['20s', '30s', '30s', '20s', '20s', '30s'], 1, True)])
def test_compute_values(data, value, obj):
    emps = pd.DataFrame(columns=['age'], data=data)
    div, check = div_age.compute(emps)
    assert pytest.approx(div, 0.001) == value
    assert check == obj


# Evaluate distance
@pytest.mark.parametrize('data, error',
                         [([], ValueError),
                          (['bla'], TypeError)])
def test_distance_errors(data, error):
    with pytest.raises(error):
        div_age.distance(data)


@pytest.mark.parametrize('data, expected, value',
                         [(['20s', '30s', '20s', '20s', '20s', '30s'], {'20s': 0, '30s': 1}, 1),
                          (['20s', '20s', '20s'], {'20s': 0, '30s': 2}, 1),
                          (['20s', '30s', '30s', '20s', '20s', '30s'], {'20s': 0, '30s': 0}, 1)])
def test_distance_values(data, expected, value):
    emps = pd.DataFrame(columns=['age'], data=data)
    increments, new_div = div_age.distance(emps)
    assert pytest.approx(new_div, 0.001) == value
    assert increments == expected


# Evaluate difficulty
@pytest.mark.parametrize('init, dist, final, error',
                         [(-1, {}, 0.5, ValueError),
                          (1.5, {}, 0.5, ValueError),
                          (0.5, {}, -0.5, ValueError),
                          (0.5, {}, 1.5, ValueError),
                          (0.5, {'A': 1}, 0.5, ValueError),
                          (0.8, {'A': 1}, 0.5, ValueError),
                          (0.5, {'A': -1}, 0.5, ValueError)])
def test_difficulty_errors(init, dist, final, error):
    with pytest.raises(error):
        div_age.difficulty(init, dist, final)


@pytest.mark.parametrize('init, dist, final, value',
                         [(0, {'A': 1}, 1, 1),
                          (0.9, {'A': 10}, 0.91, 1000)])
def test_difficulty_values(init, dist, final, value):
    assert pytest.approx(div_age.difficulty(init, dist, final), 0.01) == value


# Evaluate visualise
@pytest.mark.parametrize('data, error',
                         [(pd.DataFrame(), ValueError),
                          (['a', 'b'], TypeError),
                          (pd.DataFrame(columns=['A'], data=[1]), ValueError)])
def test_visualise_errors(data, error):
    with pytest.raises(error):
        div_age.visualise(data)


# Evaluate get_impact
@pytest.mark.parametrize('data, cat, error',
                         [(pd.DataFrame(), '20s', ValueError),
                          (['a', 'b'], '20s', TypeError),
                          (pd.DataFrame(columns=['A'], data=[1]), '20s', ValueError),
                          (pd.DataFrame(columns=['age'], data=[1]), '', ValueError)])
def test_get_impact_errors(data, cat, error):
    with pytest.raises(error):
        div_age.get_impact(data, cat)


@pytest.mark.parametrize('data, cat, value',
                         [(['20s', '30s', '20s', '20s', '20s', '30s'], '30s', 0.12),
                          (['20s', '20s', '20s'], '20s', 0),
                          (['20s', '30s', '30s', '20s', '20s', '30s'], '20s', 0)])
def test_get_impact_values(data, cat, value):
    emps = pd.DataFrame(columns=['age'], data=data)
    assert pytest.approx(div_age.get_impact(emps, cat), 0.01) == value
