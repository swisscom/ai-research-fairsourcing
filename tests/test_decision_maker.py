"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved.
"""

import pandas as pd
import pytest

from ..decision_maker import DecisionMaker


# Evaluate the constructor
@pytest.mark.parametrize('attr, error',
                         [(['a', 'b', '', {'a1': 'a'}], ValueError),
                          (['a', 'b', 'c', {'a1': 'a'}, 0.1], TypeError),
                          (['a', 'b', 'c', {'a1': 'a'}, -1], ValueError),
                          (['a', 'b', 'c', {'a1': 'a'}, 0], ValueError)])
# indirect=['person'])
def test_constructor_errors(attr, error):
    with pytest.raises(error):
        DecisionMaker(*attr)


# Evaluate _compute_landscape
dm = DecisionMaker('John', 'Sales', 'transfer',
                   {'age': '30s', 'gender': 'M', 'language': 'DE'}, 1)
cands = pd.DataFrame(columns=['age', 'gender', 'language', 'transfer'],
                     data=[['20s', 'F', 'EN', 0],
                           ['30s', 'F', 'FR', 1],
                           ['40s', 'M', 'IT', 0],
                           ['50s', 'M', 'DE', 1],
                           ['30s', 'M', 'DE', 0]])


@pytest.mark.parametrize('param, error',
                         [('', ValueError),
                          ('hello', TypeError),
                          (pd.DataFrame(columns=['A'], data=[1]), ValueError),
                          (pd.DataFrame(columns=['cand_age'], data=[1]), ValueError),
                          (pd.DataFrame(columns=['cand_age', 'cand_gender'], data=[[1, 2]]), ValueError),
                          (pd.DataFrame(columns=['cand_age', 'cand_gender', 'cand_language'], data=[[1, 2, 3]]),
                           ValueError)])
def test_compute_landscape_errors(param, error):
    with pytest.raises(error):
        dm._compute_landscape(param)


def test_compute_landscape_values():
    landscape = dm._compute_landscape(cands)
    assert landscape['total'].sum() == len(cands)
    assert landscape['pos_count'].all() <= landscape['total'].all()
    assert 0 <= landscape['probability'].all() <= 1
    assert landscape[(landscape['age'] == '50s') & (landscape['gender'] == 'M') &
                     (landscape['language'] == 'DE')]['probability'].values[0] == 1
    assert landscape[(landscape['age'] == '20s') & (landscape['gender'] == 'F') &
                     (landscape['language'] == 'EN')]['pos_count'].values[0] == 0
    assert landscape[(landscape['age'] == '30s') & (landscape['gender'] == 'F') &
                     (landscape['language'] == 'DE')]['total'].values[0] == 0


# Evaluate _distance_candidate
@pytest.mark.parametrize('attr, error',
                         [({}, ValueError),
                          ({'a1': 'a', 'b': 'b2'}, ValueError),
                          ({'b': 'b2'}, ValueError)])
def test_distance_candidate_errors(attr, error):
    with pytest.raises(ValueError):
        dm._distance_candidate(attr)


def test_distance_candidate_values():
    assert dm._distance_candidate({'age': '20s', 'gender': 'O', 'language': 'FR'}) == 3
    assert dm._distance_candidate({'age': '30s', 'gender': 'O', 'language': 'FR'}) == 2
    assert dm._distance_candidate({'age': '30s', 'gender': 'M', 'language': 'FR'}) == 1
    assert dm._distance_candidate({'age': '30s', 'gender': 'M', 'language': 'DE'}) == 0


# Evaluate _prule
@pytest.mark.parametrize('probs, error',
                         [({}, ValueError),
                          ({0: 0, 1: 0}, ValueError),
                          ({0: 0.5, 1: -0.5}, ValueError),
                          ({0: 0.5, 1: 1.5}, ValueError)])
def test_prule_errors(probs, error):
    with pytest.raises(error):
        dm._prule(probs)


@pytest.mark.parametrize('probs, value',
                         [({0: 0.5, 1: 0.5}, 1),
                          ({0: 0, 1: 1}, 0),
                          ({0: 0.3, 1: 0.4, 2: 0.3}, 0.75)])
def test_prule_values(probs, value):
    assert dm._prule(probs) == value


# Evaluate get_fairness
@pytest.mark.parametrize('data, error',
                         [(pd.DataFrame(), ValueError),
                          (['a', 'b'], TypeError)])
def test_get_fairness_errors(data, error):
    with pytest.raises(error):
        dm.get_fairness(data)


def test_get_fairness_values():
    prule, dist_probs = dm.get_fairness(cands)
    assert prule == 0
    assert dist_probs[0] == dist_probs[3] == 0
    assert dist_probs[1] == 1
    assert dist_probs[2] == 0.5


# Evaluate visualise_fairness
@pytest.mark.parametrize('data, dist, error',
                         [(pd.DataFrame(), {0: 0, 1: 1}, ValueError),
                          (['a', 'b'], {0: 0, 1: 1}, TypeError),
                          (pd.DataFrame(columns=['A'], data=[1]), {}, ValueError)])
def test_visualise_fairness_errors(data, dist, error):
    with pytest.raises(error):
        dm.visualise_fairness(data, dist)


# Evaluate visualise_landscape
@pytest.mark.parametrize('data, error',
                         [(pd.DataFrame(), ValueError),
                          (['a', 'b'], TypeError)])
def test_visualise_landscape_errors(data, error):
    with pytest.raises(error):
        dm.visualise_landscape(data)
