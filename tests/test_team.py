"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved.
"""

import pandas as pd
import pytest

from ..candidate import Candidate
from ..decision_maker import DecisionMaker
from ..metrics.diversity import Diversity
from ..metrics.fairness import Fairness
from ..team import Team

# Evaluate the constructor
team_id = 'Sales'
mng = DecisionMaker('John', team_id, 'hire', {'age': '30s'}, 1)
recr = DecisionMaker('Lisa', team_id, 'transfer', {'age': '20s'}, 1)
emps = pd.DataFrame(columns=['age'], data=['20s', '30s', '20s', '20s'])
div_age = Diversity('age', 1., 1., ['20s', '30s'], 2)
cands = pd.DataFrame(columns=['age'], data=['20s', '30s', '20s', '20s', '20s', '30s'])
cands['transfer'] = [0, 1, 0, 1, 0, 1]
fair_age = Fairness('age', 1., 1., 5, 2)


@pytest.mark.parametrize('attr, error',
                         [(['', mng, [recr], emps, [div_age], cands, [fair_age]], ValueError),
                          ([team_id, 'bla', [recr], emps, [div_age], cands, [fair_age]], TypeError),
                          ([team_id, mng, 'bla', emps, [div_age], cands, [fair_age]], TypeError),
                          ([team_id, mng, ['bla'], emps, [div_age], cands, [fair_age]], TypeError),
                          ([team_id, mng, [recr], ['bla'], [div_age], cands, [fair_age]], TypeError),
                          ([team_id, mng, [recr], pd.DataFrame(), [div_age], cands, [fair_age]], ValueError),
                          ([team_id, mng, [recr], emps, 'bla', cands, [fair_age]], TypeError),
                          ([team_id, mng, [recr], emps, ['bla'], cands, [fair_age]], TypeError),
                          ([team_id, mng, [recr], emps, [div_age, div_age], cands, [fair_age]], ValueError),
                          ([team_id, mng, [recr], emps, [div_age], ['bla'], [fair_age]], TypeError),
                          ([team_id, mng, [recr], emps, [div_age], pd.DataFrame(), [fair_age]], ValueError),
                          ([team_id, mng, [recr], emps, [div_age], cands, 'bla'], TypeError),
                          ([team_id, mng, [recr], emps, [div_age], cands, ['bla']], TypeError),
                          ([team_id, mng, [recr], emps, [div_age], cands, [fair_age, fair_age]], ValueError)])
def test_constructor_errors(attr, error):
    with pytest.raises(error):
        Team(*attr)


# Evaluate compute_social_metric
team = Team(team_id, mng, [recr], emps, [div_age], cands, [fair_age])

emps2 = pd.DataFrame(columns=['age', 'gender'], data=[['20s', 'M'], ['30s', 'F'], ['20s', 'M'], ['20s', 'F']])
div_age2 = Diversity('age', 1., 0.5, ['20s', '30s'], 2)
div_gender = Diversity('gender', 1., 0.5, ['M', 'F'], 2)
cands2 = pd.DataFrame(columns=['age', 'gender'], data=[['20s', 'M'], ['30s', 'F'], ['20s', 'F'], ['20s', 'M'],
                                                       ['20s', 'F'], ['30s', 'F']])
cands2['transfer'] = [0, 1, 0, 1, 0, 1]
fair_age2 = Fairness('age', 1., 0.5, 5, 2)
fair_gender = Fairness('gender', 1., 0.5, 5, 2)
team2 = Team(team_id, mng, [recr], emps2, [div_age2, div_gender], cands2, [fair_age2, fair_gender])


@pytest.mark.parametrize('metric, decision, error',
                         [(2, None, TypeError),
                          ('', None, ValueError),
                          ('bla', None, ValueError),
                          ('fairness', None, ValueError),
                          ('fairness', 'bla', ValueError)])
def test_compute_social_metric_errors(metric, decision, error):
    with pytest.raises(error):
        team.compute_social_metric(metric, decision)


@pytest.mark.parametrize('metric, decisions, value',
                         [('diversity', None, 0.75),
                          ('fairness', [1, 1, 0, 1, 0, 0], 1)])
def test_compute_social_metric_values(metric, decisions, value):
    if decisions is not None:
        team._fairness_state['d'] = decisions
    assert pytest.approx(team.compute_social_metric(metric, 'd'), 0.001) == value


@pytest.mark.parametrize('metric, decisions, value',
                         [('diversity', None, 0.875),
                          ('fairness', [1, 1, 0, 1, 0, 0], 0.625)])
def test_compute_social_metric_values2(metric, decisions, value):
    if decisions is not None:
        team2._fairness_state['d'] = decisions
    assert pytest.approx(team2.compute_social_metric(metric, 'd'), 0.001) == value


# Evaluate visualise_social_metric
@pytest.mark.parametrize('metric, decision, color_id, error',
                         [(2, None, 0, TypeError),
                          ('', None, 0, ValueError),
                          ('bla', None, 0, ValueError),
                          ('Fairness', '', 0, ValueError),
                          ('Fairness', 'bla', 0, ValueError),
                          ('Diversity', None, 0.1, TypeError),
                          ('Diversity', None, -1, ValueError)])
def test_visualise_social_metric_error(metric, decision, color_id, error):
    with pytest.raises(error):
        team.visualise_social_metric(metric, decision, color_id)


# Evaluate compute_cand_impacts
c = Candidate('Lea', team_id, {'age': '30s'}, 'transfer')


@pytest.mark.parametrize('candidate, error',
                         [(2, TypeError),
                          (Candidate('Lea', 'bla', {'age': '30s'}, 'transfer'), ValueError),
                          (Candidate('Lea', team_id, {'bla': 'a'}, 'transfer'), ValueError),
                          (Candidate('Lea', team_id, {'age': '30s'}, 'bla'), ValueError)])
def test_compute_cand_impacts_errors(candidate, error):
    with pytest.raises(error):
        team.compute_cand_impacts(candidate)


@pytest.mark.parametrize('candidate, div_imp, div_w, fair_imp, fair_w',
                         [(c, [0.25], [1], [0], [1])])
def test_compute_cand_impacts_values(candidate, div_imp, div_w, fair_imp, fair_w):
    res_1, res_2, res_3, res_4 = team.compute_cand_impacts(candidate)
    assert res_1 == div_imp
    assert res_2 == div_w
    assert res_3 == fair_imp
    assert res_4 == fair_w


# Evaluate compute_cand_agg_impacts
@pytest.mark.parametrize('candidate, error',
                         [(2, TypeError),
                          (Candidate('Lea', 'bla', {'age': '30s'}, 'transfer'), ValueError),
                          (Candidate('Lea', team_id, {'bla': 'a'}, 'transfer'), ValueError),
                          (Candidate('Lea', team_id, {'age': '30s'}, 'bla'), ValueError)])
def test_compute_cand_agg_impacts_errors(candidate, error):
    with pytest.raises(error):
        team.compute_cand_agg_impacts(candidate)


c2 = Candidate('Lea', team_id, {'age': '30s', 'gender': 'F'}, 'transfer')


@pytest.mark.parametrize('team, cand, agg_div, agg_fair',
                         [(team, c, 0.25, 0),
                          (team2, c2, 0.125, -0.085)])
def test_compute_cand_agg_impacts_value(team, cand, agg_div, agg_fair):
    comp_div, comp_fair = team.compute_cand_agg_impacts(cand)
    assert pytest.approx(comp_div, 0.01) == agg_div
    assert pytest.approx(comp_fair, 0.01) == agg_fair


# Evaluate _visualise_cand_impact
@pytest.mark.parametrize('candidate, metric, impacts, weights, error',
                         [(2, 'diversity', [2], [1], TypeError),
                          (Candidate('Lea', team_id, {'age': '30s', 'gender': 'F'}, 'transfer'), 'diversity', [2], [1],
                           ValueError),
                          (Candidate('Lea', 'bla', {'age': '30s'}, 'transfer'), 'diversity', [2], [1],
                           ValueError),
                          (c, 2, [2], [1], TypeError),
                          (c, '', [2], [1], ValueError),
                          (c, 'bla', [2], [1], ValueError),
                          (c, 'diversity', 1, [1], TypeError),
                          (c, 'diversity', [], [1], ValueError),
                          (c, 'diversity', ['bla'], [1], TypeError),
                          (c, 'diversity', [0.02, 0.2], [1], ValueError),
                          (c, 'diversity', [0.02], '', TypeError),
                          (c, 'diversity', [0.02], [], ValueError),
                          (c, 'diversity', [0.02], ['bla'], TypeError),
                          (c, 'diversity', [0.02], [2.], ValueError),
                          (c, 'diversity', [0.02], [0.5, 0.5], ValueError)])
def test_visualise_cand_impact_errors(candidate, metric, impacts, weights, error):
    with pytest.raises(error):
        team._visualise_cand_impact(candidate, metric, impacts, weights)


# Evaluate visualise_cand_impacts
@pytest.mark.parametrize('candidate, error',
                         [(2, TypeError),
                          (Candidate('Lea', team_id, {'age': '30s', 'gender': 'F'}, 'transfer'), ValueError),
                          (Candidate('Lea', 'bla', {'age': '30s'}, 'transfer'), ValueError)])
def test_visualise_cand_impacts_errors(candidate, error):
    with pytest.raises(error):
        team.visualise_cand_impacts(candidate)


# Compute optimal profile
@pytest.mark.parametrize('attr_space, decision, div_imp, n, error',
                         [(['bla'], 'd', 0.5, 1, TypeError),
                          ({}, 'd', 0.5, 1, ValueError),
                          ({'bla': 'a'}, 'd', 0.5, 1, ValueError),
                          ({'age': '20s'}, 'bla', 0.5, 1, ValueError),
                          ({'age': '20s'}, 'd', -1, 1, ValueError),
                          ({'age': '20s'}, 'd', 1.5, 1, ValueError),
                          ({'age': '20s'}, 'd', 0.5, 1.1, TypeError),
                          ({'age': '20s'}, 'd', 0.5, -1, ValueError),
                          ({'age': ['20s', '30s']}, 'd', 0.5, 3, ValueError)])
def test_compute_optimal_profile_errors(attr_space, decision, div_imp, n, error):
    with pytest.raises(error):
        team.compute_optimal_profile(attr_space, decision, div_imp, n)


@pytest.mark.parametrize('attr_space, decisions, div_imp, n, expected',
                         [({'age': ['20s', '30s']}, [1, 0, 1, 1, 1, 0], 0.5, 1,
                           pd.DataFrame(columns=['attr', 'div_impact', 'fair_impact', 'tot_impact'],
                                        data=[[{'age': '30s'}, 0.25, 0.33, 0.29]])),
                          ({'age': ['20s', '30s']}, [1, 0, 1, 1, 1, 0], 0.75, 1,
                           pd.DataFrame(columns=['attr', 'div_impact', 'fair_impact', 'tot_impact'],
                                        data=[[{'age': '30s'}, 0.25, 0.33, 0.27]]))
                          ])
def test_compute_optimal_profile_values(attr_space, decisions, div_imp, n, expected):
    team._fairness_state['d'] = decisions
    computed = team.compute_optimal_profile(attr_space, 'd', div_imp, n)
    assert len(computed) == n
    assert computed.values.all() == expected.values.all()
