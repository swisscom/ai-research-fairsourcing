"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved.
"""

import pytest

from ..candidate import Candidate


# Evaluate the constructor
@pytest.mark.parametrize('attr, error',
                         [(['', 'b', {'a': 'a1'}, 'd'], ValueError),
                          (['a', '', {'a': 'a1'}, 'd'], ValueError),
                          (['a', 'b', ['bla'], 'd'], TypeError),
                          (['a', 'b', {}, 'd'], ValueError),
                          (['a', 'b', {'a': 'a1'}, 4], TypeError),
                          (['a', 'b', {'a': 'a1'}, ''], ValueError)])
def test_constructor_errors(attr, error):
    with pytest.raises(error):
        Candidate(*attr)
