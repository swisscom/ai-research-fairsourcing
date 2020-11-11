"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved.
"""

import pytest

from ..person import Person


# Evaluate the constructor
@pytest.mark.parametrize('attr, error',
                         [(['a', 'b', {}], ValueError),
                          (['', 'b', {'a': 'a1'}], ValueError),
                          (['a', '', {'a': 'a1'}], ValueError)])
def test_constructor_empty(attr, error):
    with pytest.raises(error):
        Person(*attr)
