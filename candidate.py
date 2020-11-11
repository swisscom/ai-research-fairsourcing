"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved.

Candidate, used for modeling the information of candidate.
The Candidate class contains the implementation of a candidate to a job.
Its function is to evaluate its impact on a specific subpopulation, i.e. team.
"""
from .person import Person


class Candidate(Person):
    """Candidate class.
    The Candidate is used to interact with a team and see the impact.
    """

    def __init__(self, pers_id: str, team_id: str, attributes: dict, decision: str):
        """ Inits Candidates with its pers_id, team, attributes, decision
        Args:
            pers_id: A unique string representing this candidate
            team_id: A str with the id of the team this candidate is applying to
            attributes: A dict with the demographic categories of this candidate for each criteria
            decision: A string indicating at what stage is the application and which binary column is relevant
                        as decision
        """
        if len(pers_id) == 0:
            raise ValueError('Argument: pers_id is empty')
        if len(team_id) == 0:
            raise ValueError('Argument: team_id is empty')
        if not isinstance(attributes, dict):
            raise TypeError('Argument: attributes must be a dictionary')
        if len(attributes) == 0:
            raise ValueError('Argument: attributes are empty')
        if not isinstance(decision, str):
            raise TypeError('Argument: decision_name must be a string')
        if len(decision) == 0:
            raise ValueError('Argument: decision_name is empty')
        super().__init__(pers_id, team_id, attributes)
        self._decision = decision
