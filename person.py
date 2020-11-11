"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved.

Person, used for modeling the information about a physical person.
The Person class contains the basic information to describe a person.
Its function is to ensure the minimal information is provided.
"""
from abc import ABC


class Person(ABC):
    """Person class.
    The Person is used to interact with a team.
    """

    def __init__(self, pers_id: str, team_id: str, attributes: dict):
        """ Inits Person with its pers_id, team_id and attributes
        Args:
            pers_id: A unique string representing this person
            team_id: A str with the id of the team this person interacts with
            attributes: A dict with the demographic categories of this person for each criteria
        """
        if len(attributes) == 0:
            raise ValueError('Argument: attributes are empty')
        if len(pers_id) == 0:
            raise ValueError('Argument: pers_id has length 0')
        if len(team_id) == 0:
            raise ValueError('Argument: team_id has length 0')

        self._pers_id = pers_id
        self._team_id = team_id
        self._attributes = attributes
        super().__init__()
