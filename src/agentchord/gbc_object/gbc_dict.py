from copy import deepcopy

from .gbc_base import GBCBase


class GBCDict(GBCBase, dict):
    """
    GBCDict is a subclass of GBCBase and dict, providing a dictionary representation of the GBC object.
    It can be used to represent the GBC object as a dictionary while maintaining the functionality of GBCBase.
    """
    def __init__(self, map=None, connections=None, weights=None, subject=None):
        """
        Create a new GBCDict instance.
        """
        super().__init__(map or {})
        self.bind_connections(connections or [], weights or [])
        self.bind_subject(subject or "Unnamed Subject")

    def __deepcopy__(self, memo=None):
        """
        Override the deepcopy method to ensure proper copying of GBCDict.
        """
        new_map = {k: deepcopy(v) for k, v in self.items()}
        new_connections = deepcopy(self.get_connections(), memo)
        new_weights = deepcopy(self.get_weights(), memo)
        new_subject = self.get_subject()
        new_gbc_dict = GBCDict(
            new_map,
            connections=new_connections,
            weights=new_weights,
            subject=new_subject
        )
        memo[id(self)] = new_gbc_dict
        return new_gbc_dict