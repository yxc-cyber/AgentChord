from copy import deepcopy
from time import sleep

from .gbc_base import GBCBase


class GBCList(GBCBase, list):
    """
    GBCList is a subclass of GBCBase and list, providing a list representation of the GBC object.
    It can be used to represent the GBC object as a list while maintaining the functionality of GBCBase.
    """
    def __init__(self, iterable=None, connections=None, weights=None, subject=None):
        """
        Create a new GBCList instance.
        """
        super().__init__(iterable or [])
        self.bind_connections(
            connections = connections if connections is not None else [],
            weights = weights if weights is not None else []
        )
        self.bind_subject(subject or "Unnamed Subject")

    def __add__(self, other):
        """
        Override the addition operator to concatenate lists.
        """
        result = super().__add__(other)
        return GBCList(
            result, 
            connections=[self, other], 
            weights=[1.0, 1.0], 
            subject="Addition Operation"
        )
    
    def __iadd__(self, other):
        """
        Override the in-place addition operator to concatenate lists.
        """
        result = super().__iadd__(other)
        self.bind_connections(
            result,
            weights=[1.0, 1.0],
            subject="Addition Operation"
        )
        return self
    
    def __getitem__(self, key):
        """
        Override the item access operator to return a GBCList for the specific item.
        """
        result = super().__getitem__(key)
        from .gbc_factory import GBC
        return GBC(
            result, 
            connections=[self], 
            weights=[1.0], 
            subject=f"Item Access Operation"
        )
    
    def __deepcopy__(self, memo=None):
        """
        Override the deepcopy method to ensure proper copying of GBCList.
        """
        new_iterable = deepcopy(list(self), memo)
        new_connections = deepcopy(self.get_connections(), memo)
        new_weights = deepcopy(self.get_weights(), memo)
        new_subject = self.get_subject(), memo
        new_gbc_list = GBCList(
            new_iterable, 
            connections=new_connections, 
            weights=new_weights, 
            subject=new_subject
        )
        memo[id(self)] = new_gbc_list
        return new_gbc_list
    
    def append(self, object):
        """
        Override the append method to add an object to the list and bind connections.
        """
        old_self = deepcopy(self)
        self.bind_connections(
            connections=[old_self, object],
            weights=[1.0, 1.0]
        )
        super().append(object)
        self.bind_subject("Append Operation")

    def extend(self, iterable):
        """
        Override the extend method to add an iterable to the list and bind connections.
        """
        old_self = deepcopy(self)
        self.bind_connections(
            connections=[old_self, iterable],
            weights=[1.0, 1.0]
        )
        super().extend(iterable)
        self.bind_subject("Extend Operation")