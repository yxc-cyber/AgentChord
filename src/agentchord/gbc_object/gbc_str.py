from .gbc_base import GBCBase


class GBCStr(GBCBase, str):
    """
    GBCStr is a subclass of GBCBase and str, providing a string representation of the GBC object.
    It can be used to represent the GBC object as a string while maintaining the functionality of GBCBase.
    """
    def __new__(cls, value, connections=None, weights=None, subject=None):
        """
        Create a new GBCStr instance.
        """
        instance = super().__new__(cls, value)
        instance.bind_connections(connections or [], weights or [])
        instance.bind_subject(subject or "Unnamed Subject")
        return instance
    
    def __add__(self, other):
        """
        Override the addition operator to concatenate strings.
        """
        result = super().__add__(other)
        return GBCStr(
            result, 
            connections=[self, other], 
            weights=[1.0, 1.0], 
            subject="Addition Operation"
        )
    
    def __iadd__(self, other):
        """
        Override the in-place addition operator to concatenate strings.
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
        Override the item access operator to return a GBCStr for the specific character.
        """
        result = super().__getitem__(key)
        if isinstance(key, slice):
            # If a slice is used, return a new GBCStr with the sliced content
            return GBCStr(
                result, 
                connections=[self], 
                weights=[1.0], 
                subject=f"Slice Operation"
            )
        else:
            return GBCStr(
                result, 
                connections=[self], 
                weights=[1.0], 
                subject=f"Item Access Operation"
            )