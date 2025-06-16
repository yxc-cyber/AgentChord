class GBC:
    """
    Factory class for creating GBC objects.
    """

    def __new__(cls, value, connections=None, weights=None, subject=None):
        if isinstance(value, str):
            from .gbc_str import GBCStr
            return GBCStr(value, connections, weights, subject)
        elif isinstance(value, list):
            from .gbc_list import GBCList
            return GBCList(value, connections, weights, subject)
        else:
            return value  # Fallback to the original value if not a recognized type