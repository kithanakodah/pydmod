# __init__.py

# --- Primary, Functional Imports ---
# Import the real, working Chunk class from our new robust parser.
from .cnk_loader import Chunk

# For backward compatibility, your main script expects a class named ForgelightChunk.
# We create an alias here so you don't have to change your main script.
ForgelightChunk = Chunk


# --- Compatibility Placeholder Classes ---
# These are dummy classes that exist only to prevent ImportError in scripts
# that still try to import them. They have no real functionality.

class Header:
    """Placeholder for compatibility."""
    def __init__(self, **kwargs): pass

class Tiles:
    """Placeholder for compatibility."""
    def __init__(self, **kwargs): pass

class Tile:
    """Placeholder for compatibility."""
    def __init__(self, **kwargs): pass

class Eco:
    """Placeholder for compatibility."""
    def __init__(self, **kwargs): pass

class Flora:
    """Placeholder for compatibility."""
    def __init__(self, **kwargs): pass

class Layer:
    """Placeholder for compatibility."""
    def __init__(self, **kwargs): pass


# --- __all__ Definition ---
# This list controls what gets imported when a script uses "from data_classes import *"
# It also serves as a public API for your module.

__all__ = [
    'Chunk',            # The new, correct name for our parser class.
    'ForgelightChunk',  # The alias for backward compatibility.
    'Header',           # The dummy Header class.
    'Tiles',            # The dummy Tiles class.
    'Tile',             # The dummy Tile class.
    'Eco',              # The dummy Eco class.
    'Flora',            # The dummy Flora class.
    'Layer'             # The dummy Layer class.
]