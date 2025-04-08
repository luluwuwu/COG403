from datetime import timedelta
import random
from pyClarion import (Event, Agent, Priority, Input, Pool, Choice, 
    ChunkStore, BaseLevel, Family, NumDict, Atoms, Atom, Chunk, ks_crawl)

from episodic_vals import EPISODIC_VALUES_MAP  # Derived from your YA_GroupData

"""Keyspace Definition"""

def safe_key(k):
    return f"Obj_{str(k).replace('.', '_')}"  

class Object(Atoms):
    """The object stimuli.
    
    Attributes are dynamically created based on the keys in EPISODIC_VALUES_MAP.
    """
    pass

# Dynamically add an Atom for each object key in EPISODIC_VALUES_MAP.
Object.__annotations__ = {safe_key(key): Atom for key in EPISODIC_VALUES_MAP.keys()}


class epiPoint(Atoms):
    """Episodic points for each object."""
    pass

epiPoint.__annotations__ = {safe_key(val): Atom for val in EPISODIC_VALUES_MAP.keys()}


class IO(Atoms):
    input: Atom
    output: Atom
    target: Atom


class EpisodicObjPointPairs(Family):
    """
    A family for object-point pairs representing the explicit episodic
    associations (i.e., the arbitrary point values assigned to each object).
    """
    object: Object
    point: epiPoint
    io: IO


# Helper function to map a point value to the corresponding epiPoint atom.
def get_epi_atom(value: float, point_instance: epiPoint) -> Atom:
    """Return the appropriate epiPoint atom for a given value.
    
    If the value is not exactly one of the expected point values, round
    it to the nearest 25.
    """
    # Round to nearest multiple of 25.
    rounded = round(value / 25) * 25
    if rounded == 0:
        return point_instance._0p
    elif rounded == 25:
        return point_instance._25p
    elif rounded == 50:
        return point_instance._50p
    elif rounded == 75:
        return point_instance._75p
    elif rounded == 100:
        return point_instance._100p
    else:
        # Fallback in case of an unexpected value.
        return point_instance._50p


"""Chunks for bottom-up activation model"""

def init_epi_knowledge(d: EpisodicObjPointPairs, episodic_map: dict) -> list[Chunk]:
    """
    Initialize explicit episodic knowledge chunks for each object using
    the derived episodic values from the dataset.
    
    Parameters:
      d: An instance of EpisodicObjPointPairs containing keyspace definitions.
      episodic_map: Dictionary mapping object IDs (as strings) to their 
                    assigned point values (e.g., 0, 25, 50, 75, 100).
                    
    Returns:
      A list of Chunk objects that encode the object-point associations.
    """
    io_instance = d.io
    object_instance = d.object
    point_instance = d.point

    episodic_chunk_defs = []
    # Loop over each object key and its episodic value.
    for key, value in episodic_map.items():
        # Create a chunk name based on the object key.
        chunk_name = f"episodic_{key}"
        # Retrieve the corresponding Atom for the object.
        obj_atom = getattr(object_instance, key)
        # Get the appropriate episodic point atom based on the derived value.
        pt_atom = get_epi_atom(value, point_instance)
        # Construct the chunk using the caret operator (^) to bind the chunk name,
        # and the '+' operator to combine the input and output associations.
        chunk = (
            chunk_name ^
            (+ io_instance.input ** obj_atom) +
            (+ io_instance.output ** pt_atom)
        )
        episodic_chunk_defs.append(chunk)

    # Compile the chunks into the ChunkStore.
    ChunkStore.compile(*episodic_chunk_defs)
    return episodic_chunk_defs

from pyClarion import ks_crawl

