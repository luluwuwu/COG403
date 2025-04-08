# This file represents the top level or explicit representation of the NACS that contains the real-world oject-value associations
from datetime import timedelta
from pyClarion import (Event, Agent, Priority, Input, Pool, Choice, 
    ChunkStore, BaseLevel, Family, NumDict, Atoms, Atom, Chunk, ks_crawl)

from dataprep import *
# More dimensions? Which dimensions?
# generate arbitrary and unique object-point value pairs?
# explicit/top level of NACS
# how to implement episodic memory


"""Keyspace Definition"""
def safe_key(k):
    return f"Obj_{str(k).replace('.', '_')}"  

class Object(Atoms):
    """The object stimuli."""
    pass

Object.__annotations__ = {safe_key(key): Atom for key in SEM_ACTIVATION_MAP.keys()}


class semValue(Atoms):
    """Semantic Point for each object."""
    pass

semValue.__annotations__ = {safe_key(val): Atom for val in SEM_ACTIVATION_MAP.values()}


class IO(Atoms):
    input: Atom
    output: Atom
    target: Atom


class SemanticObjPointPairs(Family):
    """A family for semantic values (real-world associations) of the object-point pairs. """
    object: Object
    point: semValue
    io: IO


"""Chunks for bottom-up activation model"""
def init_sem_knowledge(d: SemanticObjPointPairs, semantic_map: dict) -> list[tuple[Chunk, Chunk]]:
    io = d.io
    object = d.object
    point = d.point

    semantic_chunk_defs = []

    for key, value in semantic_map.items():
        chunk_name = f"Obj_{str(key)}"
        chunk = (
            chunk_name ^
            + io.input ** object.__dict__[key]
            + io.output ** point.__dict__[value]
            )
        semantic_chunk_defs.append(chunk)

    ChunkStore.compile(*semantic_chunk_defs)
    return semantic_chunk_defs