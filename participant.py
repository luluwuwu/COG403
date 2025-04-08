from datetime import timedelta
import math
from ACS import *
from NACS import *
from pyClarion import (Event, Key, Agent, Priority, Input, Pool, Choice, 
    ChunkStore, BaseLevel, Family, NumDict, Atoms, Atom, Chunk, ks_crawl)
from MCS import * 
from dataprep import OA_DATA
from episodic_vals import YA_DATA
import pandas as pd
import random

"""Controls/ Single memory source agents"""
class SemanticParticipant(Agent):
    d: SemanticObjPointPairs
    input: Input
    blas: BaseLevel
    semantic_store: ChunkStore
    episodic_store: ChunkStore
    pool: Pool
    choice: Choice

    def __init__(self, name: str) -> None:
        sem_p = Family()
        sem_e = Family()
        d = SemanticObjPointPairs()
        super().__init__(name, sem_p=sem_p, sem_e=sem_e, d=d)
        self.d = d
        with self:
            self.input = Input("input", (d, d))
            self.store = ChunkStore("store", d, d, d)
            self.blas = BaseLevel("blas", sem_p, sem_e, self.store.chunks)
        self.store.bu.input = self.input.main
        self.blas.input = self.choice.main
        self.pool["store.bu"] = (
            self.store.bu.main,
            lambda d: d.shift(x=1).scale(x=0.5).logit())
        self.pool["blas"] = (
            self.blas.main,
            lambda d: d.bound_min(x=1e-8).log().with_default(c=0.0))
        self.choice.input = self.pool.main
        with self.pool.params[0].mutable():
            self.pool.params[0][~self.pool.p["blas"]] = 2e-1
        self.blas.ignore.add(~self.store.chunks.nil)

    def resolve(self, event: Event) -> None:
        if event.source == self.store.bu.update:
            self.blas.update()
        if event.source == self.blas.update:
            self.choice.trigger()

    def start_trial(self, dt: timedelta, priority: Priority = Priority.PROPAGATION) -> None:
        self.system.schedule(self.start_trial, dt=dt, priority=priority)

    def finish_trial(self, dt: timedelta, priority: Priority = Priority.PROPAGATION) -> None:
        self.system.schedule(self.finish_trial, dt=dt, priority=priority)


class EpisodicParticipant(Agent):
    d: EpisodicObjPointPairs
    input: Input
    blas: BaseLevel
    episodic_store: ChunkStore
    pool: Pool
    choice: Choice

    def __init__(self, name: str) -> None:
        d = EpisodicObjPointPairs()
        super().__init__(name, d=d)
        self.d = d

        with self:
            self.input = Input("input", (d, d))
            self.episodic_store = ChunkStore("episodic_store", d, d, d)
            self.blas = BaseLevel("blas", d, d, self.episodic_store.chunks)
            self.pool = Pool("pool", self.blas)
            self.choice = Choice("choice", self.pool)

        self.episodic_store.bu.input = self.input.main
        self.blas.input = self.choice.main

        self.pool["episodic_store.bu"] = (
            self.episodic_store.bu.main,
            lambda d: d.shift(x=1).scale(x=0.5).logit()
        )
        self.pool["blas"] = (
            self.blas.main,
            lambda d: d.bound_min(x=1e-8).log().with_default(c=0.0)
        )
        self.choice.input = self.pool.main

        with self.pool.params[0].mutable():
            self.pool.params[0][~self.pool.p["blas"]] = 2e-1

        self.blas.ignore.add(~self.episodic_store.chunks.nil)

    def resolve(self, event: Event) -> None:
        if event.source == self.episodic_store.bu.update:
            self.blas.update()
        if event.source == self.blas.update:
            self.choice.trigger()

    def start_trial(self, dt: timedelta, priority: Priority = Priority.PROPAGATION) -> None:
        self.system.schedule(self.start_trial, dt=dt, priority=priority)

    def finish_trial(self, dt: timedelta, priority: Priority = Priority.PROPAGATION) -> None:
        self.system.schedule(self.finish_trial, dt=dt, priority=priority)

"""Dual memory agents"""
class DualParticipant(Agent):
    # Subsystems
    sem_d: SemanticObjPointPairs
    epi_d: EpisodicObjPointPairs

    # Memory components
    semantic_store: ChunkStore
    episodic_store: ChunkStore
    blas_dsem: BaseLevel
    blas_depi: BaseLevel
    sem_input: Input
    epi_input: Input

    pool: Pool
    choice: Choice

    def __init__(self, name: str) -> None:
        d_p = Family()
        d_e = Family()
        dsem_d = SemanticObjPointPairs()
        depi_d = EpisodicObjPointPairs()
        super().__init__(name, dsem_d=dsem_d, depi_d=depi_d, d_p=d_p, d_e=d_e)
        self.dsem_d = dsem_d
        self.depi_d = depi_d
        self.memory_log = []

        with self:
            self.sem_input = Input("sem_input", (dsem_d, dsem_d))
            self.epi_input = Input("epi_input", (depi_d, depi_d))
            self.semantic_store = ChunkStore("semantic_store", d_p, dsem_d, dsem_d)
            self.episodic_store = ChunkStore("episodic_store", d_p, depi_d, depi_d)
            self.blas_dsem = BaseLevel("blas_sem", d_p, d_e, self.semantic_store.chunks)
            self.blas_depi = BaseLevel("blas_epi", d_p, d_e, self.episodic_store.chunks)
            self.pool = Pool("pool", d_p, self.episodic_store.chunks, func=NumDict.sum)
        # d_p.__annotations__.update({
        #     "semantic_store.bu": Atom,
        #     "episodic_store.bu": Atom,
        #     "blas_dsem": Atom, 
        #     "blas_depi": Atom
        #     })
        d_p._members_[Key("episodic_store.bu")] = Atom()
        d_p._members_[Key("semantic_store.bu")] = Atom()

        self.pool["episodic_store"] = (
                self.episodic_store.bu.main, 
                lambda depi_d: depi_d.shift(x=1).scale(x=0.5).logit()
            )
        self.pool["semantic_store"] = (
                self.semantic_store.bu.main, 
                lambda dsem_d: dsem_d.shift(x=1).scale(x=0.5).logit()
            )
        self.pool["blas_dsem"] = (
                self.blas_dsem.main, 
                lambda d: d.bound_min(x=1e-8).log().with_default(c=0.0)
            )
        self.pool["blas_depi"] = (
                self.blas_depi.main, 
                lambda d: d.bound_min(x=1e-8).log().with_default(c=0.0)
            )
        self.choice = Choice("choice", d_p, self.episodic_store.chunks)
            

        self.semantic_store.bu.input = self.pool.main
        self.episodic_store.bu.input = self.pool.main
        self.blas_dsem.input = self.choice.main
        self.blas_depi.input = self.choice.main
        
        self.mcs = MetaCognitiveController(inhibition_level=0.2, start_with="semantic")
        
        self.choice.input = self.pool.main
        with self.pool.params[0].mutable():
            self.pool.params[0][~self.pool.p["blas_sem"]] = 2e-1
            self.pool.params[0][~self.pool.p["blas_epi"]] = 2e-1
        self.blas_sem.ignore.add(~self.semantic_store.chunks.nil)
        self.blas_epi.ignore.add(~self.episodic_store.chunks.nil)

    def resolve(self, event: Event) -> None:
        if event.source == self.semantic_store.bu.update:
            self.blas_sem.update()
        if event.source == self.episodic_store.bu.update:
            self.blas_epi.update()
        
        if event.source in {self.blas_sem.update, self.blas_epi.update}:
            # Use a simulated delay (normalized value) and assume the task goal is episodic
            delay = random.randint(8,13)
            self.mcs.decide_switch(goal="episodic", delay=delay)
            # Re-route pool to the selected memory system
            self.pool.pop("semantic_store.bu", None)
            self.pool.pop("episodic_store.bu", None)
            if self.mcs.current_memory == "semantic":
                self.pool["semantic_store.bu"] = (
                    self.semantic_store.bu.main,
                    lambda d: d.shift(x=1).scale(x=0.5).logit()
                )
            else:
                self.pool["episodic_store.bu"] = (
                    self.episodic_store.bu.main,
                    lambda d: d.shift(x=1).scale(x=0.5).logit()
                )
            self.choice.trigger()
            self.memory_log.append(self.mcs.current_memory)
            print(f"[MCS] Trial using: {self.mcs.current_memory}")

    def start_trial(self, dt: timedelta, priority: Priority = Priority.PROPAGATION) -> None:
        self.system.schedule(self.start_trial, dt=dt, priority=priority)

    def finish_trial(self, dt: timedelta, priority: Priority = Priority.PROPAGATION) -> None:
        self.system.schedule(self.finish_trial, dt=dt, priority=priority)

    def get_active_memory_source(self):
        return self.mcs.current_memory

"""Event Processing"""

def retrieve_episodic_value(participant: DualParticipant| EpisodicParticipant, obj_atom:Atom, io:IO):
    """
    Retrieve the value and activation-based confidence of an object
    from episodic memory chunks.

    Parameters:
        obj_atom (Atom): The object whose memory is being retrieved.
        system (System): The pyClarion agent system (e.g., agent.system).
        io (Atoms): The IO keyspace family with `.input` and `.output`.

    Returns:
        tuple[int, float]: (value, confidence), where:
            - value is the numeric point value (0–100)
            - confidence is the chunk's activation level (0–1)
        Returns (None, 0.0) if no match is found.
    """

    # Search episodic chunks using the object as input
    response_key = participant.choice.poll()[~participant.store.chunks]
    response_chunk = ks_crawl(participant.system.root, response_key)

    if not response_chunk:
        return None, 0.0

    # Pick the most active match
    best_chunk = max(response_chunk, key=lambda chunk: chunk.main.c)
    confidence = best_chunk.main.c

    point_atom = best_chunk.key[io.output]
    value_str = str(point_atom)

    try:
        return float(value_str.split(".")[-1])
    except ValueError:
        return None
    
def retrieve_semantic_value(obj_atom, system, d: SemanticObjPointPairs, activation_map=None):
    """
    Retrieve the semantic value (and optionally activation) of an object.

    Returns:
        int or tuple[int, float]: Just the value, or (value, activation)
    """
    io = d.io
    point = d.point

    matches = ks_crawl(system.root, (io.input ** obj_atom))
    if not matches:
        return (None, 0.0) if activation_map else None

    best_chunk = max(matches, key=lambda chunk: chunk.main.c)
    point_atom = best_chunk.key[io.output]

    for name, atom in vars(point).items():
        if atom == point_atom:
            try:
                value = int(name.strip("_p"))
                if activation_map:
                    obj_name = str(obj_atom)
                    activation = activation_map.get(obj_name, 0.0)
                    return value, activation
                return value
            except ValueError:
                pass

    return (None, 0.0) if activation_map else None

def find_chunk_by_object_name(
    chunks: list[Chunk],
    obj_name: str,
    io_keyspace,
    object_keyspace
) -> Chunk | None:
    """
    A helper function to search for a chunk in a list by matching its io.input value to the given object name.

    Parameters:
        chunks: List of compiled Chunk objects.
        obj_name: Name of the object to find (e.g., "balloon").
        io_keyspace: Keyspace with an `input` member (e.g., d.io).
        object_keyspace: Keyspace where object atoms live (e.g., d.object).

    Returns:
        Chunk if found, otherwise None.
    """
    obj_atom = object_keyspace.__dict__.get(obj_name)
    if obj_atom is None:
        return None

    for chunk in chunks:
        if chunk.key.get(io_keyspace.input) == obj_atom:
            return chunk

    return None

def choose_card(
        obj1, obj2,
        participant: DualParticipant,
        epi_family: EpisodicObjPointPairs,
        sem_family: SemanticObjPointPairs,
        threshold: float = 0.7
    ) -> tuple[Atom, str, int]:
        """
        Select the better object using episodic memory when confident,
        and fall back to semantic memory when not.

        Parameters:
            obj1, obj2: Object atoms 
            system: The agent’s system 
            epi_family: Episodic memory keyspace family instance
            sem_family: Semantic memory keyspace family instance
            threshold: Confidence threshold to trust episodic memory

        Returns:
            (chosen_obj, strategy, value):
                - chosen_obj: the Atom selected
                - strategy: one of "episodic", "semantic", or "mixed"
                - value: the numeric value associated with the choice
        """

        val1_epi, conf1 = retrieve_episodic_value(participant, obj1, epi_family.io)
        val2_epi, conf2 = retrieve_episodic_value(participant, obj2, epi_family.io)

        if conf1 >= threshold and conf2 >= threshold:
            strategy = "episodic"
            if val1_epi >= val2_epi:
                return obj1, strategy, val1_epi
            else:
                return obj2, strategy, val2_epi

        elif conf1 >= threshold:
            val2_sem, _ = retrieve_semantic_value(participant, obj2, sem_family.io)
            strategy = "mixed"
            if val1_epi >= val2_sem:
                return obj1, strategy, val1_epi
            else:
                return obj2, strategy, val2_sem

        elif conf2 >= threshold:
            val1_sem, _ = retrieve_semantic_value(participant, obj1, sem_family.io)
            strategy = "mixed"
            if val2_epi >= val1_sem:
                return obj2, strategy, val2_epi
            else:
                return obj1, strategy, val1_sem

        else:
            val1_sem, _ = retrieve_semantic_value(participant, obj1, sem_family.io)
            val2_sem, _ = retrieve_semantic_value(participant, obj2, sem_family.io)
            strategy = "semantic"
            if val1_sem >= val2_sem:
                return obj1, strategy, val1_sem
            else:
                return obj2, strategy, val2_sem

participant = DualParticipant("participant")

OA_stimuli = []

for _, row in OA_DATA.iterrows():
    objA = row["ObjA"]
    objB = row["ObjB"]
    OA_stimuli.append((f"Obj_{objA}", f"Obj_{objB}"))

trial = 0 
presentations = {} 
results = {
    "trial": [],
    "stim": [],
    "time": [],
    "delta": [],
    "response": [],
    "correct": [],
    "strength": [],
    "rt": [],
    "strategy": []
}
participant.start_trial(timedelta())
while participant.system.queue:
    event = participant.system.advance()
    # print(event.describe())
    if event.source == participant.start_trial:
        obj1, obj2 = OA_stimuli[trial]

        # Extract object names
        # obj1 = obj_chunk1.key[participant.sem_d.io.input]._name_
        # obj2 = obj_chunk2.key[participant.sem_d.io.input]._name_

        # Run decision logic
        chosen, strategy, value = choose_card(
            getattr(participant.sem_d.object, obj1),
            getattr(participant.sem_d.object, obj2),
            participant,
            participant.epi_d,
            participant.sem_d,
            0.7
        )

        # Track which objects were seen on which trial
        for obj in [obj1, obj2]:
            if obj not in presentations:
                presentations[obj] = trial

        # Log trial results (to be completed in finish_trial)
    participant.finish_trial(timedelta(seconds=3))
    if event.source == participant.finish_trial:
        obj_chunk1, obj_chunk2 = OA_stimuli[trial]
        target = obj_chunk1.key[participant.sem_d.io.input]  # or however you want to define "correct"

        # Log results
        results["trial"].append(trial)
        results["stim"].append(f"{obj_chunk1.key} vs {obj_chunk2.key}")
        results["delta"].append(trial - presentations.get(target._name_, trial))
        results["strategy"].append(strategy)
        results["response"].append(chosen._name_)
        results["correct"].append(chosen == target)
        results["strength"].append(value)
        results["rt"].append(math.exp(-value))

        # Move to next trial
        trial += 1
        if trial < len(OA_stimuli):
            participant.start_trial(timedelta(seconds=2))

        

"""Initialize event logging for simulation"""

import logging
import sys

logger = logging.getLogger("pyClarion.system")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
