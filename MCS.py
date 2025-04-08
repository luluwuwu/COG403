import random
from ACS import *
from NACS import *
from pyClarion import (Event, Agent, Priority, Input, Pool, Choice, 
    ChunkStore, BaseLevel, Family, NumDict, Atoms, Atom, Chunk, ks_crawl)

class MetaCognitiveController:
    def __init__(self, age_group: str = "YA", inhibition_level: float = 0.8, start_with="semantic"):
        """
        :param age_group: "YA" for younger adults or "OA" for older adults.
        :param inhibition_level: Higher means more resistance to switching.
        :param start_with: "semantic" or "episodic".
        """
        self.age_group = age_group
        self.inhibition_level = inhibition_level
        self.current_memory = start_with
        self.trial_count = 0

    def retrieve_episodic_value(obj_atom, system, io) -> tuple[int | None, float]:
        """
        Retrieve episodic value and activation-based confidence for a given object.

        Returns:
            (value, confidence):
                - value: 0–100 in steps of 25 (or None if not found)
                - confidence: chunk.main.c (0.0–1.0)
        """
        
        matches = ks_crawl(system.root, io.input ** obj_atom)

        if not matches:
            return None, 0.0

        best_chunk = max(matches, key=lambda chunk: chunk.main.c)
        confidence = best_chunk.main.c
        point_atom = best_chunk.key[io.output]
        value_str = str(point_atom)

        try:
            # For example, if the atom's name is 'semValue._75p', extract '75'
            return int(point_atom.strip("_p")), confidence
        except ValueError:
            try:
                # Alternatively, if the atom's name is something like 'semValue.75'
                return int(float(value_str.split(".")[-1])), confidence
            except ValueError:
                return None, 0.0

    def retrieve_semantic_value(obj_atom, system, io) -> tuple[int | None, float]:
        """
        Retrieve semantic value and activation-based confidence for a given object.

        Returns:
            (value, confidence):
                - value: 0–100 in steps of 25 (or None if not found)
                - confidence: chunk.main.c (0.0–1.0)
        """

        matches = ks_crawl(system.root, io.input ** obj_atom)

        if not matches:
            return None, 0.0

        best_chunk = max(matches, key=lambda chunk: chunk.main.c)
        confidence = best_chunk.main.c
        point_atom = best_chunk.key[io.output]
        value_str = str(point_atom)

        try:
            # For example, if the atom's name is 'semValue._75p', extract '75'
            return int(point_atom.strip("_p")), confidence
        except ValueError:
            try:
                # Alternatively, if the atom's name is something like 'semValue.75'
                return int(float(value_str.split(".")[-1])), confidence
            except ValueError:
                return None, 0.0




    def reset(self):
        """
        Reset the controller's state between runs.
        """
        self.current_memory = "semantic"
        self.trial_count = 0

# # Example usage:
# controller_YA = MetaCognitiveController(age_group="YA", inhibition_level=0.8, start_with="semantic")
# controller_OA = MetaCognitiveController(age_group="OA", inhibition_level=0.2, start_with="semantic")

# print("Younger Adults:")
# for trial in range(10):
#     # For YA, we can still use a randomly generated normalized delay for demonstration.
#     print(f"Trial {trial+1}: {controller_YA.decide_switch(goal='episodic', delay=random.random()*10)}")

# print("\nOlder Adults:")
# for trial in range(10):
#     # For OA, simulate delay using a value from the dataset (here randomly between 0 and 20).
#     print(f"Trial {trial+1}: {controller_OA.decide_switch(goal='episodic', delay=random.uniform(0, 20))}")
