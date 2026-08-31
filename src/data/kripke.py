from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import random

@dataclass(frozen=True)
class KripkeFrame:
    worlds: Tuple[str, ...]
    relation: Dict[str, Tuple[str, ...]]

    def is_reflexive(self) -> bool:
        return all(w in self.relation.get(w, ()) for w in self.worlds)

    def is_symmetric(self) -> bool:
        for w in self.worlds:
            for v in self.relation.get(w, ()):
                if w not in self.relation.get(v, ()):
                    return False
        return True

    def is_transitive(self) -> bool:
        for w in self.worlds:
            for v in self.relation.get(w, ()):
                for u in self.relation.get(v, ()):
                    if u not in self.relation.get(w, ()):
                        return False
        return True

    def is_serial(self) -> bool:
        return all(len(self.relation.get(w, ())) > 0 for w in self.worlds)

    def is_euclidean(self) -> bool:
        for w in self.worlds:
            for v in self.relation.get(w, ()):
                for u in self.relation.get(w, ()):
                    if u not in self.relation.get(v, ()):
                        return False
        return True

@dataclass(frozen=True)
class KripkeModel:
    frame: KripkeFrame
    valuation: Dict[str, Dict[str, bool]]
    eval_world: str

    def accessible(self, world: str) -> Tuple[str, ...]:
        return self.frame.relation.get(world, ())


def _generate_worlds(n_worlds: int) -> Tuple[str, ...]:
    return tuple(f'w{i}' for i in range(n_worlds))

def generate_arbitrary_frame(rng: random.Random, n_worlds: int) -> KripkeFrame:
    worlds = _generate_worlds(n_worlds)
    relation = {}
    for w in worlds:
        successors = [v for v in worlds if rng.choice([True, False])]
        relation[w] = tuple(successors)
    return KripkeFrame(worlds=worlds, relation=relation)

def generate_reflexive_frame(rng: random.Random, n_worlds: int) -> KripkeFrame:
    worlds = _generate_worlds(n_worlds)
    relation = {}
    for w in worlds:
        successors = [v for v in worlds if rng.choice([True, False])]
        if w not in successors:
            successors.append(w)
        relation[w] = tuple(successors)
    return KripkeFrame(worlds=worlds, relation=relation)

def generate_serial_frame(rng: random.Random, n_worlds: int) -> KripkeFrame:
    worlds = _generate_worlds(n_worlds)
    relation = {}
    for w in worlds:
        successors = [v for v in worlds if rng.choice([True, False])]
        if not successors:
            successors.append(rng.choice(worlds))
        relation[w] = tuple(successors)
    return KripkeFrame(worlds=worlds, relation=relation)

def generate_transitive_frame(rng: random.Random, n_worlds: int) -> KripkeFrame:
    worlds = _generate_worlds(n_worlds)
    # Generate arbitrary and then take transitive closure
    relation_sets = {w: set(v for v in worlds if rng.choice([True, False])) for w in worlds}
    
    changed = True
    while changed:
        changed = False
        for w in worlds:
            current_successors = list(relation_sets[w])
            for v in current_successors:
                for u in relation_sets[v]:
                    if u not in relation_sets[w]:
                        relation_sets[w].add(u)
                        changed = True

    relation = {w: tuple(relation_sets[w]) for w in worlds}
    return KripkeFrame(worlds=worlds, relation=relation)

def generate_symmetric_frame(rng: random.Random, n_worlds: int) -> KripkeFrame:
    worlds = _generate_worlds(n_worlds)
    relation_sets = {w: set() for w in worlds}
    for i in range(n_worlds):
        for j in range(i, n_worlds):
            if rng.choice([True, False]):
                relation_sets[worlds[i]].add(worlds[j])
                relation_sets[worlds[j]].add(worlds[i])
    
    relation = {w: tuple(relation_sets[w]) for w in worlds}
    return KripkeFrame(worlds=worlds, relation=relation)

def generate_euclidean_frame(rng: random.Random, n_worlds: int) -> KripkeFrame:
    worlds = _generate_worlds(n_worlds)
    relation_sets = {w: set(v for v in worlds if rng.choice([True, False])) for w in worlds}
    
    changed = True
    while changed:
        changed = False
        for w in worlds:
            current_successors = list(relation_sets[w])
            for v in current_successors:
                for u in current_successors:
                    if u not in relation_sets[v]:
                        relation_sets[v].add(u)
                        changed = True

    relation = {w: tuple(relation_sets[w]) for w in worlds}
    return KripkeFrame(worlds=worlds, relation=relation)

def generate_equivalence_frame(rng: random.Random, n_worlds: int) -> KripkeFrame:
    worlds = _generate_worlds(n_worlds)
    # Partition into random equivalence classes
    classes = []
    unassigned = list(worlds)
    rng.shuffle(unassigned)
    
    while unassigned:
        size = rng.randint(1, len(unassigned))
        cls = unassigned[:size]
        classes.append(cls)
        unassigned = unassigned[size:]
        
    relation = {}
    for cls in classes:
        for w in cls:
            relation[w] = tuple(cls)
            
    return KripkeFrame(worlds=worlds, relation=relation)

def generate_model(frame: KripkeFrame, variables: List[str], rng: random.Random, eval_world: Optional[str] = None) -> KripkeModel:
    valuation = {}
    for w in frame.worlds:
        valuation[w] = {var: rng.choice([True, False]) for var in variables}
        
    return KripkeModel(
        frame=frame,
        valuation=valuation,
        eval_world=eval_world if eval_world is not None else frame.worlds[0]
    )

def generate_frame_for_axiom(axiom: str, rng: random.Random, n_worlds: int = 3) -> KripkeFrame:
    if axiom == 'axiom_T':
        return generate_reflexive_frame(rng, n_worlds)
    elif axiom == 'axiom_D':
        return generate_serial_frame(rng, n_worlds)
    elif axiom == 'axiom_4':
        return generate_transitive_frame(rng, n_worlds)
    elif axiom == 'axiom_5':
        return generate_euclidean_frame(rng, n_worlds)
    elif axiom == 'axiom_B':
        return generate_symmetric_frame(rng, n_worlds)
    else:
        # Defaults to arbitrary for axiom_K, modal_modus_ponens, etc.
        return generate_arbitrary_frame(rng, n_worlds)

def format_frame(frame: KripkeFrame) -> str:
    worlds_str = ", ".join(frame.worlds)
    edges = []
    for w, successors in frame.relation.items():
        for s in successors:
            edges.append(f"{w} -> {s}")
    edges_str = ", ".join(edges) if edges else "None"
    return f"Worlds: {worlds_str}. Accessibility: {edges_str}."

def format_valuation(model: KripkeModel) -> str:
    parts = []
    for w, vals in model.valuation.items():
        val_strs = [f"{k} is {v}" for k, v in vals.items()]
        val_str = ", ".join(val_strs)
        parts.append(f"At {w}: {val_str}.")
    return " ".join(parts)

def format_model(model: KripkeModel) -> str:
    return f"{format_frame(model.frame)} {format_valuation(model)}"
