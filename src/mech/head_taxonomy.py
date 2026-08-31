from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence


@dataclass(frozen=True)
class HeadRef:
    """Reference to a specific attention head by layer and head index."""
    layer: int
    head: int


def _normalize(items: Sequence[dict]) -> List[HeadRef]:
    """Parse a list of dicts into HeadRef instances."""
    out: List[HeadRef] = []
    for item in items:
        out.append(HeadRef(layer=int(item["layer"]), head=int(item["head"])))
    return out


# Extended default taxonomy for modal logic analysis
DEFAULT_TAXONOMY: Dict[str, List[HeadRef]] = {
    # From Ref-A (propositional)
    "splitting": [],
    "transmission": [],
    "fact_retrieval": [],
    # New for modal logic
    "world_locating": [],
    "accessibility_tracing": [],
    "valuation_retrieval": [],
    "modal_operator": [],
    "decision": [],
}


def load_taxonomy(path: Path | None) -> Dict[str, List[HeadRef]]:
    """Load a head taxonomy from a JSON file, or return the default."""
    if path is None:
        return DEFAULT_TAXONOMY
    obj = json.loads(path.read_text(encoding="utf-8"))
    return {k: _normalize(v) for k, v in obj.items()}


def get_head_set(name: str, taxonomy: Dict[str, List[HeadRef]]) -> List[HeadRef]:
    """Retrieve a named head set from the taxonomy."""
    if name not in taxonomy:
        raise KeyError(f"Head set {name!r} not found in taxonomy")
    return taxonomy[name]


def save_taxonomy(taxonomy: Dict[str, List[HeadRef]], path: Path) -> None:
    """Save a head taxonomy to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        k: [{"layer": h.layer, "head": h.head} for h in v]
        for k, v in taxonomy.items()
    }
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
