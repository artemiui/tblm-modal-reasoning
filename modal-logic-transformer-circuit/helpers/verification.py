from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple
try:
    import torch
    import torch as t
except ImportError:
    torch = None
    t = None


def get_heads_and_posns_to_keep(
    ctfl_dataset: Any,
    model: Any,
    circuit: Dict[str, List[Tuple[int, int]]],
    seq_pos_to_keep: Dict[str, Any],
) -> Dict[int, Any]:
    """
    Builds layer-wise boolean masks telling complement patching where to preserve clean activations.
    """
    heads_and_posns_to_keep = {}
    batch, seq = len(ctfl_dataset), len(ctfl_dataset[0])
    n_heads = model.cfg.n_heads

    for layer in range(model.cfg.n_layers):
        mask = t.zeros(size=(batch, seq, n_heads), dtype=t.bool)
        for head_type, head_list in circuit.items():
            seq_pos = seq_pos_to_keep.get(head_type, slice(None))
            for (l_idx, h_idx) in head_list:
                if l_idx == layer:
                    mask[:, seq_pos, h_idx] = True
        heads_and_posns_to_keep[layer] = mask

    return heads_and_posns_to_keep


def hook_fn_mask_z(
    z: Any,
    hook: Any,
    heads_and_posns_to_keep: Dict[int, Any],
    ctfl_actns: Any,
) -> Any:
    """
    Hook function replacing non-circuit head outputs with counterfactual cache activations.
    """
    layer_idx = hook.layer() if hasattr(hook, "layer") else int(hook.name.split(".")[1])
    mask = heads_and_posns_to_keep[layer_idx].unsqueeze(-1).to(z.device)
    ctfl_val = ctfl_actns[layer_idx].to(z.device)
    return t.where(mask, z, ctfl_val)


def add_ctfl_ablation_hook(
    model: Any,
    ctfl_dataset: Any,
    circuit: Dict[str, List[Tuple[int, int]]],
    seq_pos_to_keep: Dict[str, Any],
) -> Any:
    """
    Adds complement-ablation hooks to the model according to the circuit specification.
    """
    model.reset_hooks(including_permanent=True)

    _, ctfl_cache = model.run_with_cache(
        ctfl_dataset,
        return_type=None,
        names_filter=lambda name: name.endswith("z"),
    )

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head
    batch, seq_len = len(ctfl_dataset), len(ctfl_dataset[0])
    device = getattr(model.cfg, "device", "cpu")

    ctfl_actns = t.zeros(size=(n_layers, batch, seq_len, n_heads, d_head), device=device)
    for l in range(n_layers):
        act_name = f"blocks.{l}.attn.hook_z" if hasattr(model, "blocks") else f"blocks.{l}.hook_z"
        ctfl_actns[l] = ctfl_cache[act_name]

    heads_and_posns_to_keep = get_heads_and_posns_to_keep(ctfl_dataset, model, circuit, seq_pos_to_keep)

    for l in range(n_layers):
        act_name = f"blocks.{l}.attn.hook_z" if hasattr(model, "blocks") else f"blocks.{l}.hook_z"
        hook_fn = lambda z, hook, l=l: hook_fn_mask_z(z, hook, heads_and_posns_to_keep, ctfl_actns)
        model.add_hook(act_name, hook_fn, is_permanent=True)

    return model


def circuit_specification(circuit_option: str = "full") -> Tuple[Dict[str, List[Tuple[int, int]]], Dict[str, Any]]:
    """
    Standard modal logic circuit families for Gemma-2-9B (Hong et al. + Modal Extensions):
      - QRLH: Queried-Rule Locators
      - MOH: Modal-Operator Heads (new)
      - WAH: World-Accessibility Heads (new)
      - FPH: Fact Processors
      - QRMH: Queried-Rule Movers
      - DH: Decision Heads
    """
    base_circuit = {
        "QRLH": [(19, 11), (21, 0), (21, 7), (22, 5), (23, 12)],
        "MOH": [(20, 3), (22, 9), (23, 4)],
        "WAH": [(21, 14), (23, 1), (25, 10)],
        "FPH": [(24, 5), (25, 7), (26, 0), (26, 12)],
        "QRMH": [(20, 7), (23, 6), (24, 15), (27, 15)],
        "DH": [(28, 12), (30, 9)],
    }

    seq_pos_to_keep = {
        "QRLH": slice(None),
        "MOH": slice(None),
        "WAH": slice(None),
        "FPH": slice(None),
        "QRMH": slice(None),
        "DH": slice(None),
    }

    if circuit_option == "full":
        return base_circuit, seq_pos_to_keep
    elif circuit_option == "no_moh":
        c = dict(base_circuit)
        c["MOH"] = []
        return c, seq_pos_to_keep
    elif circuit_option == "no_wah":
        c = dict(base_circuit)
        c["WAH"] = []
        return c, seq_pos_to_keep
    elif circuit_option == "no_qrlh":
        c = dict(base_circuit)
        c["QRLH"] = []
        return c, seq_pos_to_keep
    elif circuit_option == "no_qrmh":
        c = dict(base_circuit)
        c["QRMH"] = []
        return c, seq_pos_to_keep
    elif circuit_option == "no_fph":
        c = dict(base_circuit)
        c["FPH"] = []
        return c, seq_pos_to_keep
    elif circuit_option == "no_dh":
        c = dict(base_circuit)
        c["DH"] = []
        return c, seq_pos_to_keep
    elif circuit_option == "random":
        return {"RANDOM": [(5, 0), (7, 2), (10, 4)]}, {"RANDOM": slice(None)}
    else:
        raise ValueError(f"Unknown circuit option {circuit_option}")
