from __future__ import annotations

from typing import Any


def project_qwen_expert_residuals(
    hidden_states: Any,
    top_k_index: Any,
    top_k_weights: Any,
    source_gate: Any,
    source_up: Any,
    source_down: Any,
    candidate_gate: Any,
    candidate_up: Any,
    candidate_down: Any,
    downstream_score_gradients: Any,
):
    """Project exact routed full-expert residuals through downstream score gradients.

    Inputs are torch tensors. `downstream_score_gradients` is [rank, tokens,
    hidden] and can be obtained by backpropagating score-function probes from
    next-token logits to the selected MoE block output. Returned z[e, r]
    contains the router-weighted candidate-minus-source expert residual.
    """
    import torch
    import torch.nn.functional as functional

    if hidden_states.ndim not in (2, 3):
        raise ValueError("hidden_states must be [tokens, hidden] or [batch, tokens, hidden]")
    x = hidden_states.reshape(-1, hidden_states.shape[-1])
    indices = top_k_index.reshape(-1, top_k_index.shape[-1])
    weights_by_route = top_k_weights.reshape(-1, top_k_weights.shape[-1])
    gradients_flat = downstream_score_gradients.reshape(
        downstream_score_gradients.shape[0], -1, downstream_score_gradients.shape[-1]
    )
    if indices.shape[0] != x.shape[0] or gradients_flat.shape[1] != x.shape[0]:
        raise ValueError("token dimensions disagree across hidden states, routes, and gradients")
    expert_count = source_gate.shape[0]
    rank = downstream_score_gradients.shape[0]
    projected = torch.zeros((expert_count, rank), dtype=torch.float64, device=x.device)
    direct_energy = torch.zeros(expert_count, dtype=torch.float64, device=x.device)
    route_mass = torch.zeros(expert_count, dtype=torch.float64, device=x.device)
    for expert in range(expert_count):
        positions = torch.nonzero(indices == expert, as_tuple=False)
        if positions.numel() == 0:
            continue
        token_index = positions[:, 0]
        slot_index = positions[:, 1]
        expert_x = x[token_index]
        source_hidden = functional.silu(functional.linear(expert_x, source_gate[expert])) * functional.linear(expert_x, source_up[expert])
        source_output = functional.linear(source_hidden, source_down[expert])
        candidate_hidden = functional.silu(functional.linear(expert_x, candidate_gate[expert])) * functional.linear(expert_x, candidate_up[expert])
        candidate_output = functional.linear(candidate_hidden, candidate_down[expert])
        weights = weights_by_route[token_index, slot_index].to(candidate_output.dtype)
        delta = (candidate_output - source_output) * weights[:, None]
        gradients = gradients_flat[:, token_index, :].to(delta.dtype)
        projected[expert] = torch.einsum("rth,th->r", gradients, delta).double()
        direct_energy[expert] = 0.5 * torch.mean(delta.double().square())
        route_mass[expert] = weights.double().sum()
    return {
        "projected_residuals": projected,
        "direct_output_energy": direct_energy,
        "route_mass": route_mass,
    }


def fisher_score_gradients(logits, block_output, rank: int, seed: int = 0):
    """Score-function Fisher/Jacobian sketch d log p(y|x) / d block_output."""
    import torch

    if not block_output.requires_grad:
        raise ValueError("block_output must remain attached to the autograd graph")
    probabilities = torch.softmax(logits.float(), dim=-1)
    generator = torch.Generator(device=logits.device)
    generator.manual_seed(seed)
    samples = torch.multinomial(probabilities.reshape(-1, probabilities.shape[-1]), rank, replacement=True, generator=generator)
    gradients = []
    flat_log_probs = torch.log_softmax(logits.float(), dim=-1).reshape(-1, logits.shape[-1])
    for probe in range(rank):
        selected = flat_log_probs.gather(1, samples[:, probe : probe + 1]).sum() / (flat_log_probs.shape[0] ** 0.5)
        gradient = torch.autograd.grad(selected, block_output, retain_graph=probe + 1 < rank)[0]
        gradients.append(gradient.detach())
    return torch.stack(gradients)
