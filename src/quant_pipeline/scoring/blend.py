from __future__ import annotations

from collections.abc import Callable, Sequence


class BlendedModule:
    """Factory for a torch module that blends BF16 and candidate outputs."""

    @staticmethod
    def wrap(source, candidate, alpha):
        import torch

        class _Blend(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.source = source
                self.candidate = candidate
                self.alpha = alpha

            def forward(self, *args, **kwargs):
                source_output = self.source(*args, **kwargs)
                candidate_output = self.candidate(*args, **kwargs)
                if not isinstance(source_output, torch.Tensor) or not isinstance(candidate_output, torch.Tensor):
                    raise TypeError("blended modules must return a tensor")
                blend = self.alpha.to(device=source_output.device, dtype=source_output.dtype)
                return source_output + blend * (candidate_output - source_output)

        return _Blend()


def module_path_attribution(
    model,
    module_pairs: Sequence[tuple[object, str, object]],
    loss_for_batch: Callable[[object], object],
    batches: Sequence[object],
    path_nodes: int,
):
    """Run simultaneous layer/module Aumann-Shapley attribution.

    module_pairs contains (parent_module, attribute_name, candidate_module).
    The source module is restored even on failure. Parameters should be frozen;
    only one scalar alpha per unit receives gradients.
    """
    import numpy as np
    import torch

    nodes, quadrature = np.polynomial.legendre.leggauss(path_nodes)
    nodes = (nodes + 1.0) / 2.0
    quadrature = quadrature / 2.0
    alphas = [torch.nn.Parameter(torch.tensor(0.0, device=next(model.parameters()).device)) for _ in module_pairs]
    sources = []
    for (parent, name, candidate), alpha in zip(module_pairs, alphas, strict=True):
        source = getattr(parent, name)
        sources.append(source)
        setattr(parent, name, BlendedModule.wrap(source, candidate, alpha))
    attribution = torch.zeros(len(alphas), dtype=torch.float64)
    try:
        for node, weight in zip(nodes, quadrature, strict=True):
            for alpha in alphas:
                alpha.data.fill_(float(node))
                alpha.grad = None
            for batch in batches:
                loss = loss_for_batch(batch)
                gradients = torch.autograd.grad(loss, alphas)
                attribution += float(weight) * torch.tensor([float(g.detach().cpu()) for g in gradients], dtype=torch.float64) / len(batches)
    finally:
        for (parent, name, _), source in zip(module_pairs, sources, strict=True):
            setattr(parent, name, source)
    return attribution.numpy()
