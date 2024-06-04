import torch

def check_mask(x: torch.Tensor, node_mask: torch.Tensor, dim_mask: torch.Tensor):
    eps = 1e-4
    assert (x * (1 - node_mask)).abs().max().item() < eps, \
        'Variables not masked properly'
    assert (x * (1 - dim_mask)).abs().max().item() < eps, \
        'Variables not masked properly'
    
    return