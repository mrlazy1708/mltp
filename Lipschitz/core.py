import torch

### Lp norm of v's last dimension
def norm(v, p):
    if p == torch.inf:
        v = torch.max(v, dim=-1)
        return v.values

    else:
        v = torch.pow(v, p)
        v = torch.sum(v, dim=-1)
        return v.pow(1 / p)


### Lp distance of the last dimension
def dist(x, w, p):
    return norm(torch.abs(x * w), p)
