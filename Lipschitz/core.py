import torch

### Lp norm of v's last dimension
def norm(v, p):
    if p == torch.inf:
        v = torch.max(v, dim=-1)
        return v.values

    else:
        v = torch.pow(v, p)
        v = torch.sum(v, dim=-1)
        return  v.pow(1/ p)


### Lp-Neuron output on the last two dimensions
def Lp(x, w, p):
    with torch.no_grad():
        w_lp = norm(torch.abs(w.flatten(-2)), p)
    return norm(torch.abs(x * w), p) / w_lp.unsqueeze(-1)
