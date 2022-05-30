import torch

### Lp distance of the last dimension
def dist(x, w, p=torch.inf):
    y = torch.abs(x - w)
    # with torch.no_grad():
    #     from torch.linalg import vector_norm as norm
    #     n = norm(y, dim=2, ord=torch.inf).unsqueeze(-1)
    # y = norm(y / n, dim=2, ord=p)

    if p == torch.inf:
        y = torch.max(y, dim=-1)
        return y.values  # value

    else:
        y = torch.pow(y, p)
        y = torch.sum(y, dim=-1)
        return y.pow(1 / p)
