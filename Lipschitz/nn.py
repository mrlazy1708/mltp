import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchMean(nn.Module):
    def __init__(self, num_features, momentum=0.1):
        super().__init__()
        self.num_features = num_features

        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros(num_features))

    def forward(self, x):
        y = x.view(*x.size()[:2], -1)

        if self.training:
            mean = y.mean(dim=-1).mean(dim=0)
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum)
                self.running_mean.add_(mean, alpha=self.momentum)
        else:
            mean = self.running_mean

        return (y - mean.unsqueeze(-1)).view_as(x)

class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weights = nn.Parameter(torch.Tensor(out_features, in_features))
        with torch.no_grad():
            nn.init.normal_(self.weights)

    def forward(self, inputs, p=torch.inf):
        x = inputs.unsqueeze(1)         # Batch x 1   x In
        w = self.weights.unsqueeze(0)   # 1     x Out x In

        from .core import dist
        return dist(x, w, p)

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, init=-10.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.weights = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        with torch.no_grad():
            nn.init.normal_(self.weights)
            if init is not None and in_channels <= out_channels:
                for i in range(out_channels):
                    self.weights[i, i % (in_channels), kernel_size // 2, kernel_size // 2] = init

    def forward(self, inputs, p=torch.inf):
        B, _, H, W = inputs.shape

        x = F.unfold(inputs, self.kernel_size, self.dilation, self.padding, self.stride)
        x = x.transpose(1, 2).unsqueeze(1)              # Batch x 1   x H*W x In*Ker^2
        w = self.weights.flatten(-3)[None, :, None, :]  # 1     x Out x 1   x In*Ker^2

        from .core import dist
        y = dist(x, w, p=p)

        return y.view(B, -1,
            (H + 2 * self.padding - self.kernel_size) // self.stride + 1,
            (W + 2 * self.padding - self.kernel_size) // self.stride + 1)
