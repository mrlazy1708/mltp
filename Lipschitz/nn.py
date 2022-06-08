import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Module):
    def __init__(self, in_features, out_features, p=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.p = p

        self.weights = nn.Parameter(torch.Tensor(out_features, in_features))
        with torch.no_grad():
            nn.init.normal_(self.weights)

    def forward(self, inputs, p=None):
        x = inputs.unsqueeze(1)         # Batch x 1   x In
        w = self.weights.unsqueeze(0)   # 1     x Out x In

        from .core import Lp
        return Lp(x, w, self.p if p is None else p)

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, p=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.p = p

        self.weights = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        with torch.no_grad():
            nn.init.normal_(self.weights)

    def forward(self, inputs, p=None):
        B, _, H, W = inputs.shape

        x = F.unfold(inputs, self.kernel_size, self.dilation, self.padding, self.stride)
        x = x.transpose(1, 2).unsqueeze(1)              # Batch x 1   x H*W x In*Ker^2
        w = self.weights.flatten(-3)[None, :, None, :]  # 1     x Out x 1   x In*Ker^2

        from .core import Lp
        y = Lp(x, w, self.p if p is None else p)

        return y.view(B, -1,
            (H + 2 * self.padding - self.kernel_size) // self.stride + 1,
            (W + 2 * self.padding - self.kernel_size) // self.stride + 1)
