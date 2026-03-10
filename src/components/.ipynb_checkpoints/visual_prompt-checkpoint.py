import torch
import torch.nn as nn

from src.components.mask_generator import get_attr_net

class InstancewiseVisualPrompt(nn.Module):
    def __init__(self, size, layers=5, patch_size=8, channels=3, dropout_p=0.5, attr_net_type: str = "conv"):
        """
        Args:
            size: input image size (assumed square)
            layers: #layers of mask-training CNN
            patch_size: patch size for same mask value
            channels: 3 -> per-RGB mask, 1 -> shared mask across RGB
        """
        super(InstancewiseVisualPrompt, self).__init__()
        if layers not in [5, 6]:
            raise ValueError("Input layer number is not supported")
        if patch_size not in [1, 2, 4, 8, 16, 32]:
            raise ValueError("Input patch size is not supported")
        if channels not in [1, 3]:
            raise ValueError("Input channel number is not supported")
        if patch_size == 32 and layers != 6:
            raise ValueError("Input layer number and patch size are conflict with each other")

        self.patch_num = int(size / patch_size)
        self.imagesize = size
        self.patch_size = patch_size
        self.channels = channels

        self.priority = get_attr_net(layers=layers, patch_size=patch_size, out_channels=channels, dropout_p=dropout_p, type=attr_net_type)
        self.program = nn.Parameter(1e-4 * torch.randn(3, size, size))

    def forward(self, x):
        attention = (
            self.priority(x)
            .view(-1, self.channels, self.patch_num * self.patch_num, 1)
            .expand(-1, 3, -1, self.patch_size * self.patch_size)
            .view(
                -1, 3, self.patch_num, self.patch_num, self.patch_size, self.patch_size
            )
            .transpose(3, 4)
        )
        attention = attention.reshape(-1, 3, self.imagesize, self.imagesize)
        x = x + self.program * attention
        return x