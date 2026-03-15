import torch
import torch.nn as nn

from src.components.mask_generator import get_attr_net

class MultiHeadVisualPrompt(nn.Module):
    def __init__(self, 
        imgsize: int=224, 
        layers: list[int]=[5,6,5,6], 
        patch_size: list[int]=[4,8,16,32], 
        channels=3, 
        dropout: list[float]=[0.1,0.1,0.2,0.2], 
        attr_net: list[str] = ["conv", "conv", "transformer", "transformer"]
    ):
        """
        Args:
            size: input image size (assumed square)
            layers: #layers of mask-training CNN
            patch_size: patch size for same mask value
            channels: 3 -> per-RGB mask, 1 -> shared mask across RGB
        """
        super(MultiHeadVisualPrompt, self).__init__()
        assert len(layers) == len(patch_size) == len(dropout) == len(attr_net)

        self.num_head = len(layers)
        self.imagesize = imgsize
        self._patch_size = patch_size
        self._patch_num = [imgsize//patch_size[i] for i in range(self.num_head)]
        self.channels = channels
        self._attr_net = attr_net
        self.priority = nn.ModuleList(
            get_attr_net(
                layers=layers[i],
                patch_size=patch_size[i],
                out_channels=channels,
                dropout_p=dropout[i],
                type=attr_net[i]
            )
            for i in range(self.num_head)
        )
        self.program = nn.Parameter(1e-4 * torch.randn(self.num_head, 3, self.imagesize, self.imagesize))
    
    def _forward_head(self, x: torch.Tensor, idx: int):
        attention = (
            self.priority[idx](x)
            .view(-1, self.channels, self._patch_num[idx] ** 2, 1)
            .expand(-1, 3, -1, self._patch_size[idx] ** 2)
            .view(
                -1, 3, self._patch_num[idx], self._patch_num[idx], self._patch_size[idx], self._patch_size[idx]
            )
            .transpose(3, 4)
        ).reshape(-1, 3, self.imagesize, self.imagesize).contiguous()

        return x + self.program[idx] * attention 


    def forward(self, x: torch.Tensor, use_head: int | None = None):
        """
        x: Input tensor
        use_head: The id of head being use. If not stated, infer all heads
        """
        head_idx = [i for i in range(self.num_head)] if use_head is None else [use_head]
        out = {}
        for idx in head_idx:
            out[f"{self._attr_net[idx]}_{idx}"] = self._forward_head(x, idx)
        return out