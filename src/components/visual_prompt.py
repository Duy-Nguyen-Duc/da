import torch
import torch.nn as nn
import torch.nn.functional as F


class AttributeNet(nn.Module):
    def __init__(self, layers=5, patch_size=8, out_channels=3, dropout_p=0.5):
        """
        Modified from paper: https://arxiv.org/abs/2406.03150
        """
        super(AttributeNet, self).__init__()
        self.layers = layers
        self.patch_size = patch_size
        self.out_channels = out_channels

        self.pooling = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 8, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout2d(p=dropout_p)

        self.conv2 = nn.Conv2d(8, 16, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout2d(p=dropout_p)

        self.conv3 = nn.Conv2d(16, 32, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU(inplace=True)
        self.dropout3 = nn.Dropout2d(p=dropout_p)

        self.conv4 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)
        self.dropout4 = nn.Dropout2d(p=dropout_p)

        last_channels = 64
        if self.layers == 6:
            self.conv5 = nn.Conv2d(64, 128, 3, 1, 1)
            self.bn5 = nn.BatchNorm2d(128)
            self.relu5 = nn.ReLU(inplace=True)
            self.dropout5 = nn.Dropout2d(p=dropout_p)
            last_channels = 128

        self.conv_out = nn.Conv2d(last_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.dropout1(y)
        if self.patch_size in [2, 4, 8, 16, 32]:
            y = self.pooling(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu2(y)
        y = self.dropout2(y)
        if self.patch_size in [4, 8, 16, 32]:
            y = self.pooling(y)

        y = self.conv3(y)
        y = self.bn3(y)
        y = self.relu3(y)
        y = self.dropout3(y)
        if self.patch_size in [8, 16, 32]:
            y = self.pooling(y)

        y = self.conv4(y)
        y = self.bn4(y)
        y = self.relu4(y)
        y = self.dropout4(y)
        if self.patch_size in [16, 32]:
            y = self.pooling(y)

        if self.layers == 6:
            y = self.conv5(y)
            y = self.bn5(y)
            y = self.relu5(y)
            y = self.dropout5(y)
            if self.patch_size == 32:
                y = self.pooling(y)

        y = self.conv_out(y)
        return y


class InstancewiseVisualPrompt(nn.Module):
    def __init__(self, size, layers=5, patch_size=8, channels=3, dropout_p=0.5, prompt_scale=0.5):
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

        self.priority = AttributeNet(layers=layers, patch_size=patch_size, out_channels=channels * 2, dropout_p=dropout_p)

        # Separate reprogram params
        self.program_fg = nn.Parameter(1e-4 * torch.randn(3, size, size))
        self.program_bg = nn.Parameter(1e-4 * torch.randn(3, size, size))
        self.prompt_scale = float(prompt_scale)

    def forward(self, x):
        B = x.shape[0]
        masks = self.priority(x)
        masks = masks.view(B, 2, self.channels, self.patch_num, self.patch_num)

        if self.channels == 1:
            masks = masks.expand(-1, -1, 3, -1, -1)

        masks_rgb = F.gumbel_softmax(masks, tau=1.0, hard=False, dim=1)
        masks_rgb = masks_rgb.reshape(B, 2, 3, self.patch_num, self.patch_num)

        masks = masks_rgb.repeat_interleave(self.patch_size, dim=-2).repeat_interleave(self.patch_size, dim=-1)

        attention_fg = masks[:, 0]  # (B, 3, H, W)
        attention_bg = masks[:, 1]  # (B, 3, H, W)
        program_fg = self.prompt_scale * torch.tanh(self.program_fg)
        program_bg = self.prompt_scale * torch.tanh(self.program_bg)
        return {
            "bg": x + attention_bg * program_bg,
            "fg": x + attention_fg * program_fg,
            "full": x + attention_bg * program_bg + attention_fg * program_fg,
            "bg_mask": attention_bg.mean(dim=1), 
            "fg_mask": attention_fg.mean(dim=1)
        } 