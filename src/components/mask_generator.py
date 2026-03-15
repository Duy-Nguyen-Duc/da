import torch
import torch.nn as nn

from src.components.transformer import PatchEmbed, TransformerEncoderBlock

class ConvNet(nn.Module):
    def __init__(self, layers=5, patch_size=8, channels=3, dropout_p=0.5):
        """
        Paper: https://arxiv.org/abs/2406.03150
        """

        super(ConvNet, self).__init__()
        self.layers = layers
        self.patch_size = patch_size
        self.channels = channels

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
        if self.layers == 5 and self.channels == 3:
            self.conv6 = nn.Conv2d(64, 3, 3, 1, 1)
        elif self.layers == 6:
            self.conv5 = nn.Conv2d(64, 128, 3, 1, 1)
            self.bn5 = nn.BatchNorm2d(128)
            self.relu5 = nn.ReLU(inplace=True)
            self.dropout5 = nn.Dropout2d(p=dropout_p)

            if self.channels == 3:
                self.conv6 = nn.Conv2d(128, 3, 3, 1, 1)

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

        if self.channels == 3:
            y = self.conv6(y)
        elif self.channels == 1:
            y = torch.mean(y, dim=1)
        return y

class TransformerNet(nn.Module):
    def __init__(
        self,
        layers: int = 5,
        patch_size: int = 8,
        out_channels: int = 3,
        dropout_p: float = 0.1,
        embed_dim: int = 128,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
    ):
        super(TransformerNet, self).__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.patch_embed = PatchEmbed(patch_size, in_channels=3, embed_dim=embed_dim)
        self.pos_embed = None
        self.embed_dim = embed_dim

        self.blocks = nn.Sequential(*[
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout_p)
            for _ in range(layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(embed_dim, out_channels)

    def _get_pos_embed(self, Hp: int, Wp: int, device: torch.device) -> torch.Tensor:
        if self.pos_embed is None or self.pos_embed.shape[1] != Hp * Wp:
            y_pos = torch.arange(Hp, device=device).unsqueeze(1).float()
            x_pos = torch.arange(Wp, device=device).unsqueeze(0).float()
            dim = self.embed_dim
            div = torch.exp(
                torch.arange(0, dim, 2, device=device).float()
                * -(torch.log(torch.tensor(10000.0)) / dim)
            )
            pe = torch.zeros(Hp, Wp, dim, device=device)
            pe[:, :, 0::2] = torch.sin(x_pos.unsqueeze(-1) * div)
            pe[:, :, 1::2] = torch.cos(y_pos.unsqueeze(-1) * div[:dim // 2])
            self.pos_embed = pe.view(1, Hp * Wp, dim)
        return self.pos_embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape

        tokens, Hp, Wp = self.patch_embed(x) # [B, N, D]
        tokens = tokens + self._get_pos_embed(Hp, Wp, x.device)
        tokens = self.blocks(tokens) # [B, N, D]
        tokens = self.norm(tokens)

        out = self.head(tokens)
        out = out.transpose(1, 2) 
        out = out.contiguous().view(B, self.out_channels, Hp, Wp)
        return out

def get_attr_net(layers: int = 5, patch_size: int = 8, out_channels: int = 3, dropout_p: float = 0.5, type: str = "conv"):
    if type == "conv":
        return ConvNet(layers, patch_size, out_channels, dropout_p)
    elif type == "transformer":
        return TransformerNet(layers, patch_size, out_channels, dropout_p)
    else:
        raise ValueError("Unsupported attribute network type")