import torch.nn as nn

from torchvision.models import (
    ResNet18_Weights,
    resnet18,
    ResNet50_Weights,
    resnet50,
    ResNet101_Weights,
    resnet101,
)
from pytorch_pretrained_vit import ViT

def make_backbone(backbone_name):
    if backbone_name == "resnet18":
        backbone = resnet18(ResNet18_Weights.IMAGENET1K_V1)
    elif backbone_name == "resnet50":
        backbone = resnet50(ResNet50_Weights.IMAGENET1K_V1)
    elif backbone_name == "resnet101":
        backbone = resnet101(ResNet101_Weights.IMAGENET1K_V1)
    elif backbone_name == "vit_b_16":
        backbone = ViT("B_32_imagenet1k", pretrained=True)
    elif backbone_name == "vit_b_32":
        backbone =  ViT("B_32_imagenet1k", pretrained=True)
    else:
        raise ValueError("Unsupported backbone architecture")
    return backbone


class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=2, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x): 
        return self.net(x)

def make_classifier_head(in_dim, hidden_dim, dropout, out_dim: int = None, type="domain"):
    if type == "domain":
        return DomainDiscriminator(in_dim=in_dim,hidden_dim=hidden_dim, dropout=dropout)
    elif type == "class":
        return Classifier(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, dropout=dropout)
    else:
        raise ValueError("Unsupported classifier head architecture")