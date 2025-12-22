import torch
import torch.nn as nn

from src.components.torch_nn import make_backbone, make_classifier_head
from src.components.mask_generator import InstancewiseVisualPrompt

from src.utils import freeze_layers, grad_reverse


class Model(nn.Module):
    def __init__(
        self,
        backbone_type="resnet18",
        in_dim=512,
        hidden_dim=256,
        out_dim=10,
        imgsize=64,
        attribute_layers=5,
        patch_size=8,
        attribute_channels=3,
        freeze_backbone=True
    ):
        super(Model, self).__init__()
        self.backbone = make_backbone(backbone_type)
        self.backbone.fc = nn.Identity()
        if freeze_backbone: 
            freeze_layers([self.backbone])
        
        self.visual_prompt_src = InstancewiseVisualPrompt(
            imgsize,
            attribute_layers,
            patch_size,
            attribute_channels,
        )
        self.visual_prompt_tgt = InstancewiseVisualPrompt(
            imgsize,
            attribute_layers,
            patch_size,
            attribute_channels,
        )
        self.classifier_head_src = make_classifier_head(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            dropout=0.3,
            type="class",
        )
        self.classifier_head_tgt = make_classifier_head(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            dropout=0.3,
            type="class",
        )
        self.discriminator = make_classifier_head(
            in_dim=in_dim, 
            hidden_dim=hidden_dim, 
            dropout=0.1, 
            type="domain",
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor=None, branch: str="src", region: str="fg", grl_alpha: float=None):
        if branch=="src":
            prompt, head = self.visual_prompt_src, self.classifier_head_src
            feat = self.backbone(prompt(x)[region])
            logit = head(feat)
            return logit
        elif branch=="tgt":
            prompt, head = self.visual_prompt_tgt, self.classifier_head_tgt
            feat = self.backbone(prompt(x)[region])
            logit = head(feat)
            return logit
        elif branch=="adversarial":
            assert y is not None, "cross domain images required"
            assert grl_alpha is not None, "grl alpha required"
            feat_s = self.backbone(self.visual_prompt_src(x)[region])
            feat_t = self.backbone(self.visual_prompt_tgt(y)[region])
            logit_s = self.discriminator(grad_reverse(feat_s, grl_alpha))
            logit_t = self.discriminator(grad_reverse(feat_t, grl_alpha))
            return logit_s, logit_t
        else: 
            raise ValueError(f"Unknown branch {branch}")
    
    def test(self, x, branch: str="src"):
        prompt, head = (self.visual_prompt_src, self.classifier_head_src) if branch=="src" else (self.visual_prompt_tgt, self.classifier_head_tgt)
        reprog_img = prompt(x)['reprog_img']
        feat = self.backbone(reprog_img)
        logit = head(feat)
        return logit
