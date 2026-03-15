import torch
import torch.nn as nn

from src.components.torch_nn import make_backbone, make_classifier_head
from src.components.visual_prompt import MultiHeadVisualPrompt

from src.utils import freeze_layers, grad_reverse


class SingleModel(nn.Module):
    def __init__(
        self,
        backbone_type:str="vit_b_16",
        in_dim:int=768,
        hidden_dim:int=256,
        out_dim:int=31,
        imgsize:int=64,
        attribute_layers=5,
        patch_size=8,
        attribute_channels=3,
        freeze_backbone=True, 
        attr_net_type="conv"
    ):
        super(SingleModel, self).__init__()
        self.backbone = make_backbone(backbone_type)
        self.backbone.fc = nn.Identity()
        if freeze_backbone: 
            freeze_layers([self.backbone])
        
        self.visual_prompt_src = MultiHeadVisualPrompt(size=imgsize, layers=attribute_layers, patch_size=patch_size, channels=attribute_channels, dropout_p=0.2, attr_net_type=attr_net_type)
        self.visual_prompt_tgt = MultiHeadVisualPrompt(size=imgsize, layers=attribute_layers, patch_size=patch_size, channels=attribute_channels, dropout_p=0.2, attr_net_type=attr_net_type)
        self.classifier_head_src = make_classifier_head(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, dropout=0.1, type="class")
        self.classifier_head_tgt = make_classifier_head(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, dropout=0.1, type="class")
        self.discriminator = make_classifier_head(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=2, dropout=0.1, type="domain")

    def forward(self, x: torch.Tensor, branch: str="src"):
        prompt, head = (self.visual_prompt_src, self.classifier_head_src) if branch=="src" else (self.visual_prompt_tgt, self.classifier_head_tgt)
        feat = self.backbone(prompt(x))
        logit = head(feat)
        return logit
    
    def forward_adversarial(self, x: torch.Tensor, y: torch.Tensor, grl_alpha: float):
        feat_s = self.backbone(self.visual_prompt_src(x))
        feat_t = self.backbone(self.visual_prompt_tgt(y))
        logit_s = self.discriminator(grad_reverse(feat_s, grl_alpha))
        logit_t = self.discriminator(grad_reverse(feat_t, grl_alpha))
        return logit_s, logit_t
    
