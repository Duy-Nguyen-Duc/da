import torch
import torch.nn as nn

from src.components.torch_nn import make_backbone, make_classifier_head
from src.components.visual_prompt import InstancewiseVisualPrompt

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
            dropout=0.1,
            type="class",
        )
        self.classifier_head_tgt = make_classifier_head(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            dropout=0.1,
            type="class",
        )
        self.discriminator = make_classifier_head(
            in_dim=in_dim, 
            hidden_dim=hidden_dim, 
            dropout=0.1, 
            type="domain",
        )
    def forward_adversarial(self, x_s: torch.Tensor, x_t: torch.Tensor, region: list[str]=["fg"], grl_alpha=1.0):    
        src_out, tgt_out = {}, {}
        ps = self.visual_prompt_src(x_s)
        pt = self.visual_prompt_tgt(x_t)
    
        for re in region:
            feat_s = self.backbone(ps[re])
            feat_t = self.backbone(pt[re])
            src_out[re] = self.discriminator(grad_reverse(feat_s, grl_alpha))
            tgt_out[re] = self.discriminator(grad_reverse(feat_t, grl_alpha))
        return src_out, tgt_out


    def forward_sample(self, x: torch.Tensor, region: list[str]=["fg"], branch: str="src", return_all=False):
        out = {}
        prompt = self.visual_prompt_src if branch=="src" else self.visual_prompt_tgt
        head = self.classifier_head_src if branch=="src" else self.classifier_head_tgt
        reprog_imgs = prompt(x)
        for re in region:
            feat = self.backbone(reprog_imgs[re])
            out[re] = head(feat)
        
        if return_all: 
            out['bg_mask'] = reprog_imgs['bg_mask']
            out['fg_mask'] = reprog_imgs['fg_mask']
            out['full_img'] = reprog_imgs['full']
        return out
    
    def test(self, x, branch: str="src"):
        prompt = self.visual_prompt_src if branch=="src" else self.visual_prompt_tgt
        head = self.classifier_head_src if branch=="src" else self.classifier_head_tgt
        reprog_img = prompt(x)['full']
        feat = self.backbone(reprog_img)
        logit = head(feat)
        return logit
