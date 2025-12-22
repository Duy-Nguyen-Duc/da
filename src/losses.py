import torch 
# import torch.nn as nn
import torch.nn.functional as F

def supervised_loss(logit: torch.Tensor, label: torch.Tensor):
    return F.cross_entropy(logit, label)

def kl_divergene_loss(logit_p: torch.Tensor, logit_q: torch.Tensor):
    logit_p = logit_p.clamp(1e-12)
    logit_q = logit_q.clamp(1e-12)
    return F.kl_div(
        logit_p.log(), logit_q.detach(), reduction="none"
    ).sum(dim=1)

def adverarial_loss(logit_s, logit_t):
    # logit_s = grad_reverse(f_backbone(x_s))
    # logit_t = grad_reverse(f_backbone(x_t))
    label_s= torch.zeros(logit_s.shape[0], dtype=torch.long, device=logit_s.device)
    label_t = torch.ones(logit_t.shape[0], dtype=torch.long, device=logit_t.device)
    return supervised_loss(logit_s, label_s).mean() + supervised_loss(logit_t, label_t).mean()

def fg_bg_loss(feat):
    # To be implemented
    return