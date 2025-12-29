import torch
import torch.nn.functional as F

def supervised_loss(logits: torch.Tensor, labels: torch.Tensor, reduction="mean"):
    return F.cross_entropy(logits, labels, reduction=reduction)

def kl_divergence_loss(logits_p: torch.Tensor, logits_q: torch.Tensor, detach_q=True, reduction="batchmean"):
    """
    KL( p || q ) where p,q are categorical distributions from logits.
    """
    log_p = F.log_softmax(logits_p, dim=-1)
    q = F.softmax(logits_q.detach() if detach_q else logits_q, dim=-1)
    return F.kl_div(log_p, q, reduction=reduction)

def symmetric_kl_loss(logits_a: torch.Tensor, logits_b: torch.Tensor, reduction="batchmean"):
    return kl_divergence_loss(logits_a, logits_b, detach_q=False, reduction=reduction) + \
           kl_divergence_loss(logits_b, logits_a, detach_q=False, reduction=reduction)

def max_entropy_loss_from_logits(logits: torch.Tensor):
    """
    Minimizing this = maximizing entropy => makes predictions less confident.
    Useful for "bg-only should be uninformative".
    """
    p = F.softmax(logits, dim=-1)
    ent = -(p * (p + 1e-12).log()).sum(dim=-1)  # (B,)
    return -ent.mean()

def adversarial_loss(domain_logits_s: torch.Tensor, domain_logits_t: torch.Tensor):
    label_s = torch.zeros(domain_logits_s.shape[0], dtype=torch.long, device=domain_logits_s.device)
    label_t = torch.ones(domain_logits_t.shape[0], dtype=torch.long, device=domain_logits_t.device)
    return F.cross_entropy(domain_logits_s, label_s) + F.cross_entropy(domain_logits_t, label_t)

def masked_ce_loss(logits: torch.Tensor, labels: torch.Tensor, keep: torch.Tensor):
    """
    keep: bool tensor (B,) indicating which samples to include.
    """
    if keep is None:
        return F.cross_entropy(logits, labels)
    keep = keep.bool()
    if keep.sum().item() == 0:
        return logits.sum() * 0.0  # safe zero
    loss = F.cross_entropy(logits, labels, reduction="none")
    return (loss[keep]).mean()

def right_reasons_grad_penalty(
    logits_full: torch.Tensor,
    labels: torch.Tensor,
    reprog_full_img: torch.Tensor,
    mask_bg_1c: torch.Tensor,
) -> torch.Tensor:
    """
    Ross et al. style: penalize input-gradients in background. 
    This is 2nd-order (create_graph=True), so keep weight small or run intermittently.
    """
    logp = F.log_softmax(logits_full, dim=1)
    score = logp.gather(1, labels.view(-1, 1)).sum()
    grads = torch.autograd.grad(
        outputs=score,
        inputs=reprog_full_img,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]  # (B,3,H,W)

    g = grads.abs().mean(dim=1, keepdim=True)
    return (mask_bg_1c * (g ** 2)).mean()

def js_divergence_from_logits(p_logits, q_logits, detach_q=True):
    if detach_q:
        q_logits = q_logits.detach()
    p = F.softmax(p_logits, dim=1)
    q = F.softmax(q_logits, dim=1)
    m = 0.5 * (p + q)
    js = 0.5 * (
        F.kl_div(torch.log(p + 1e-8), m, reduction="batchmean") +
        F.kl_div(torch.log(q + 1e-8), m, reduction="batchmean")
    )
    return js

def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(logits, dim=1)
    return -(p * torch.log(p + 1e-8)).sum(dim=1).mean()

def mask_binarize_loss(mask_fg_1c: torch.Tensor) -> torch.Tensor:
    return (mask_fg_1c * (1.0 - mask_fg_1c)).mean()

def mask_area_range_loss(mask_fg_1c, rho_min=0.20, rho_max=0.85):
    m = mask_fg_1c.mean()
    return F.relu(rho_min - m) + F.relu(m - rho_max)

def total_variation(x: torch.Tensor) -> torch.Tensor:
    # x: (B,C,H,W) or (C,H,W)
    if x.dim() == 3:
        x = x.unsqueeze(0)
    tv_h = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    tv_w = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    return tv_h + tv_w