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
