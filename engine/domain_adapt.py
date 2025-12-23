import os
import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from yacs.config import CfgNode as CN

from data.dataset import make_dataset
from eval import evaluate
from src.model import Model
from src.utils import clean_exp_savedir, setup
from src.losses import (
    supervised_loss,  
    adversarial_loss, 
    js_divergence_from_logits,
    entropy_from_logits,
)

import argparse


def run_da_step(cfg: CN, exp_save_dir: str, best_bi_ckpt: str):
    #################################### DATA ####################################
    source_train_loader, target_train_loader, source_test_loader, target_test_loader = (
        make_dataset(
            source_dataset=cfg.dataset.source,
            target_dataset=cfg.dataset.target,
            img_size=cfg.img_size,
            train_bs=cfg.domain_adapt.train_bs,
            eval_bs=cfg.domain_adapt.eval_bs,
            num_workers=cfg.domain_adapt.num_workers,
        )
    )
    #################################### MODEL ####################################
    model = Model(
        backbone_type=cfg.model.backbone.type,
        in_dim=cfg.model.backbone.in_dim,
        hidden_dim=cfg.model.backbone.hidden_dim,
        out_dim=cfg.dataset.num_classes,
        imgsize=cfg.img_size,
        attribute_layers=cfg.model.attribute_layers,
        patch_size=cfg.model.patch_size,
    )
    device = torch.device(cfg.device)
    # Init with same weights
    ckpt = torch.load(best_bi_ckpt)
    model.load_state_dict(ckpt["model_state_dict"])
    with torch.no_grad():
        model.visual_prompt_tgt.load_state_dict(
            model.visual_prompt_src.state_dict(), strict=False
        )
        model.classifier_head_tgt.load_state_dict(
            model.classifier_head_src.state_dict(), strict=False
        )
    model = model.to(device)
    #################################### OPTIMIZER ####################################
    scaler = GradScaler("cuda")
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(model.classifier_head_src.parameters())
                + list(model.classifier_head_tgt.parameters()),
                "lr": cfg.optimizer.lr,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": model.discriminator.parameters(),
                "lr": cfg.optimizer_d.lr,
                "weight_decay": cfg.optimizer_d.weight_decay,
            },
            {
                "params": list(model.visual_prompt_src.parameters())
                + list(model.visual_prompt_tgt.parameters()),
                "lr": cfg.optimizer_vr.lr,
                "weight_decay": cfg.optimizer_vr.weight_decay,
            },
        ]
    )

    epochs = cfg.domain_adapt.epochs
    steps = min(len(source_train_loader), len(target_train_loader))
    total_steps = epochs * steps
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    #################################### MAIN LOOP ####################################
    writer = SummaryWriter(exp_save_dir)
    best_test_acc = 0
    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        pbar = tqdm(
            zip(source_train_loader, target_train_loader),
            total=steps,
            desc=f"Epoch {epoch + 1}",
            ncols=100,
        )
        for batch_idx, (source_data, target_data) in enumerate(pbar):
            pbar.set_description_str(f"Epoch {epoch + 1}", refresh=True)
            current_step = epoch * steps + batch_idx
            # gradient reversal layer alpha 
            grl_alpha = 2.0 / (1.0 + np.exp(-10 * (current_step / total_steps))) - 1.0
            # src_strong_aug, src_label
            _, src_strong_img, src_labels = source_data
            # tgt_weak_aug, tgt_strong_aug
            tgt_weak_img, tgt_strong_img, _ = target_data

            src_strong_img = src_strong_img.to(device)
            src_labels = src_labels.to(device)
            tgt_strong_img = tgt_strong_img.to(device)
            tgt_weak_img = tgt_weak_img.to(device)  # weak
            optimizer.zero_grad()
            with autocast('cuda'):
                # 1. Source cls loss
                logits_s = model.forward_sample(src_strong_img, branch="src", region=["bg","fg", "full"], return_mask=False)
                loss_cls = supervised_loss(logits_s['fg'], src_labels) + supervised_loss(logits_s['full'], src_labels)
                loss_src_div = js_divergence_from_logits(logits_s['fg'], logits_s['full'])
                loss_src_bg = -entropy_from_logits(logits_s['bg'])

                # 2. Self-supervised tgt loss
                logit_t_strong = model.forward_sample(tgt_strong_img, branch="tgt", region=['fg', 'full'], return_mask=False)
                logit_t_weak = model.forward_sample(tgt_weak_img,  branch="src", region=['bg','fg', 'full'], return_mask=False)

                with torch.no_grad():
                    pseudo_label = logit_t_weak.argmax(dim=1)
                
                loss_ssl = supervised_loss(logit_t_strong['full'], pseudo_label)
                loss_div = js_divergence_from_logits(logit_t_strong['fg'], logit_t_weak['fg'])
                loss_tgt_bg = -entropy_from_logits(logit_t_weak['bg'])

                # 3. Adv loss
                logit_s, logit_t = model.forward_adversarial(src_strong_img, tgt_strong_img, grl_alpha=grl_alpha, region=['fg'])
                loss_adv = adversarial_loss(logit_s['fg'], logit_t['fg'])
                loss = (
                    loss_cls
                    + cfg.alpha_div * (loss_div + loss_src_div)
                    + cfg.alpha_adv * loss_adv
                    + cfg.alpha_ssl * loss_ssl
                    + 0.1 * (loss_tgt_bg + loss_src_bg)
                )

            running_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Logging
            writer.add_scalar("DA/Train Cls loss", loss_cls.item(), current_step)
            writer.add_scalar("DA/Train Adv loss", loss_adv.item(), current_step)
            writer.add_scalar("DA/Train Ssl loss", loss_ssl.item(), current_step)
            writer.add_scalar("DA/Train Div loss", loss_div.item(), current_step)
            writer.add_scalar("DA/Train Src Div loss", loss_src_div.item(), current_step)
            writer.add_scalar("DA/Train BG tgt loss", loss_tgt_bg.item(), current_step)
            writer.add_scalar("DA/Train BG src loss", loss_src_bg.item(), current_step)
            writer.add_scalar("DA/Train BatchLoss", loss.item(), current_step)
        scheduler.step()
        test_loss_src, test_acc_src = evaluate(
            model, branch="src", test_loader=source_test_loader, device=device
        )
        test_loss_tgt, test_acc_tgt = evaluate(
            model, branch="tgt", test_loader=target_test_loader, device=device
        )

        writer.add_scalar("Source/Test EpochLoss", test_loss_src, epoch)
        writer.add_scalar("Source/Test Accuracy", test_acc_src, epoch)
        writer.add_scalar("Target/Test EpochLoss", test_loss_tgt, epoch)
        writer.add_scalar("Target/Test Accuracy", test_acc_tgt, epoch)
        writer.add_scalar(
            "DA/Epoch loss", running_loss / len(source_train_loader), epoch
        )

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Source Loss: {test_loss_src:.4f}, Source Acc: {test_acc_src:.2f}%"
        )
        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Target Loss: {test_loss_tgt:.4f}, Target Acc: {test_acc_tgt:.2f}%"
        )

        # Save the best model checkpoint (including optimizer, scheduler, scaler, etc.)
        if test_acc_tgt > best_test_acc:
            best_test_acc = test_acc_tgt
            ckpt_path = os.path.join(exp_save_dir, f"da_best_{test_acc_tgt:.2f}.pth")

            torch.save(
                {
                    "epoch": epoch,
                    "best_test_acc": best_test_acc,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                },
                ckpt_path,
            )

            print(f"New best checkpoint saved: {ckpt_path}")
    clean_exp_savedir(exp_save_dir, ckpt_path, prefix="da")
    return ckpt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the checkpoint file")
    args, _ = parser.parse_known_args()
    cfg = CN(new_allowed=True)
    cfg.merge_from_file(args.config)
    exp_save_dir = setup(cfg)

    print("Running DA step")
    best_ckpt = args.ckpt
    print("Loading best checkpoint from burn-in step:", best_ckpt)
    # Run domain adaptation step
    run_da_step(cfg, exp_save_dir=exp_save_dir, best_bi_ckpt=best_ckpt)
