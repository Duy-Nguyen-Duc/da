import os
import torch

from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from yacs.config import CfgNode as CN

from data.dataset import make_dataset
from eval import evaluate
from src.model import Model
from src.utils import clean_exp_savedir
from src.losses import supervised_loss


def run_bi_step(cfg: CN, exp_save_dir: str):
    #################################### DATA ####################################
    source_train_loader, _, source_test_loader, target_test_loader = (
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
    model = model.to(device)
    #################################### OPTIMIZER ####################################
    scaler = GradScaler('cuda')
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(model.classifier_head_src.parameters())
                + list(model.classifier_head_tgt.parameters()),
                "lr": cfg.optimizer.lr,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": list(model.visual_prompt_src.parameters())
                + list(model.visual_prompt_tgt.parameters()),
                "lr": cfg.optimizer_vr.lr,
                "weight_decay": cfg.optimizer_vr.weight_decay,
            },
        ]
    )

    epochs = cfg.burn_in.epochs
    total_steps = epochs * len(source_train_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
    #################################### MAIN LOOP ####################################
    writer = SummaryWriter(exp_save_dir)
    best_test_acc = 0
    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        pbar = tqdm(
            source_train_loader,
            total=len(source_train_loader),
            desc=f"Epoch {epoch + 1}",
            ncols=100,
        )

        for batch_idx, source_data in enumerate(pbar):
            pbar.set_description_str(f"Epoch {epoch + 1}", refresh=True)
            current_step = epoch * len(source_train_loader) + batch_idx
            # weak_img, strong_img, label
            weak_img, _, src_labels = source_data 

            weak_img = weak_img.to(device)
            src_labels = src_labels.to(device)
            optimizer.zero_grad()
            with autocast('cuda'):
                logit_s = model(weak_img, branch="src", region=["full"])
                loss = supervised_loss(logit_s['full'], src_labels)
                running_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            writer.add_scalar("Source/Train BatchLoss", loss.item(), current_step)
            writer.add_scalar(
                "Source/Running loss",
                running_loss / len(source_train_loader),
                current_step,
            )

        test_loss_src, test_accuracy_src = evaluate(
            model, branch="src", test_loader=source_test_loader, device=device
        )
        test_loss_tgt, test_accuracy_tgt = evaluate(
            model, branch="src", test_loader=target_test_loader, device=device
        )
        writer.add_scalar("Source/Test EpochLoss", test_loss_src, epoch)
        writer.add_scalar("Source/Test Accuracy", test_accuracy_src, epoch)

        writer.add_scalar("Target/Test EpochLoss", test_loss_tgt, epoch)
        writer.add_scalar("Target/Test Accuracy", test_accuracy_tgt, epoch)

        print(
            f"Epoch [{epoch + 1}/{epochs}] Test Loss Source: {test_loss_src:.4f}, Test Accuracy Source: {test_accuracy_src:.2f}%"
        )
        print(
            f"Epoch [{epoch + 1}/{epochs}] Test Loss Target: {test_loss_tgt:.4f}, Test Accuracy Target: {test_accuracy_tgt:.2f}%"
        )

        if test_accuracy_src > best_test_acc:
            best_test_acc = test_accuracy_src
            ckpt_path = os.path.join(
                exp_save_dir, f"bi_best_{test_accuracy_src:.2f}.pth"
            )
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
            if test_accuracy_src == 100:
                break
    clean_exp_savedir(exp_save_dir, ckpt_path, prefix="bi")
    return ckpt_path
