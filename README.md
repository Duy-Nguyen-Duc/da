#  NeuroVR: Domain Adapt with Neuro-Visual Reprogramming

## 📒 Abstract

![Main model](assets/model.png)

## 💡 Preparation

Run these installations to download the datasets: Office-Home and Office-31. The Digits dataset is included within the Torchvision library.

```bash
bash datasets.sh
````

## 🔥 Get Start

Train the model from scratch with the default settings:
1. Train with ResNet config: 
```bash
python train.py --configs/office_31/resnet50/a2d.yaml
```
2. Train with Vit config: 
```bash
python train.py --configs/office_31/vit_b_32/a2d.yaml
```

Train the model for domain adaptation starting from a well-trained source model:

```bash
python domain_adapt.py --config=configs/office_31/a2d.yaml --ckpt=path/to/checkpoints
```

Evaluate the model:

```bash
python eval.py --config=configs/office_31/resnet50/a2d.yaml --ckpt=checkpoints/office_31/resnet50/da_best...pth
```

## 📦 Well-Trained Models

We saved our checkpoints on HuggingFace at this [repo](https://huggingface.co/G7xHp2Qv/ViRDA) and thus can be downloaded via the prepared script. For each task, we provide both the domain-adaptation best checkpoints for ViT and Resnet backbone. Please run the following command to get all the well-trained models: 

```bash
python download_all_checkpoints.py
```

## Citation 
If you use VirDA in your research or wish to refer to the results published in the paper, please use the following BibTeX entry.
```bash
@article{
nguyen2025virda,
title={Vir{DA}: Reusing Backbone for Unsupervised Domain Adaptation with Visual Reprogramming},
author={Duc-Duy Nguyen and Dat Nguyen},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=Qh7or7JRFI},
note={}
}
```
